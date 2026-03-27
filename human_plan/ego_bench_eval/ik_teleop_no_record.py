import argparse
import random
import time

import cv2
import numpy as np
import torch

from omni.isaac.lab.app import AppLauncher

SEED_MAP = {
    "Humanoid-Pour-Balls-v0": 0,
    "Humanoid-Sort-Cans-v0": 1,
    "Humanoid-Insert-Cans-v0": 2,
    "Humanoid-Unload-Cans-v0": 3,
    "Humanoid-Insert-And-Unload-Cans-v0": 4,
    "Humanoid-Push-Box-v0": 5,
    "Humanoid-Open-Drawer-v0": 6,
    "Humanoid-Close-Drawer-v0": 7,
    "Humanoid-Open-Laptop-v0": 8,
    "Humanoid-Flip-Mug-v0": 9,
    "Humanoid-Stack-Can-v0": 10,
    "Humanoid-Stack-Can-Into-Drawer-v0": 11,
}


def quat_wxyz_normalize(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_wxyz_rotate_vec(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    q = quat_wxyz_normalize(q_wxyz)
    q_xyz = q[1:4]
    q_w = float(q[0])
    v = np.asarray(v_xyz, dtype=np.float64)
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def ee_link_pose_to_tcp_pose(ee_pose_wxyz: np.ndarray, tcp_offset_xyz: tuple[float, float, float]) -> np.ndarray:
    pose = np.asarray(ee_pose_wxyz, dtype=np.float64).copy()
    pose[3:7] = quat_wxyz_normalize(pose[3:7])
    offset_world = quat_wxyz_rotate_vec(pose[3:7], np.asarray(tcp_offset_xyz, dtype=np.float64))
    pose[:3] = pose[:3] + offset_world
    return pose


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    out = image
    if out.dtype != np.uint8:
        out = np.asarray(out, dtype=np.float32)
        if out.max() <= 1.0:
            out = out * 255.0
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out


def overlay_info(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    vis = frame_rgb.copy()
    y = 24
    for line in lines:
        cv2.putText(
            vis,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 64, 64),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return vis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Init-only scene checker: keep USD reset pose (no warmup, no teleop)."
    )
    parser.add_argument("--task", type=str, default="Humanoid-Push-Box-v0")
    parser.add_argument("--room_idx", type=int, default=1)
    parser.add_argument("--table_idx", type=int, default=2)
    parser.add_argument("--env_randomize", type=int, default=0, help="0=fixed scene, 1=randomized scene")
    parser.add_argument("--spawn_background", type=int, default=1)

    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--loop_hz", type=float, default=30.0)
    parser.add_argument("--freeze_after_reset", type=int, default=1, help="1=hold reset frame, 0=step with zero action")

    parser.add_argument("--left_ee_body_name", type=str, default="left_arm_link7")
    parser.add_argument("--right_ee_body_name", type=str, default="right_arm_link7")

    parser.add_argument("--ik_tcp_offset_enable", type=int, default=1)
    parser.add_argument("--ik_left_tcp_offset_x", type=float, default=0.0)
    parser.add_argument("--ik_left_tcp_offset_y", type=float, default=0.0)
    parser.add_argument("--ik_left_tcp_offset_z", type=float, default=0.2104)
    parser.add_argument("--ik_right_tcp_offset_x", type=float, default=0.0)
    parser.add_argument("--ik_right_tcp_offset_y", type=float, default=0.0)
    parser.add_argument("--ik_right_tcp_offset_z", type=float, default=0.2104)

    parser.add_argument("--robot_base_x", type=float, default=None)
    parser.add_argument("--robot_base_y", type=float, default=None)
    parser.add_argument("--robot_base_z", type=float, default=None)
    parser.add_argument("--box_init_x", type=float, default=None)
    parser.add_argument("--box_init_y", type=float, default=None)
    parser.add_argument("--box_init_z", type=float, default=None)
    parser.add_argument("--goal_x", type=float, default=None)
    parser.add_argument("--goal_y", type=float, default=None)
    parser.add_argument("--goal_z", type=float, default=None)

    parser.add_argument("--use_benchmark_random_idx", type=int, default=1)
    parser.add_argument("--bench_num_episodes", type=int, default=3)
    parser.add_argument("--bench_num_trials", type=int, default=1)
    parser.add_argument("--bench_episode_rank", type=int, default=0)
    parser.add_argument("--bench_trial_idx", type=int, default=0)

    # Compatibility args (not used in this init-only script).
    parser.add_argument("--start_mode", type=str, default="reset")
    parser.add_argument("--preserve_usd_initial_ee", type=int, default=1)
    parser.add_argument("--right_only_mode", type=int, default=1)
    parser.add_argument("--ik_active_arm", type=str, default="right")
    parser.add_argument("--ik_ignore_orientation", type=int, default=1)
    return parser


def apply_env_overrides(env_cfg, args):
    left_name = str(args.left_ee_body_name).strip()
    right_name = str(args.right_ee_body_name).strip()
    if left_name:
        env_cfg.left_arm_cfg.body_names = [left_name]
    if right_name:
        env_cfg.right_arm_cfg.body_names = [right_name]

    if any("hand_tcp" in str(n).lower() for n in env_cfg.left_arm_cfg.body_names):
        print("[InitOnly] left *_hand_tcp is non-rigid; fallback to left_arm_link7.")
        env_cfg.left_arm_cfg.body_names = ["left_arm_link7"]
    if any("hand_tcp" in str(n).lower() for n in env_cfg.right_arm_cfg.body_names):
        print("[InitOnly] right *_hand_tcp is non-rigid; fallback to right_arm_link7.")
        env_cfg.right_arm_cfg.body_names = ["right_arm_link7"]

    if any(v is not None for v in (args.robot_base_x, args.robot_base_y, args.robot_base_z)):
        default_robot_pos = list(env_cfg.robot.init_state.pos)
        env_cfg.robot.init_state.pos = (
            args.robot_base_x if args.robot_base_x is not None else default_robot_pos[0],
            args.robot_base_y if args.robot_base_y is not None else default_robot_pos[1],
            args.robot_base_z if args.robot_base_z is not None else default_robot_pos[2],
        )
        print(f"[InitOnly] robot base override: {env_cfg.robot.init_state.pos}")

    if hasattr(env_cfg, "box") and any(v is not None for v in (args.box_init_x, args.box_init_y, args.box_init_z)):
        default_box_pos = list(env_cfg.box.init_state.pos)
        env_cfg.box.init_state.pos = (
            args.box_init_x if args.box_init_x is not None else default_box_pos[0],
            args.box_init_y if args.box_init_y is not None else default_box_pos[1],
            args.box_init_z if args.box_init_z is not None else default_box_pos[2],
        )
        print(f"[InitOnly] box init override: {env_cfg.box.init_state.pos}")

    if hasattr(env_cfg, "goal_default_pos") and any(v is not None for v in (args.goal_x, args.goal_y, args.goal_z)):
        default_goal_pos = list(env_cfg.goal_default_pos)
        env_cfg.goal_default_pos = (
            args.goal_x if args.goal_x is not None else default_goal_pos[0],
            args.goal_y if args.goal_y is not None else default_goal_pos[1],
            args.goal_z if args.goal_z is not None else default_goal_pos[2],
        )
        print(f"[InitOnly] goal override: {env_cfg.goal_default_pos}")


def maybe_apply_benchmark_randomize_idx(env, args):
    if not bool(args.env_randomize):
        return
    if not bool(args.use_benchmark_random_idx):
        return
    if args.task not in SEED_MAP:
        print(f"[InitOnly] task not in SEED_MAP, skip benchmark randomize idx: {args.task}")
        return

    random.seed(SEED_MAP[args.task])
    randomize_idxes = list(range(10000))
    random.shuffle(randomize_idxes)

    bench_num_episodes = max(1, int(args.bench_num_episodes))
    bench_num_trials = max(1, int(args.bench_num_trials))
    bench_episode_rank = max(0, min(int(args.bench_episode_rank), bench_num_episodes - 1))
    bench_trial_idx = max(0, min(int(args.bench_trial_idx), bench_num_trials - 1))

    base = 100 + (int(args.room_idx) * 5 + int(args.table_idx)) * bench_num_trials * bench_num_episodes
    offset = 1 + bench_episode_rank * bench_num_trials + bench_trial_idx
    raw_idx = (base + offset) % len(randomize_idxes)
    env.cfg.randomize_idx = randomize_idxes[raw_idx]
    print(f"[InitOnly] benchmark-like randomize_idx={env.cfg.randomize_idx}")


def print_init_state(env, env_results, args):
    left_link = env_results[0]["left_ee_pose"][0].detach().cpu().numpy().copy()
    right_link = env_results[0]["right_ee_pose"][0].detach().cpu().numpy().copy()
    left_link[3:7] = quat_wxyz_normalize(left_link[3:7])
    right_link[3:7] = quat_wxyz_normalize(right_link[3:7])

    left_tcp_offset = (args.ik_left_tcp_offset_x, args.ik_left_tcp_offset_y, args.ik_left_tcp_offset_z)
    right_tcp_offset = (args.ik_right_tcp_offset_x, args.ik_right_tcp_offset_y, args.ik_right_tcp_offset_z)
    if bool(args.ik_tcp_offset_enable):
        left_tcp = ee_link_pose_to_tcp_pose(left_link, left_tcp_offset)
        right_tcp = ee_link_pose_to_tcp_pose(right_link, right_tcp_offset)
    else:
        left_tcp = left_link.copy()
        right_tcp = right_link.copy()

    print("\n[InitOnly] -------- RESET STATE (NO WARMUP / NO TELEOP) --------")
    print(f"[InitOnly] left_ee_link xyz=({left_link[0]:.4f}, {left_link[1]:.4f}, {left_link[2]:.4f})")
    print(f"[InitOnly] right_ee_link xyz=({right_link[0]:.4f}, {right_link[1]:.4f}, {right_link[2]:.4f})")
    print(f"[InitOnly] left_target xyz=({left_tcp[0]:.4f}, {left_tcp[1]:.4f}, {left_tcp[2]:.4f})")
    print(f"[InitOnly] right_target xyz=({right_tcp[0]:.4f}, {right_tcp[1]:.4f}, {right_tcp[2]:.4f})")

    if "object_pose" in env_results[0]:
        box = env_results[0]["object_pose"][0].detach().cpu().numpy().copy()
        print(f"[InitOnly] box xyz=({box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f})")

    if hasattr(env, "goal_pos_w"):
        goal = (env.goal_pos_w[0].detach().cpu() - env.scene.env_origins[0].detach().cpu()).numpy()
        print(f"[InitOnly] goal xyz=({goal[0]:.4f}, {goal[1]:.4f}, {goal[2]:.4f})")

    try:
        root_pos = env.robot.data.root_pos_w[0].detach().cpu().numpy().copy()
        print(f"[InitOnly] robot root xyz=({root_pos[0]:.4f}, {root_pos[1]:.4f}, {root_pos[2]:.4f})")
    except Exception:
        pass

    try:
        q = env.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        left_ids = [int(x) for x in env.cfg.left_arm_cfg.joint_ids]
        right_ids = [int(x) for x in env.cfg.right_arm_cfg.joint_ids]
        left_q = q[left_ids]
        right_q = q[right_ids]
        print(f"[InitOnly] left_arm_q(first7)={np.array2string(left_q[:7], precision=4, separator=',')}")
        print(f"[InitOnly] right_arm_q(first7)={np.array2string(right_q[:7], precision=4, separator=',')}")
    except Exception as e:
        print(f"[InitOnly] joint qpos print skipped: {e}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(enable_cameras=True, device="cuda", headless=bool(args.headless))
    simulation_app = app_launcher.app

    import gymnasium as gym
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from humanoid.tasks.base_env import BaseEnv, BaseEnvCfg

    env_cfg: BaseEnvCfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.randomize = bool(args.env_randomize)
    env_cfg.spawn_background = bool(args.spawn_background)
    env_cfg.room_idx = int(args.room_idx)
    env_cfg.table_idx = int(args.table_idx)
    env_cfg.episode_length_s = 120

    apply_env_overrides(env_cfg, args)

    env: BaseEnv = gym.make(args.task, cfg=env_cfg)
    maybe_apply_benchmark_randomize_idx(env, args)

    print("[InitOnly] preserve_usd_initial_ee=1 behavior is enforced in this script.")
    print("[InitOnly] NO init_poses warmup, NO IK tracking, NO teleop deltas.")

    env_results = env.reset()
    print_init_state(env, env_results, args)

    dt = 1.0 / max(1e-3, float(args.loop_hz))
    zero_action = torch.zeros((env.scene.num_envs, env.num_actions), device=env.robot.device)

    window_name = "Init-Only Fixed RGB"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[InitOnly] Ready. Q/Esc quit, N reset scene.")
    while simulation_app.is_running():
        tic = time.time()

        rgb_obs = env_results[0]["fixed_rgb"][0].detach().cpu().numpy()
        rgb_obs = to_uint8_rgb(rgb_obs)

        lines = [
            "Init-only mode: USD reset pose kept",
            "No warmup / no teleop arm control",
            "Keys: N reset | Q quit",
            f"freeze_after_reset={int(bool(args.freeze_after_reset))}",
        ]
        vis = overlay_info(rgb_obs, lines)
        cv2.imshow(window_name, vis[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            env_results = env.reset()
            print_init_state(env, env_results, args)
            continue

        if bool(args.freeze_after_reset):
            simulation_app.update()
        else:
            env_results = env.step(zero_action)

        spent = time.time() - tic
        if spent < dt:
            time.sleep(dt - spent)

    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
