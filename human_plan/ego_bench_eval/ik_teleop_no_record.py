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
        description="No-warmup teleop checker with optional right-arm joint control."
    )
    parser.add_argument("--task", type=str, default="Humanoid-Push-Box-v0")
    parser.add_argument("--room_idx", type=int, default=1)
    parser.add_argument("--table_idx", type=int, default=2)
    parser.add_argument("--env_randomize", type=int, default=0, help="0=fixed scene, 1=randomized scene")
    parser.add_argument("--spawn_background", type=int, default=1)

    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--loop_hz", type=float, default=30.0)
    parser.add_argument("--freeze_after_reset", type=int, default=1, help="1=hold reset frame, 0=step with zero action")
    parser.add_argument("--enable_right_joint_teleop", type=int, default=1, help="1=enable keyboard teleop for right arm joint targets")
    parser.add_argument("--right_joint_step", type=float, default=0.02, help="right-arm joint increment in rad")
    parser.add_argument("--right_joint_clip_to_limits", type=int, default=1, help="clip right joint target to dof limits")
    parser.add_argument("--pose_pos_step", type=float, default=0.006, help="right tcp xyz increment in meters for pose mode")

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
    # Align teleop default box pose with current benchmark reset scene.
    parser.add_argument("--box_init_x", type=float, default=0.4283)
    parser.add_argument("--box_init_y", type=float, default=-0.3960)
    parser.add_argument("--box_init_z", type=float, default=1.0400)
    # Align teleop default goal pose with current benchmark reset scene.
    parser.add_argument("--goal_x", type=float, default=0.5288)
    parser.add_argument("--goal_y", type=float, default=0.0717)
    parser.add_argument("--goal_z", type=float, default=1.0210)

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
    from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from human_plan.ego_bench_eval.utils import ik_step
    from humanoid.tasks.base_env import BaseEnv, BaseEnvCfg

    env_cfg: BaseEnvCfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.randomize = bool(args.env_randomize)
    env_cfg.spawn_background = bool(args.spawn_background)
    env_cfg.room_idx = int(args.room_idx)
    env_cfg.table_idx = int(args.table_idx)
    env_cfg.episode_length_s = 1200

    apply_env_overrides(env_cfg, args)

    env: BaseEnv = gym.make(args.task, cfg=env_cfg)
    maybe_apply_benchmark_randomize_idx(env, args)

    print("[Teleop] preserve_usd_initial_ee=1 behavior is enforced in this script.")
    print("[Teleop] NO init_poses warmup, NO IK tracking.")

    env_results = env.reset()
    print_init_state(env, env_results, args)

    left_tcp_offset = (args.ik_left_tcp_offset_x, args.ik_left_tcp_offset_y, args.ik_left_tcp_offset_z)
    right_tcp_offset = (args.ik_right_tcp_offset_x, args.ik_right_tcp_offset_y, args.ik_right_tcp_offset_z)
    ik_active_arm = str(args.ik_active_arm).strip().lower()
    if bool(args.right_only_mode):
        ik_active_arm = "right"
    if ik_active_arm not in ("both", "left", "right"):
        ik_active_arm = "right"

    dt = 1.0 / max(1e-3, float(args.loop_hz))
    zero_action = torch.zeros((env.scene.num_envs, env.num_actions), device=env.robot.device)
    action_target = torch.zeros((env.scene.num_envs, env.num_actions), device=env.robot.device)
    dof_lower = env.robot_dof_lower_limits
    dof_upper = env.robot_dof_upper_limits
    left_arm_joint_ids = [int(x) for x in env.cfg.left_arm_cfg.joint_ids]
    left_hand_joint_ids = [int(x) for x in getattr(env.cfg.left_hand_cfg, "joint_ids", [])]
    right_arm_joint_ids = [int(x) for x in env.cfg.right_arm_cfg.joint_ids]
    right_hand_joint_ids = [int(x) for x in getattr(env.cfg.right_hand_cfg, "joint_ids", [])]
    left_hand_dof_zeros = np.zeros((len(left_hand_joint_ids),), dtype=np.float32)
    right_hand_dof_zeros = np.zeros((len(right_hand_joint_ids),), dtype=np.float32)

    command_type = "pose"
    left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    left_ik_commands_world = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_world = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    left_ik_commands_robot = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_robot = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)

    def get_current_left_right_tcp_pose(curr_env_results):
        left_pose = curr_env_results[0]["left_ee_pose"][0].detach().cpu().numpy().copy()
        right_pose = curr_env_results[0]["right_ee_pose"][0].detach().cpu().numpy().copy()
        left_pose[3:7] = quat_wxyz_normalize(left_pose[3:7])
        right_pose[3:7] = quat_wxyz_normalize(right_pose[3:7])
        if bool(args.ik_tcp_offset_enable):
            left_pose = ee_link_pose_to_tcp_pose(left_pose, left_tcp_offset)
            right_pose = ee_link_pose_to_tcp_pose(right_pose, right_tcp_offset)
        return left_pose, right_pose

    left_lock_joint_ids = []
    for jid in left_arm_joint_ids + left_hand_joint_ids:
        if jid not in left_lock_joint_ids:
            left_lock_joint_ids.append(jid)

    right_arm_joint_names = list(getattr(env.cfg.right_arm_cfg, "joint_names", []))
    if len(right_arm_joint_names) != len(right_arm_joint_ids):
        right_arm_joint_names = [f"right_arm_joint{i + 1}" for i in range(len(right_arm_joint_ids))]
    right_hand_joint_names = list(getattr(env.cfg.right_hand_cfg, "joint_names", []))
    if len(right_hand_joint_names) != len(right_hand_joint_ids):
        right_hand_joint_names = [f"right_hand_joint{i + 1}" for i in range(len(right_hand_joint_ids))]

    right_control_joint_ids = []
    right_control_joint_names = []
    for jid, jname in list(zip(right_arm_joint_ids, right_arm_joint_names)) + list(zip(right_hand_joint_ids, right_hand_joint_names)):
        if jid not in right_control_joint_ids:
            right_control_joint_ids.append(jid)
            right_control_joint_names.append(jname)

    selected_right_joint = 0
    teleop_enabled = bool(args.enable_right_joint_teleop)
    left_lock_initial_q = None
    all_joint_names = list(getattr(env.robot, "joint_names", []))
    if len(all_joint_names) < env.num_actions:
        all_joint_names = all_joint_names + [f"joint_{i}" for i in range(len(all_joint_names), env.num_actions)]
    try:
        q_now = env_results[0]["qpos"]
        action_target[:, :] = q_now[:, : env.num_actions].clone()
        if len(left_lock_joint_ids) > 0:
            left_lock_initial_q = env_results[0]["qpos"][0, left_lock_joint_ids].detach().clone()
            action_target[0, left_lock_joint_ids] = left_lock_initial_q.to(action_target.dtype)
    except Exception:
        action_target[:, :] = zero_action
        left_lock_initial_q = None
    pose_hold_target = action_target.clone()
    pose_left_tcp_target, pose_right_tcp_target = get_current_left_right_tcp_pose(env_results)

    def print_all_joint_values():
        try:
            q_now_vec = env_results[0]["qpos"][0].detach().cpu()
        except Exception:
            q_now_vec = env.robot.data.joint_pos[0].detach().cpu()
        q_tgt_vec = action_target[0].detach().cpu()
        n = min(int(env.num_actions), int(q_now_vec.shape[0]), int(q_tgt_vec.shape[0]))
        print("[Teleop] ===== All Joint Values =====")
        for j in range(n):
            name = all_joint_names[j] if j < len(all_joint_names) else f"joint_{j}"
            q_now_j = float(q_now_vec[j].item())
            q_tgt_j = float(q_tgt_vec[j].item())
            lo = float(dof_lower[j].detach().cpu().item())
            hi = float(dof_upper[j].detach().cpu().item())
            print(
                f"[Teleop] {j:02d}:{name} "
                f"q_now={q_now_j:.6f} q_tgt={q_tgt_j:.6f} "
                f"range=[{lo:.6f},{hi:.6f}]"
            )
        print("[Teleop] ============================")

    def print_right_ee_pose():
        try:
            pose = env_results[0]["right_ee_pose"][0].detach().cpu().numpy()
            print(
                "[Teleop] right_ee_pose(x,y,z,w,x,y,z)="
                f"({pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}, "
                f"{pose[3]:.6f}, {pose[4]:.6f}, {pose[5]:.6f}, {pose[6]:.6f})"
            )
        except Exception as e:
            print(f"[Teleop] right_ee_pose print failed: {e}")

    def print_right_tcp_pose():
        try:
            pose = env_results[0]["right_ee_pose"][0].detach().cpu().numpy().copy()
            pose[3:7] = quat_wxyz_normalize(pose[3:7])
            if bool(args.ik_tcp_offset_enable):
                tcp = ee_link_pose_to_tcp_pose(
                    pose,
                    (args.ik_right_tcp_offset_x, args.ik_right_tcp_offset_y, args.ik_right_tcp_offset_z),
                )
                print(
                    "[Teleop] right_tcp_pose(x,y,z,w,x,y,z)="
                    f"({tcp[0]:.6f}, {tcp[1]:.6f}, {tcp[2]:.6f}, "
                    f"{tcp[3]:.6f}, {tcp[4]:.6f}, {tcp[5]:.6f}, {tcp[6]:.6f})"
                )
            else:
                print("[Teleop] ik_tcp_offset_enable=0, right_tcp_pose == right_ee_pose.")
                print(
                    "[Teleop] right_tcp_pose(x,y,z,w,x,y,z)="
                    f"({pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}, "
                    f"{pose[3]:.6f}, {pose[4]:.6f}, {pose[5]:.6f}, {pose[6]:.6f})"
                )
        except Exception as e:
            print(f"[Teleop] right_tcp_pose print failed: {e}")

    window_name = "Teleop Fixed RGB (Right Joint Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    control_mode = "joint"
    print("[Teleop] Ready. Q/Esc quit, N reset scene, P print all joints, M toggle joint/pose mode.")
    if teleop_enabled:
        print("[Teleop] Right joint control ON: 1-9 select joint, J/K dec/inc selected joint, U/I prev/next joint.")
        print("[Teleop] Pose mode control: W/S -> +/-X, A/D -> +/-Y, R/F -> +/-Z on right tcp.")
        print("[Teleop] Right control joints (arm + finger), limits in rad from articulation/USD:")
        for idx, jid in enumerate(right_control_joint_ids):
            lo = float(dof_lower[jid].detach().cpu().item())
            hi = float(dof_upper[jid].detach().cpu().item())
            print(f"  - {idx + 1}:{right_control_joint_names[idx]} (dof_id={jid}) in [{lo:.4f}, {hi:.4f}]")
        if left_lock_initial_q is not None:
            print("[Teleop] Left arm+hand are locked to reset initial joint values.")
    else:
        print("[Teleop] Right joint control OFF (enable with --enable_right_joint_teleop 1).")
    while simulation_app.is_running():
        tic = time.time()

        rgb_obs = env_results[0]["fixed_rgb"][0].detach().cpu().numpy()
        rgb_obs = to_uint8_rgb(rgb_obs)

        # Keep q_tgt persistent across steps; only keyboard edits should change it.
        # Left arm is forced to reset values so it stays in initial pose.
        if left_lock_initial_q is not None and len(left_lock_joint_ids) > 0:
            action_target[0, left_lock_joint_ids] = left_lock_initial_q.to(action_target.dtype)

        lines = [
            "No-warmup mode: USD reset pose kept",
            f"right_joint_teleop={int(teleop_enabled)}",
            f"mode={control_mode}",
            "Keys: M toggle mode | N reset | P print joints | Q quit",
            f"freeze_after_reset={int(bool(args.freeze_after_reset))}",
        ]
        if control_mode == "pose":
            lines.append("pose mode: W/S +/-X | A/D +/-Y | R/F +/-Z")
            try:
                curr_left_tcp, curr_right_tcp = get_current_left_right_tcp_pose(env_results)
                tcp_err = pose_right_tcp_target[:3] - curr_right_tcp[:3]
                lines.append(
                    f"right_tcp_curr=({curr_right_tcp[0]:.4f},{curr_right_tcp[1]:.4f},{curr_right_tcp[2]:.4f})"
                )
                lines.append(
                    f"right_tcp_tgt =({pose_right_tcp_target[0]:.4f},{pose_right_tcp_target[1]:.4f},{pose_right_tcp_target[2]:.4f})"
                )
                lines.append(
                    f"tcp_err=({tcp_err[0]:+.4f},{tcp_err[1]:+.4f},{tcp_err[2]:+.4f}) |err|={float(np.linalg.norm(tcp_err)):.4f}m"
                )
            except Exception:
                lines.append("right tcp telemetry unavailable")
        else:
            lines.append("joint mode: 1-9/U/I select | J/K dec/inc")
        if "object_pose" in env_results[0]:
            box = env_results[0]["object_pose"][0].detach().cpu().numpy()
            lines.append(f"box_xyz=({box[0]:.4f},{box[1]:.4f},{box[2]:.4f})")
        if hasattr(env, "goal_pos_w"):
            goal = (env.goal_pos_w[0].detach().cpu() - env.scene.env_origins[0].detach().cpu()).numpy()
            lines.append(f"goal_xyz=({goal[0]:.4f},{goal[1]:.4f},{goal[2]:.4f})")
        if teleop_enabled and control_mode == "joint" and len(right_control_joint_ids) > 0:
            sel = int(np.clip(selected_right_joint, 0, len(right_control_joint_ids) - 1))
            jid = right_control_joint_ids[sel]
            try:
                q_now_val = float(env_results[0]["qpos"][0, jid].detach().cpu().item())
            except Exception:
                q_now_val = float("nan")
            q_tgt_val = float(action_target[0, jid].detach().cpu().item())
            q_lo = float(dof_lower[jid].detach().cpu().item())
            q_hi = float(dof_upper[jid].detach().cpu().item())
            jname = right_control_joint_names[sel]
            lines.append(
                f"selected_right={sel+1}:{jname} dof_id={jid} "
                f"q_now={q_now_val:.4f} q_tgt={q_tgt_val:.4f}"
            )
            lines.append(f"selected_range=[{q_lo:.4f}, {q_hi:.4f}] rad")
        vis = overlay_info(rgb_obs, lines)
        cv2.imshow(window_name, vis[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("m"), ord("M")):
            if control_mode == "joint":
                control_mode = "pose"
                try:
                    q_now = env_results[0]["qpos"]
                    pose_hold_target[:, :] = q_now[:, : env.num_actions].clone()
                except Exception:
                    pass
                if left_lock_initial_q is not None and len(left_lock_joint_ids) > 0:
                    pose_hold_target[0, left_lock_joint_ids] = left_lock_initial_q.to(pose_hold_target.dtype)
                action_target[:, :] = pose_hold_target
                left_ik_controller.reset()
                right_ik_controller.reset()
                pose_left_tcp_target, pose_right_tcp_target = get_current_left_right_tcp_pose(env_results)
                print(
                    "[Teleop] entered POSE control mode. "
                    f"right_tcp_target=({pose_right_tcp_target[0]:.4f}, {pose_right_tcp_target[1]:.4f}, {pose_right_tcp_target[2]:.4f})"
                )
            else:
                control_mode = "joint"
                print("[Teleop] switched back to JOINT control mode.")
            continue
        if key == ord("n"):
            env_results = env.reset()
            print_init_state(env, env_results, args)
            selected_right_joint = 0
            try:
                q_now = env_results[0]["qpos"]
                action_target[:, :] = q_now[:, : env.num_actions].clone()
                if len(left_lock_joint_ids) > 0:
                    left_lock_initial_q = env_results[0]["qpos"][0, left_lock_joint_ids].detach().clone()
                    action_target[0, left_lock_joint_ids] = left_lock_initial_q.to(action_target.dtype)
                pose_hold_target[:, :] = action_target
                pose_left_tcp_target, pose_right_tcp_target = get_current_left_right_tcp_pose(env_results)
                left_ik_controller.reset()
                right_ik_controller.reset()
            except Exception:
                action_target[:, :] = zero_action
                left_lock_initial_q = None
                pose_hold_target[:, :] = action_target
            continue
        if key in (ord("p"), ord("P")):
            print_all_joint_values()
            print_right_ee_pose()
            print_right_tcp_pose()
            continue

        delta = 0.0
        if teleop_enabled and control_mode == "joint" and len(right_control_joint_ids) > 0:
            if key >= ord("1") and key <= ord("9"):
                selected_candidate = int(chr(key)) - 1
                if selected_candidate < len(right_control_joint_ids):
                    selected_right_joint = selected_candidate
                else:
                    print(f"[Teleop] key {chr(key)} has no mapped joint (total={len(right_control_joint_ids)}).")
                sel = selected_right_joint
                jid = right_control_joint_ids[sel]
                lo = float(dof_lower[jid].detach().cpu().item())
                hi = float(dof_upper[jid].detach().cpu().item())
                print(
                    f"[Teleop] selected right joint {sel + 1}: {right_control_joint_names[sel]} "
                    f"(dof_id={jid}, range=[{lo:.4f},{hi:.4f}] rad)"
                )
            elif key == ord("u"):
                selected_right_joint = max(0, selected_right_joint - 1)
                sel = selected_right_joint
                jid = right_control_joint_ids[sel]
                lo = float(dof_lower[jid].detach().cpu().item())
                hi = float(dof_upper[jid].detach().cpu().item())
                print(
                    f"[Teleop] selected right joint {sel + 1}: {right_control_joint_names[sel]} "
                    f"(dof_id={jid}, range=[{lo:.4f},{hi:.4f}] rad)"
                )
            elif key == ord("i"):
                selected_right_joint = min(len(right_control_joint_ids) - 1, selected_right_joint + 1)
                sel = selected_right_joint
                jid = right_control_joint_ids[sel]
                lo = float(dof_lower[jid].detach().cpu().item())
                hi = float(dof_upper[jid].detach().cpu().item())
                print(
                    f"[Teleop] selected right joint {sel + 1}: {right_control_joint_names[sel]} "
                    f"(dof_id={jid}, range=[{lo:.4f},{hi:.4f}] rad)"
                )
            elif key == ord("j"):
                delta = -float(args.right_joint_step)
            elif key == ord("k"):
                delta = float(args.right_joint_step)

            if abs(delta) > 0.0:
                sel = int(np.clip(selected_right_joint, 0, len(right_control_joint_ids) - 1))
                jid = right_control_joint_ids[sel]
                action_target[0, jid] = action_target[0, jid] + delta
                if bool(args.right_joint_clip_to_limits):
                    action_target[0, jid] = torch.clamp(action_target[0, jid], dof_lower[jid], dof_upper[jid])
                print(
                    f"[Teleop] {right_control_joint_names[sel]} -> "
                    f"{float(action_target[0, jid].detach().cpu().item()):.4f} rad"
                )
        pose_step = float(args.pose_pos_step)
        if teleop_enabled and control_mode == "pose":
            pose_delta = np.zeros((3,), dtype=np.float64)
            if key in (ord("w"), ord("W")):
                pose_delta[0] += pose_step
            elif key in (ord("s"), ord("S")):
                pose_delta[0] -= pose_step
            elif key in (ord("a"), ord("A")):
                pose_delta[1] += pose_step
            elif key in (ord("d"), ord("D")):
                pose_delta[1] -= pose_step
            elif key in (ord("r"), ord("R")):
                pose_delta[2] += pose_step
            elif key in (ord("f"), ord("F")):
                pose_delta[2] -= pose_step
            if float(np.linalg.norm(pose_delta)) > 0.0:
                pose_right_tcp_target[0:3] = pose_right_tcp_target[0:3] + pose_delta
                print(
                    "[Teleop] right_tcp_target xyz -> "
                    f"({pose_right_tcp_target[0]:.4f}, {pose_right_tcp_target[1]:.4f}, {pose_right_tcp_target[2]:.4f})"
                )

        if teleop_enabled and control_mode == "joint":
            env_results = env.step(action_target)
        elif teleop_enabled and control_mode == "pose":
            # Pose teleop: track tcp target via IK every step (physics ON).
            ik_step(
                env,
                left_ik_controller,
                right_ik_controller,
                left_ik_commands_world,
                right_ik_commands_world,
                left_ik_commands_robot,
                right_ik_commands_robot,
                pose_left_tcp_target,
                pose_right_tcp_target,
                left_hand_dof_zeros,
                right_hand_dof_zeros,
                action_target,
                ignore_orientation=bool(args.ik_ignore_orientation),
                active_arm=ik_active_arm,
                tcp_offset_enable=bool(args.ik_tcp_offset_enable),
                left_tcp_offset=left_tcp_offset,
                right_tcp_offset=right_tcp_offset,
            )
            # Keep left arm/hand fixed at reset pose in teleop.
            if left_lock_initial_q is not None and len(left_lock_joint_ids) > 0:
                action_target[0, left_lock_joint_ids] = left_lock_initial_q.to(action_target.dtype)
            env_results = env.step(action_target)
        elif bool(args.freeze_after_reset):
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
