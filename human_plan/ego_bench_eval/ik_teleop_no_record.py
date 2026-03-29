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
        description="No-warmup teleop checker with right-arm joint or EE-XYZ control."
    )
    parser.add_argument("--task", type=str, default="Humanoid-Push-Box-v0")
    parser.add_argument("--room_idx", type=int, default=1)
    parser.add_argument("--table_idx", type=int, default=2)
    parser.add_argument("--env_randomize", type=int, default=0, help="0=fixed scene, 1=randomized scene")
    parser.add_argument("--spawn_background", type=int, default=1)

    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--loop_hz", type=float, default=30.0)
    parser.add_argument("--freeze_after_reset", type=int, default=1, help="1=hold reset frame, 0=step with zero action")
    parser.add_argument("--teleop_mode", type=str, default="ee_xyz", choices=["joint", "ee_xyz", "hybrid"], help="initial mode: ee_xyz or joint; press M at runtime to switch")
    parser.add_argument("--enable_right_joint_teleop", type=int, default=1, help="1=enable keyboard teleop for right arm joint targets")
    parser.add_argument("--right_joint_step", type=float, default=0.02, help="right-arm joint increment in rad")
    parser.add_argument("--right_joint_clip_to_limits", type=int, default=1, help="clip right joint target to dof limits")
    parser.add_argument("--ee_pos_step", type=float, default=0.006, help="right gripper xyz step in meters for ee_xyz mode")

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
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
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

    teleop_mode = str(args.teleop_mode).strip().lower()
    if teleop_mode == "hybrid":
        # Backward compatibility: old "hybrid" now starts in ee_xyz and toggles with M.
        teleop_mode = "ee_xyz"
    if teleop_mode not in ("joint", "ee_xyz"):
        teleop_mode = "ee_xyz"
    current_control_mode = teleop_mode
    joint_teleop_enabled = bool(args.enable_right_joint_teleop)

    ik_active_arm = str(args.ik_active_arm).strip().lower()
    if bool(args.right_only_mode):
        ik_active_arm = "right"
    if ik_active_arm not in ("both", "left", "right"):
        ik_active_arm = "right"

    left_tcp_offset = (
        float(args.ik_left_tcp_offset_x),
        float(args.ik_left_tcp_offset_y),
        float(args.ik_left_tcp_offset_z),
    )
    right_tcp_offset = (
        float(args.ik_right_tcp_offset_x),
        float(args.ik_right_tcp_offset_y),
        float(args.ik_right_tcp_offset_z),
    )

    print("[Teleop] preserve_usd_initial_ee=1 behavior is enforced in this script.")
    print("[Teleop] NO init_poses warmup.")
    print(f"[Teleop] initial_mode={current_control_mode} (press M to switch) | ik_tcp_offset_enable={int(bool(args.ik_tcp_offset_enable))}")
    if bool(args.ik_tcp_offset_enable):
        print(f"[Teleop] tcp offsets: left={left_tcp_offset}, right={right_tcp_offset}")

    env_results = env.reset()
    print_init_state(env, env_results, args)

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

    left_ik_controller = None
    right_ik_controller = None
    left_ik_commands_world = None
    right_ik_commands_world = None
    left_ik_commands_robot = None
    right_ik_commands_robot = None
    left_target_pose = None
    right_target_pose = None

    def build_current_targets_from_env(curr_env_results):
        left_link = curr_env_results[0]["left_ee_pose"][0].detach().cpu().numpy().copy()
        right_link = curr_env_results[0]["right_ee_pose"][0].detach().cpu().numpy().copy()
        left_link[3:7] = quat_wxyz_normalize(left_link[3:7])
        right_link[3:7] = quat_wxyz_normalize(right_link[3:7])
        if bool(args.ik_tcp_offset_enable):
            left_target = ee_link_pose_to_tcp_pose(left_link, left_tcp_offset)
            right_target = ee_link_pose_to_tcp_pose(right_link, right_tcp_offset)
        else:
            left_target = left_link
            right_target = right_link
        return left_target, right_target

    command_type = "pose"
    left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    left_ik_commands_world = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_world = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    left_ik_commands_robot = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_robot = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    left_target_pose, right_target_pose = build_current_targets_from_env(env_results)

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
    left_lock_initial_q = None
    manual_override_mask = torch.zeros((env.num_actions,), dtype=torch.bool, device=env.robot.device)
    manual_override_values = torch.zeros((env.num_actions,), dtype=action_target.dtype, device=env.robot.device)
    all_joint_names = list(getattr(env.robot, "joint_names", []))
    if len(all_joint_names) < env.num_actions:
        all_joint_names = all_joint_names + [f"joint_{i}" for i in range(len(all_joint_names), env.num_actions)]
    try:
        q_now = env_results[0]["qpos"]
        action_target[:, :] = q_now[:, : env.num_actions].clone()
        manual_override_values[:] = action_target[0, :]
        if len(left_lock_joint_ids) > 0:
            left_lock_initial_q = env_results[0]["qpos"][0, left_lock_joint_ids].detach().clone()
            action_target[0, left_lock_joint_ids] = left_lock_initial_q.to(action_target.dtype)
    except Exception:
        action_target[:, :] = zero_action
        manual_override_values[:] = action_target[0, :]
        left_lock_initial_q = None

    def print_all_joint_values():
        try:
            q_now_vec = env_results[0]["qpos"][0].detach().cpu()
        except Exception:
            q_now_vec = env.robot.data.joint_pos[0].detach().cpu()
        q_tgt_vec = action_target[0].detach().cpu().clone()
        try:
            manual_mask_cpu = manual_override_mask.detach().cpu()
            if manual_mask_cpu.shape[0] == q_tgt_vec.shape[0]:
                q_tgt_vec[~manual_mask_cpu] = q_now_vec[~manual_mask_cpu]
        except Exception:
            pass
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

    def print_right_tcp_compare():
        if right_target_pose is None:
            print("[Teleop] right_target_tcp is not initialized.")
            return
        try:
            right_link_curr = env_results[0]["right_ee_pose"][0].detach().cpu().numpy().copy()
            right_link_curr[3:7] = quat_wxyz_normalize(right_link_curr[3:7])
            if bool(args.ik_tcp_offset_enable):
                right_tcp_curr = ee_link_pose_to_tcp_pose(right_link_curr, right_tcp_offset)
            else:
                right_tcp_curr = right_link_curr
            tcp_err = right_target_pose[0:3] - right_tcp_curr[0:3]
            print(
                "[Teleop] right_tcp_curr_xyz="
                f"({right_tcp_curr[0]:.4f}, {right_tcp_curr[1]:.4f}, {right_tcp_curr[2]:.4f})"
            )
            print(
                "[Teleop] right_target_tcp_xyz="
                f"({right_target_pose[0]:.4f}, {right_target_pose[1]:.4f}, {right_target_pose[2]:.4f})"
            )
            print(
                "[Teleop] right_tcp_err_xyz="
                f"({tcp_err[0]:+.4f}, {tcp_err[1]:+.4f}, {tcp_err[2]:+.4f}) "
                f"|err|={float(np.linalg.norm(tcp_err)):.4f}m"
            )
        except Exception as e:
            print(f"[Teleop] right tcp compare failed: {e}")

    window_name = "Teleop Fixed RGB (Right Joint / EE-XYZ Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[Teleop] Ready. Q/Esc quit, N reset, P print joints, M toggle mode.")
    print("[Teleop] Mode ee_xyz: W/S +/-X, A/D +/-Y, R/F +/-Z.")
    if joint_teleop_enabled:
        print("[Teleop] Mode joint: 1-9/U/I select, J/K dec/inc, C clear joint overrides.")
        print("[Teleop] Right control joints (arm + finger), limits in rad from articulation/USD:")
        for idx, jid in enumerate(right_control_joint_ids):
            lo = float(dof_lower[jid].detach().cpu().item())
            hi = float(dof_upper[jid].detach().cpu().item())
            print(f"  - {idx + 1}:{right_control_joint_names[idx]} (dof_id={jid}) in [{lo:.4f}, {hi:.4f}]")
    else:
        print("[Teleop] Joint mode keyboard edits are disabled by --enable_right_joint_teleop 0.")
    if left_lock_initial_q is not None:
        print("[Teleop] Left arm+hand are locked to reset initial joint values.")
    print(f"[Teleop] IK active_arm={ik_active_arm}, ignore_orientation={int(bool(args.ik_ignore_orientation))}")
    print(f"[Teleop] Current mode: {current_control_mode}")

    while simulation_app.is_running():
        tic = time.time()
        joint_mode_active = current_control_mode == "joint"
        ee_mode_active = current_control_mode == "ee_xyz"

        rgb_obs = env_results[0]["fixed_rgb"][0].detach().cpu().numpy()
        rgb_obs = to_uint8_rgb(rgb_obs)

        # Keep q_tgt persistent across steps; only keyboard edits should change it.
        # Left arm is forced to reset values so it stays in initial pose.
        if left_lock_initial_q is not None and len(left_lock_joint_ids) > 0:
            action_target[0, left_lock_joint_ids] = left_lock_initial_q.to(action_target.dtype)

        lines = [
            "No-warmup mode: USD reset pose kept",
            f"mode={current_control_mode} (M to toggle)",
            f"freeze_after_reset={int(bool(args.freeze_after_reset))}",
        ]
        if joint_mode_active:
            lines.append("Joint keys: 1-9/U/I select | J/K dec/inc | C clear joint overrides")
            lines.append(f"right_joint_teleop={int(joint_teleop_enabled)}")
        else:
            lines.append("EE keys: W/S +/-X | A/D +/-Y | R/F +/-Z")
            lines.append(f"ee_step={float(args.ee_pos_step):.4f}m | tcp_offset={int(bool(args.ik_tcp_offset_enable))}")
        if "object_pose" in env_results[0]:
            box = env_results[0]["object_pose"][0].detach().cpu().numpy()
            lines.append(f"box_xyz=({box[0]:.4f},{box[1]:.4f},{box[2]:.4f})")
        if hasattr(env, "goal_pos_w"):
            goal = (env.goal_pos_w[0].detach().cpu() - env.scene.env_origins[0].detach().cpu()).numpy()
            lines.append(f"goal_xyz=({goal[0]:.4f},{goal[1]:.4f},{goal[2]:.4f})")
        if joint_mode_active and joint_teleop_enabled and len(right_control_joint_ids) > 0:
            sel = int(np.clip(selected_right_joint, 0, len(right_control_joint_ids) - 1))
            jid = right_control_joint_ids[sel]
            try:
                q_now_val = float(env_results[0]["qpos"][0, jid].detach().cpu().item())
            except Exception:
                q_now_val = float("nan")
            if bool(manual_override_mask[jid].item()):
                q_tgt_val = float(manual_override_values[jid].detach().cpu().item())
                tgt_tag = "manual"
            else:
                q_tgt_val = q_now_val
                tgt_tag = "auto"
            q_lo = float(dof_lower[jid].detach().cpu().item())
            q_hi = float(dof_upper[jid].detach().cpu().item())
            jname = all_joint_names[jid] if jid < len(all_joint_names) else right_control_joint_names[sel]
            lines.append(
                f"selected_right={sel+1}:{jname} dof_id={jid} "
                f"q_now={q_now_val:.4f} q_tgt={q_tgt_val:.4f} ({tgt_tag})"
            )
            lines.append(f"selected_range=[{q_lo:.4f}, {q_hi:.4f}] rad")
        if ee_mode_active and right_target_pose is not None:
            try:
                right_link_curr = env_results[0]["right_ee_pose"][0].detach().cpu().numpy().copy()
                right_link_curr[3:7] = quat_wxyz_normalize(right_link_curr[3:7])
                if bool(args.ik_tcp_offset_enable):
                    right_tcp_curr = ee_link_pose_to_tcp_pose(right_link_curr, right_tcp_offset)
                else:
                    right_tcp_curr = right_link_curr
                tcp_err = right_target_pose[0:3] - right_tcp_curr[0:3]
                lines.append(
                    "right_tcp_curr_xyz="
                    f"({right_tcp_curr[0]:.4f},{right_tcp_curr[1]:.4f},{right_tcp_curr[2]:.4f})"
                )
                lines.append(
                    "right_tcp_err_xyz="
                    f"({tcp_err[0]:+.4f},{tcp_err[1]:+.4f},{tcp_err[2]:+.4f}) "
                    f"|err|={float(np.linalg.norm(tcp_err)):.4f}m"
                )
            except Exception:
                lines.append("right_tcp_curr_xyz=N/A")
            lines.append(
                "right_target_tcp_xyz="
                f"({right_target_pose[0]:.4f},{right_target_pose[1]:.4f},{right_target_pose[2]:.4f})"
            )
        vis = overlay_info(rgb_obs, lines)
        cv2.imshow(window_name, vis[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("m"), ord("M")):
            if current_control_mode == "ee_xyz":
                current_control_mode = "joint"
                try:
                    q_now = env_results[0]["qpos"]
                    action_target[:, :] = q_now[:, : env.num_actions].clone()
                    manual_override_values[:] = action_target[0, :]
                    manual_override_mask[:] = False
                except Exception:
                    action_target[:, :] = zero_action
                    manual_override_values[:] = action_target[0, :]
                    manual_override_mask[:] = False
                print("[Teleop] mode switched to JOINT.")
            else:
                current_control_mode = "ee_xyz"
                left_ik_controller.reset()
                right_ik_controller.reset()
                left_target_pose, right_target_pose = build_current_targets_from_env(env_results)
                print(
                    "[Teleop] mode switched to EE_XYZ. "
                    f"right_target_tcp_sync=({right_target_pose[0]:.4f}, {right_target_pose[1]:.4f}, {right_target_pose[2]:.4f})"
                )
            continue
        if key == ord("n"):
            env_results = env.reset()
            print_init_state(env, env_results, args)
            selected_right_joint = 0
            try:
                q_now = env_results[0]["qpos"]
                action_target[:, :] = q_now[:, : env.num_actions].clone()
                manual_override_values[:] = action_target[0, :]
                manual_override_mask[:] = False
                if len(left_lock_joint_ids) > 0:
                    left_lock_initial_q = env_results[0]["qpos"][0, left_lock_joint_ids].detach().clone()
                    action_target[0, left_lock_joint_ids] = left_lock_initial_q.to(action_target.dtype)
            except Exception:
                action_target[:, :] = zero_action
                manual_override_values[:] = action_target[0, :]
                manual_override_mask[:] = False
                left_lock_initial_q = None
            left_ik_controller.reset()
            right_ik_controller.reset()
            left_target_pose, right_target_pose = build_current_targets_from_env(env_results)
            continue
        if key in (ord("p"), ord("P")):
            print_all_joint_values()
            print_right_tcp_compare()
            continue

        delta = 0.0
        ee_key_cmd_this_frame = False
        if joint_mode_active and joint_teleop_enabled and len(right_control_joint_ids) > 0:
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
            elif key in (ord("c"), ord("C")):
                manual_override_mask[:] = False
                print("[Teleop] cleared all right-joint manual overrides.")

            if abs(delta) > 0.0:
                sel = int(np.clip(selected_right_joint, 0, len(right_control_joint_ids) - 1))
                jid = right_control_joint_ids[sel]
                if bool(manual_override_mask[jid].item()):
                    base_val = manual_override_values[jid]
                else:
                    base_val = action_target[0, jid]
                new_val = base_val + delta
                if bool(args.right_joint_clip_to_limits):
                    new_val = torch.clamp(new_val, dof_lower[jid], dof_upper[jid])
                manual_override_values[jid] = new_val
                manual_override_mask[jid] = True
                action_target[0, jid] = new_val
                print(
                    f"[Teleop] {right_control_joint_names[sel]} manual -> "
                    f"{float(manual_override_values[jid].detach().cpu().item()):.4f} rad"
                )

        if ee_mode_active and right_target_pose is not None:
            delta_xyz = np.zeros((3,), dtype=np.float64)
            if key in (ord("w"), ord("W")):
                delta_xyz[0] += float(args.ee_pos_step)
            elif key in (ord("s"), ord("S")):
                delta_xyz[0] -= float(args.ee_pos_step)
            elif key in (ord("a"), ord("A")):
                delta_xyz[1] += float(args.ee_pos_step)
            elif key in (ord("d"), ord("D")):
                delta_xyz[1] -= float(args.ee_pos_step)
            elif key in (ord("r"), ord("R")):
                delta_xyz[2] += float(args.ee_pos_step)
            elif key in (ord("f"), ord("F")):
                delta_xyz[2] -= float(args.ee_pos_step)
            if float(np.linalg.norm(delta_xyz)) > 0.0:
                right_target_pose[0:3] = right_target_pose[0:3] + delta_xyz
                ee_key_cmd_this_frame = True
                print(
                    "[Teleop] right_target_tcp xyz -> "
                    f"({right_target_pose[0]:.4f}, {right_target_pose[1]:.4f}, {right_target_pose[2]:.4f})"
                )

        if ee_mode_active and right_target_pose is not None:
            if ee_key_cmd_this_frame:
                ik_step(
                    env,
                    left_ik_controller,
                    right_ik_controller,
                    left_ik_commands_world,
                    right_ik_commands_world,
                    left_ik_commands_robot,
                    right_ik_commands_robot,
                    left_target_pose,
                    right_target_pose,
                    left_hand_dof_zeros,
                    right_hand_dof_zeros,
                    action_target,
                    ignore_orientation=bool(args.ik_ignore_orientation),
                    active_arm=ik_active_arm,
                    tcp_offset_enable=bool(args.ik_tcp_offset_enable),
                    left_tcp_offset=left_tcp_offset,
                    right_tcp_offset=right_tcp_offset,
                )
            else:
                try:
                    q_now = env_results[0]["qpos"]
                    action_target[:, :] = q_now[:, : env.num_actions].clone()
                    left_target_pose, right_target_pose = build_current_targets_from_env(env_results)
                except Exception:
                    pass
            if left_lock_initial_q is not None and len(left_lock_joint_ids) > 0:
                action_target[0, left_lock_joint_ids] = left_lock_initial_q.to(action_target.dtype)
            env_results = env.step(action_target)
        elif joint_mode_active:
            if len(right_control_joint_ids) > 0:
                action_target[0, manual_override_mask] = manual_override_values[manual_override_mask]
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
