import argparse
import time
import os
import pickle
import random

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R

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


def quat_wxyz_apply_rpy_delta(q_wxyz: np.ndarray, d_roll: float, d_pitch: float, d_yaw: float) -> np.ndarray:
    """Apply local RPY delta (radians) to wxyz quaternion."""
    q = quat_wxyz_normalize(q_wxyz)
    r_curr = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses xyzw
    r_delta = R.from_euler("xyz", [d_roll, d_pitch, d_yaw], degrees=False)
    r_new = r_curr * r_delta
    q_xyzw = r_new.as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return quat_wxyz_normalize(q_wxyz)


def quat_wxyz_rotate_vec(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    """Rotate a vector by a wxyz quaternion."""
    q = quat_wxyz_normalize(q_wxyz)
    q_xyz = q[1:4]
    q_w = float(q[0])
    v = np.asarray(v_xyz, dtype=np.float64)
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def ee_link_pose_to_tcp_pose(ee_pose_wxyz: np.ndarray, tcp_offset_xyz: tuple[float, float, float]) -> np.ndarray:
    """
    Convert link7 pose to TCP pose in world frame using local-frame translational offset.
    Orientation is unchanged.
    """
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
        # Draw default overlay text in red for better readability.
        color = (255, 64, 64)
        if line.startswith("[WARN]"):
            color = (255, 0, 0)
        cv2.putText(
            vis,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 24
    return vis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-record teleop for Push-Box IK debugging.")
    parser.add_argument("--task", type=str, default="Humanoid-Push-Box-v0")
    parser.add_argument("--room_idx", type=int, default=1)
    parser.add_argument("--table_idx", type=int, default=2)
    parser.add_argument("--env_randomize", type=int, default=0, help="0=fixed scene, 1=randomized scene")
    parser.add_argument("--spawn_background", type=int, default=1)

    parser.add_argument("--ik_active_arm", type=str, default="right", choices=["left", "right", "both"])
    parser.add_argument("--right_only_mode", type=int, default=1, help="1=lock to right arm teleop, ignore 1/3 arm-switch keys")
    parser.add_argument("--ik_ignore_orientation", type=int, default=1, help="1=position-only IK, 0=track full pose")
    parser.add_argument("--ik_tcp_offset_enable", type=int, default=1)
    parser.add_argument("--ik_left_tcp_offset_x", type=float, default=0.0)
    parser.add_argument("--ik_left_tcp_offset_y", type=float, default=0.0)
    parser.add_argument("--ik_left_tcp_offset_z", type=float, default=0.2104)
    parser.add_argument("--ik_right_tcp_offset_x", type=float, default=0.0)
    parser.add_argument("--ik_right_tcp_offset_y", type=float, default=0.0)
    parser.add_argument("--ik_right_tcp_offset_z", type=float, default=0.2104)
    parser.add_argument("--left_ee_body_name", type=str, default="left_arm_link7")
    parser.add_argument("--right_ee_body_name", type=str, default="right_arm_link7")

    parser.add_argument("--robot_base_x", type=float, default=None)
    parser.add_argument("--robot_base_y", type=float, default=None)
    parser.add_argument("--robot_base_z", type=float, default=None)
    parser.add_argument("--box_init_x", type=float, default=None)
    parser.add_argument("--box_init_y", type=float, default=None)
    parser.add_argument("--box_init_z", type=float, default=None)
    parser.add_argument("--goal_x", type=float, default=None)
    parser.add_argument("--goal_y", type=float, default=None)
    parser.add_argument("--goal_z", type=float, default=None)

    parser.add_argument("--start_mode", type=str, default="benchmark", choices=["reset", "eval_init", "benchmark"])
    parser.add_argument("--init_pose_pkl", type=str, default="init_poses_fixed_set_100traj.pkl")
    parser.add_argument("--init_episode_rank", type=int, default=0, help="index in TASK_INIT_EPISODE list")
    parser.add_argument("--init_warmup_steps", type=int, default=100)
    parser.add_argument(
        "--preserve_usd_initial_ee",
        type=int,
        default=1,
        help="1=keep USD-loaded EE pose exactly at reset (skip eval-init warmup and post-init retreat)",
    )
    parser.add_argument("--lock_box_after_init", type=int, default=1, help="restore box pose after init warmup to prevent accidental pre-contact")
    parser.add_argument("--post_init_retreat_steps", type=int, default=24, help="interpolation steps for post-init retreat")
    parser.add_argument("--post_init_left_retreat_x", type=float, default=-0.12, help="left TCP retreat along world X after init")
    parser.add_argument("--post_init_right_retreat_x", type=float, default=-0.06, help="right TCP retreat along world X after init")
    parser.add_argument("--post_init_left_retreat_y", type=float, default=0.0, help="left TCP retreat along world Y after init")
    parser.add_argument("--post_init_right_retreat_y", type=float, default=0.0, help="right TCP retreat along world Y after init")
    parser.add_argument("--bench_num_episodes", type=int, default=3, help="same meaning as benchmark arg --num_episodes")
    parser.add_argument("--bench_num_trials", type=int, default=1, help="same meaning as benchmark arg --num_trials")
    parser.add_argument("--bench_trial_idx", type=int, default=0, help="which trial index to mimic in benchmark flow")

    parser.add_argument("--pos_step", type=float, default=0.006, help="position step in meters")
    parser.add_argument("--rot_step_deg", type=float, default=3.0, help="rotation step in degrees")
    parser.add_argument("--loop_hz", type=float, default=30.0, help="control loop frequency")
    parser.add_argument("--workspace_warn_enable", type=int, default=1, help="warn when right-arm target exceeds task-space bounds")
    parser.add_argument("--right_ws_x_min", type=float, default=0.20)
    parser.add_argument("--right_ws_x_max", type=float, default=0.90)
    parser.add_argument("--right_ws_y_min", type=float, default=-0.60)
    parser.add_argument("--right_ws_y_max", type=float, default=0.40)
    parser.add_argument("--right_ws_z_min", type=float, default=0.88)
    parser.add_argument("--right_ws_z_max", type=float, default=1.42)
    parser.add_argument("--right_ws_rot_max_deg", type=float, default=95.0, help="max allowed angular delta from reset orientation")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(enable_cameras=True, device="cuda", headless=False)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from omni.isaac.lab_tasks.utils import parse_env_cfg

    from humanoid.tasks.base_env import BaseEnv, BaseEnvCfg
    from human_plan.ego_bench_eval.utils import ik_step, TASK_INIT_EPISODE

    env_cfg: BaseEnvCfg = parse_env_cfg(args.task, num_envs=1)
    task_name = args.task[9:-3] if args.task.startswith("Humanoid-") and args.task.endswith("-v0") else args.task
    # Keep benchmark-style reset/init flow, but respect env_randomize so we can lock scene when needed.
    env_cfg.randomize = bool(args.env_randomize)
    env_cfg.spawn_background = bool(args.spawn_background)
    env_cfg.room_idx = args.room_idx
    env_cfg.table_idx = args.table_idx
    env_cfg.episode_length_s = 120

    if args.left_ee_body_name.strip():
        env_cfg.left_arm_cfg.body_names = [args.left_ee_body_name.strip()]
    if args.right_ee_body_name.strip():
        env_cfg.right_arm_cfg.body_names = [args.right_ee_body_name.strip()]

    if any(v is not None for v in (args.robot_base_x, args.robot_base_y, args.robot_base_z)):
        default_robot_pos = list(env_cfg.robot.init_state.pos)
        env_cfg.robot.init_state.pos = (
            args.robot_base_x if args.robot_base_x is not None else default_robot_pos[0],
            args.robot_base_y if args.robot_base_y is not None else default_robot_pos[1],
            args.robot_base_z if args.robot_base_z is not None else default_robot_pos[2],
        )

    if hasattr(env_cfg, "box") and any(v is not None for v in (args.box_init_x, args.box_init_y, args.box_init_z)):
        default_box_pos = list(env_cfg.box.init_state.pos)
        env_cfg.box.init_state.pos = (
            args.box_init_x if args.box_init_x is not None else default_box_pos[0],
            args.box_init_y if args.box_init_y is not None else default_box_pos[1],
            args.box_init_z if args.box_init_z is not None else default_box_pos[2],
        )

    if hasattr(env_cfg, "goal_default_pos") and any(v is not None for v in (args.goal_x, args.goal_y, args.goal_z)):
        default_goal_pos = list(env_cfg.goal_default_pos)
        env_cfg.goal_default_pos = (
            args.goal_x if args.goal_x is not None else default_goal_pos[0],
            args.goal_y if args.goal_y is not None else default_goal_pos[1],
            args.goal_z if args.goal_z is not None else default_goal_pos[2],
        )

    env: BaseEnv = gym.make(args.task, cfg=env_cfg)

    left_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    right_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)

    left_ik_commands_world = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    left_ik_commands_robot = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_world = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_robot = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    action = torch.zeros((env.scene.num_envs, env.num_actions), device=env.robot.device)

    left_hand_dof = np.zeros(len(env.cfg.left_hand_cfg.joint_ids), dtype=np.float64)
    right_hand_dof = np.zeros(len(env.cfg.right_hand_cfg.joint_ids), dtype=np.float64)

    left_tcp_offset = (args.ik_left_tcp_offset_x, args.ik_left_tcp_offset_y, args.ik_left_tcp_offset_z)
    right_tcp_offset = (args.ik_right_tcp_offset_x, args.ik_right_tcp_offset_y, args.ik_right_tcp_offset_z)
    tcp_offset_enable = bool(args.ik_tcp_offset_enable)
    ignore_orientation = bool(args.ik_ignore_orientation)
    if tcp_offset_enable:
        print(
            f"[Teleop] target frame=TCP, IK body=link7, offsets L={left_tcp_offset} R={right_tcp_offset}"
        )
    else:
        print("[Teleop] target frame=link7 (tcp offset compensation disabled)")
    print(f"[Teleop] IK mode={'position-only' if ignore_orientation else 'full-pose'}")

    bench_seq_name = None
    bench_randomize_idx = None
    if args.start_mode == "benchmark":
        try:
            if args.task not in SEED_MAP:
                raise ValueError(f"task not in SEED_MAP: {args.task}")
            if task_name not in TASK_INIT_EPISODE:
                raise ValueError(f"task not in TASK_INIT_EPISODE: {task_name}")

            episode_pool = TASK_INIT_EPISODE[task_name]
            bench_num_episodes = max(1, min(int(args.bench_num_episodes), len(episode_pool)))
            bench_num_trials = max(1, int(args.bench_num_trials))
            bench_episode_rank = max(0, min(int(args.init_episode_rank), bench_num_episodes - 1))
            bench_trial_idx = max(0, min(int(args.bench_trial_idx), bench_num_trials - 1))
            bench_seq_name = episode_pool[bench_episode_rank][0]

            random.seed(SEED_MAP[args.task])
            randomize_idxes = list(range(10000))
            random.shuffle(randomize_idxes)
            base = 100 + (args.room_idx * 5 + args.table_idx) * bench_num_trials * bench_num_episodes
            offset = 1 + bench_episode_rank * bench_num_trials + bench_trial_idx
            raw_idx = (base + offset) % len(randomize_idxes)
            bench_randomize_idx = randomize_idxes[raw_idx]
            print(
                f"[Teleop] benchmark init: task={task_name} seq={bench_seq_name} "
                f"episode_rank={bench_episode_rank}/{bench_num_episodes-1} trial={bench_trial_idx}/{bench_num_trials-1} "
                f"randomize_idx={bench_randomize_idx} env_randomize={env_cfg.randomize}"
            )
        except Exception as e:
            print(f"[Teleop] benchmark context build failed, fallback to eval_init/reset. reason={e}")
            args.start_mode = "eval_init"

    def load_eval_init_target(seq_name: str):
        if not os.path.exists(args.init_pose_pkl):
            raise FileNotFoundError(args.init_pose_pkl)
        with open(args.init_pose_pkl, "rb") as f:
            init_poses = pickle.load(f)
        left_dof = init_poses[task_name][seq_name][0]["left_dof"]
        right_dof = init_poses[task_name][seq_name][0]["right_dof"]
        left_ee_pose = init_poses[task_name][seq_name][0]["left_ee"]
        right_ee_pose = init_poses[task_name][seq_name][0]["right_ee"]
        return left_dof, right_dof, left_ee_pose, right_ee_pose

    def reset_and_optional_init():
        if args.start_mode == "benchmark" and env_cfg.randomize and bench_randomize_idx is not None:
            env.cfg.randomize_idx = bench_randomize_idx
        env_results_local = env.reset()
        box_pose_before_warmup = None
        if "object_pose" in env_results_local[0]:
            box_pose_before_warmup = env_results_local[0]["object_pose"][0].detach().cpu().numpy().copy()
        left_ik_controller.reset()
        right_ik_controller.reset()

        if bool(args.preserve_usd_initial_ee):
            print("[Teleop] preserve_usd_initial_ee=1: skip eval-init warmup; keep USD reset EE pose.")
        elif args.start_mode in ("eval_init", "benchmark"):
            try:
                if args.start_mode == "benchmark" and bench_seq_name is not None:
                    seq_name = bench_seq_name
                else:
                    init_episode_list = TASK_INIT_EPISODE[task_name]
                    rank = max(0, min(args.init_episode_rank, len(init_episode_list) - 1))
                    seq_name = init_episode_list[rank][0]
                left_dof, right_dof, left_ee_pose, right_ee_pose = load_eval_init_target(seq_name)

                for _ in range(max(1, int(args.init_warmup_steps))):
                    ik_step(
                        env,
                        left_ik_controller,
                        right_ik_controller,
                        left_ik_commands_world,
                        right_ik_commands_world,
                        left_ik_commands_robot,
                        right_ik_commands_robot,
                        left_ee_pose,
                        right_ee_pose,
                        left_dof,
                        right_dof,
                        action,
                        ignore_orientation=ignore_orientation,
                        active_arm="both",
                        tcp_offset_enable=False,
                        left_tcp_offset=(0.0, 0.0, 0.0),
                        right_tcp_offset=(0.0, 0.0, 0.0),
                    )
                    env_results_local = env.step(action)
                print(f"[Teleop] start_mode={args.start_mode} loaded: task={task_name}, seq={seq_name}")
            except Exception as e:
                print(f"[Teleop] {args.start_mode} init failed, fallback to reset pose. reason={e}")

        box_pose_after_warmup = None
        if "object_pose" in env_results_local[0]:
            box_pose_after_warmup = env_results_local[0]["object_pose"][0].detach().cpu().numpy().copy()
        if box_pose_before_warmup is not None and box_pose_after_warmup is not None:
            warmup_box_shift = float(np.linalg.norm(box_pose_after_warmup[:3] - box_pose_before_warmup[:3]))
            print(f"[Teleop] init warmup box shift={warmup_box_shift:.4f} m")

        # Teleop-friendly protection: keep box at reset pose so user starts from a clean state.
        if (
            (not bool(args.preserve_usd_initial_ee))
            and
            bool(args.lock_box_after_init)
            and box_pose_before_warmup is not None
            and hasattr(env, "box")
        ):
            env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
            box_root_state = env.box.data.default_root_state[env_ids, :].clone()
            box_pose_before_t = torch.as_tensor(
                box_pose_before_warmup, dtype=box_root_state.dtype, device=env.device
            ).unsqueeze(0)
            box_root_state[:, 0:3] = box_pose_before_t[:, 0:3] + env.scene.env_origins[env_ids, 0:3]
            box_root_state[:, 3:7] = box_pose_before_t[:, 3:7]
            box_root_state[:, 7:13] = 0.0
            env.box.write_root_state_to_sim(box_root_state, env_ids=env_ids)
            print("[Teleop] restored box pose after init warmup.")

        left_curr_pose = env_results_local[0]["left_ee_pose"][0].detach().cpu().numpy().copy()
        right_curr_pose = env_results_local[0]["right_ee_pose"][0].detach().cpu().numpy().copy()
        left_curr_pose[3:7] = quat_wxyz_normalize(left_curr_pose[3:7])
        right_curr_pose[3:7] = quat_wxyz_normalize(right_curr_pose[3:7])
        if tcp_offset_enable:
            left_target_pose = ee_link_pose_to_tcp_pose(left_curr_pose, left_tcp_offset)
            right_target_pose = ee_link_pose_to_tcp_pose(right_curr_pose, right_tcp_offset)
        else:
            left_target_pose = left_curr_pose
            right_target_pose = right_curr_pose

        # Optional retreat mode (disabled when preserving USD reset EE pose).
        if not bool(args.preserve_usd_initial_ee):
            left_retreat = np.array([float(args.post_init_left_retreat_x), float(args.post_init_left_retreat_y), 0.0], dtype=np.float64)
            right_retreat = np.array([float(args.post_init_right_retreat_x), float(args.post_init_right_retreat_y), 0.0], dtype=np.float64)
            retreat_steps = max(0, int(args.post_init_retreat_steps))
            if retreat_steps > 0 and (np.linalg.norm(left_retreat) > 1e-9 or np.linalg.norm(right_retreat) > 1e-9):
                left_start = left_target_pose.copy()
                right_start = right_target_pose.copy()
                left_goal = left_start.copy()
                right_goal = right_start.copy()
                left_goal[:3] = left_goal[:3] + left_retreat
                right_goal[:3] = right_goal[:3] + right_retreat

                for s in range(retreat_steps):
                    alpha = float(s + 1) / float(retreat_steps)
                    left_cmd = left_start.copy()
                    right_cmd = right_start.copy()
                    left_cmd[:3] = left_start[:3] + alpha * (left_goal[:3] - left_start[:3])
                    right_cmd[:3] = right_start[:3] + alpha * (right_goal[:3] - right_start[:3])
                    ik_step(
                        env,
                        left_ik_controller,
                        right_ik_controller,
                        left_ik_commands_world,
                        right_ik_commands_world,
                        left_ik_commands_robot,
                        right_ik_commands_robot,
                        left_cmd,
                        right_cmd,
                        left_hand_dof,
                        right_hand_dof,
                        action,
                        ignore_orientation=ignore_orientation,
                        active_arm="both",
                        tcp_offset_enable=tcp_offset_enable,
                        left_tcp_offset=left_tcp_offset,
                        right_tcp_offset=right_tcp_offset,
                    )
                    env_results_local = env.step(action)

                left_target_pose = left_goal
                right_target_pose = right_goal
                print(
                    "[Teleop] post-init retreat applied: "
                    f"left(dx={left_retreat[0]:.3f},dy={left_retreat[1]:.3f}) "
                    f"right(dx={right_retreat[0]:.3f},dy={right_retreat[1]:.3f}), "
                    f"steps={retreat_steps}"
                )

        if "object_pose" in env_results_local[0]:
            box0 = env_results_local[0]["object_pose"][0].detach().cpu().numpy()[:3]
            if hasattr(env, "goal_pos_w"):
                goal0 = (
                    env.goal_pos_w[0].detach().cpu() - env.scene.env_origins[0].detach().cpu()
                ).numpy()
                print(
                    f"[Teleop] reset scene: box=({box0[0]:.3f},{box0[1]:.3f},{box0[2]:.3f}) "
                    f"goal=({goal0[0]:.3f},{goal0[1]:.3f},{goal0[2]:.3f})"
                )
            else:
                print(f"[Teleop] reset scene: box=({box0[0]:.3f},{box0[1]:.3f},{box0[2]:.3f})")
        return env_results_local, left_target_pose, right_target_pose

    env_results, left_target, right_target = reset_and_optional_init()
    right_reset_quat = quat_wxyz_normalize(right_target[3:7].copy())

    def check_right_workspace(target_pose: np.ndarray):
        warnings = []
        x, y, z = [float(v) for v in target_pose[:3]]
        if x < args.right_ws_x_min or x > args.right_ws_x_max:
            warnings.append(f"x={x:.3f} out [{args.right_ws_x_min:.3f},{args.right_ws_x_max:.3f}]")
        if y < args.right_ws_y_min or y > args.right_ws_y_max:
            warnings.append(f"y={y:.3f} out [{args.right_ws_y_min:.3f},{args.right_ws_y_max:.3f}]")
        if z < args.right_ws_z_min or z > args.right_ws_z_max:
            warnings.append(f"z={z:.3f} out [{args.right_ws_z_min:.3f},{args.right_ws_z_max:.3f}]")

        q_curr = quat_wxyz_normalize(target_pose[3:7])
        dot = float(np.clip(np.abs(np.dot(right_reset_quat, q_curr)), -1.0, 1.0))
        ang_deg = float(np.degrees(2.0 * np.arccos(dot)))
        if ang_deg > float(args.right_ws_rot_max_deg):
            warnings.append(f"rot_delta={ang_deg:.1f}deg > {float(args.right_ws_rot_max_deg):.1f}deg")
        return warnings, ang_deg

    print("[Teleop] Ready. Focus the OpenCV window and use keys.")
    print("[Teleop] Move: W/S(+/-X), A/D(+/-Y), R/F(+/-Z)")
    if ignore_orientation:
        print("[Teleop] Rotate keys disabled in position-only mode.")
    else:
        print("[Teleop] Rotate: U/O(+/-Roll), I/K(+/-Pitch), J/L(+/-Yaw)")
    if bool(args.right_only_mode):
        print("[Teleop] Right-only mode ON: only right arm moves. P=sync, N=reset, Q=quit")
    else:
        print("[Teleop] Arm: 1=left, 2=right, 3=both. P=sync target to current, N=reset, Q=quit")
    print(f"[Teleop] start_mode={args.start_mode}, env_randomize={env_cfg.randomize}")

    active_arm_runtime = args.ik_active_arm
    if bool(args.right_only_mode):
        active_arm_runtime = "right"
    dt = 1.0 / max(1e-3, float(args.loop_hz))
    rot_step_rad = np.deg2rad(float(args.rot_step_deg))
    step_idx = 0
    last_warn_step = -999999

    window_name = "Teleop Fixed RGB (No Record)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while simulation_app.is_running():
        tic = time.time()

        rgb_obs = env_results[0]["fixed_rgb"][0].detach().cpu().numpy()
        rgb_obs = to_uint8_rgb(rgb_obs)

        box_xy = None
        success_flag = -1
        reach_flag = -1
        if "object_pose" in env_results[0]:
            box_xy = env_results[0]["object_pose"][0].detach().cpu().numpy()[:2]
        if "success" in env_results[0]:
            success_flag = int(env_results[0]["success"][0].detach().cpu().item())
        if "reach_success" in env_results[0]:
            reach_flag = int(env_results[0]["reach_success"][0].detach().cpu().item())
        right_curr = env_results[0]["right_ee_pose"][0].detach().cpu().numpy()
        right_curr_target_frame = right_curr.copy()
        if tcp_offset_enable:
            right_curr_target_frame = ee_link_pose_to_tcp_pose(right_curr_target_frame, right_tcp_offset)

        ws_warnings = []
        ws_rot_delta_deg = 0.0
        if bool(args.workspace_warn_enable):
            ws_warnings, ws_rot_delta_deg = check_right_workspace(right_target)

        controls_line = "Move: W/S A/D R/F"
        if not ignore_orientation:
            controls_line += " | Rot: U/O I/K J/L"
        controls_line += " | P sync | N reset | Q quit"
        if not bool(args.right_only_mode):
            controls_line = controls_line.replace(" | P sync", " | 1/2/3 arm | P sync")

        lines = [
            f"step={step_idx} active_arm={active_arm_runtime}",
            f"pos_step={args.pos_step:.3f}m rot_step={args.rot_step_deg:.1f}deg",
            f"reach={reach_flag} success={success_flag}",
            controls_line,
        ]
        if box_xy is not None:
            lines.append(
                f"box_xy=({box_xy[0]:.3f},{box_xy[1]:.3f}) "
                f"right_curr_xyz=({right_curr_target_frame[0]:.3f},{right_curr_target_frame[1]:.3f},{right_curr_target_frame[2]:.3f})"
            )
        if bool(args.workspace_warn_enable):
            lines.append(
                f"right_target_xyz=({right_target[0]:.3f},{right_target[1]:.3f},{right_target[2]:.3f}) rot_delta={ws_rot_delta_deg:.1f}deg"
            )
            if ws_warnings:
                lines.append("[WARN] target out of configured task-space bounds")
                for item in ws_warnings[:2]:
                    lines.append(f"[WARN] {item}")
                if step_idx - last_warn_step >= max(1, int(args.loop_hz)):
                    print("[Teleop][WARN] target out of task-space: " + " | ".join(ws_warnings))
                    last_warn_step = step_idx

        vis = overlay_info(rgb_obs, lines)
        cv2.imshow(window_name, vis[:, :, ::-1])  # BGR for OpenCV
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("1"):
            if bool(args.right_only_mode):
                print("[Teleop] right_only_mode is on; ignore key '1'.")
            else:
                active_arm_runtime = "left"
                print("[Teleop] active_arm=left")
        elif key == ord("2"):
            active_arm_runtime = "right"
            print("[Teleop] active_arm=right")
        elif key == ord("3"):
            if bool(args.right_only_mode):
                print("[Teleop] right_only_mode is on; ignore key '3'.")
            else:
                active_arm_runtime = "both"
                print("[Teleop] active_arm=both")
        elif key == ord("n"):
            env_results, left_target, right_target = reset_and_optional_init()
            right_reset_quat = quat_wxyz_normalize(right_target[3:7].copy())
            if bool(args.right_only_mode):
                active_arm_runtime = "right"
            print("[Teleop] scene reset")
            step_idx = 0
            continue
        elif key == ord("p"):
            left_curr_pose = env_results[0]["left_ee_pose"][0].detach().cpu().numpy().copy()
            right_curr_pose = env_results[0]["right_ee_pose"][0].detach().cpu().numpy().copy()
            left_curr_pose[3:7] = quat_wxyz_normalize(left_curr_pose[3:7])
            right_curr_pose[3:7] = quat_wxyz_normalize(right_curr_pose[3:7])
            if tcp_offset_enable:
                left_target = ee_link_pose_to_tcp_pose(left_curr_pose, left_tcp_offset)
                right_target = ee_link_pose_to_tcp_pose(right_curr_pose, right_tcp_offset)
            else:
                left_target = left_curr_pose
                right_target = right_curr_pose
            print("[Teleop] target pose synced to current EE pose")

        dp = np.zeros(3, dtype=np.float64)
        drpy = np.zeros(3, dtype=np.float64)  # roll, pitch, yaw

        if key == ord("w"):
            dp[0] += args.pos_step
        elif key == ord("s"):
            dp[0] -= args.pos_step
        elif key == ord("a"):
            dp[1] += args.pos_step
        elif key == ord("d"):
            dp[1] -= args.pos_step
        elif key == ord("r"):
            dp[2] += args.pos_step
        elif key == ord("f"):
            dp[2] -= args.pos_step
        elif (not ignore_orientation) and key == ord("u"):
            drpy[0] += rot_step_rad
        elif (not ignore_orientation) and key == ord("o"):
            drpy[0] -= rot_step_rad
        elif (not ignore_orientation) and key == ord("i"):
            drpy[1] += rot_step_rad
        elif (not ignore_orientation) and key == ord("k"):
            drpy[1] -= rot_step_rad
        elif (not ignore_orientation) and key == ord("j"):
            drpy[2] += rot_step_rad
        elif (not ignore_orientation) and key == ord("l"):
            drpy[2] -= rot_step_rad

        def apply_delta(target_pose: np.ndarray):
            target_pose[:3] = target_pose[:3] + dp
            if np.linalg.norm(drpy) > 0:
                target_pose[3:7] = quat_wxyz_apply_rpy_delta(
                    target_pose[3:7], drpy[0], drpy[1], drpy[2]
                )
            return target_pose

        if active_arm_runtime in ("left", "both"):
            left_target = apply_delta(left_target)
        if active_arm_runtime in ("right", "both"):
            right_target = apply_delta(right_target)

        ik_step(
            env,
            left_ik_controller,
            right_ik_controller,
            left_ik_commands_world,
            right_ik_commands_world,
            left_ik_commands_robot,
            right_ik_commands_robot,
            left_target,
            right_target,
            left_hand_dof,
            right_hand_dof,
            action,
            ignore_orientation=ignore_orientation,
            active_arm=active_arm_runtime,
            tcp_offset_enable=tcp_offset_enable,
            left_tcp_offset=left_tcp_offset,
            right_tcp_offset=right_tcp_offset,
        )
        env_results = env.step(action)

        step_idx += 1
        spent = time.time() - tic
        if spent < dt:
            time.sleep(dt - spent)

    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
