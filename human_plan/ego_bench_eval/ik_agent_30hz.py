
import os
import tqdm

from transformers import HfArgumentParser
from human_plan.vila_train.args import (
  VLATrainingArguments, VLAModelArguments, VLADataArguments
)

from collections import deque
from omni.isaac.lab.app import AppLauncher

# We fix the seed for tasks to make sure the object position during evaluation
# are never seen during training.
seed_map = {
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

parser = HfArgumentParser((VLAModelArguments, VLADataArguments, VLATrainingArguments))
# add argparse arguments
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--room_idx", type=int, default=None, help="Room Idx")
parser.add_argument("--table_idx", type=int, default=None, help="Table Idx")
parser.add_argument("--smooth_weight", type=float, default=None, help="smooth weight")
parser.add_argument("--hand_smooth_weight", type=float, default=None, help="smooth weight")
parser.add_argument("--num_episodes", type=int, default=None, help="episode_label")
parser.add_argument("--num_trials", type=int, default=None, help="trial label")
parser.add_argument("--result_saving_path", type=str, default=None, help="result saving path")
parser.add_argument("--video_saving_path", type=str, default=None, help="video saving path")
parser.add_argument("--save_frames", type=int, default=0, help="result saving path")
parser.add_argument("--project_trajs", type=int, default=0, help="result saving path")
parser.add_argument("--additional_label", type=str, default=None, help="additional_label")

# 1.用来获取图像，20260318
parser.add_argument("--save_input_obs", type=int, default=0, help="whether to save raw input observations")
parser.add_argument("--input_obs_stride", type=int, default=10, help="save one input frame every N env steps")
parser.add_argument("--input_obs_max", type=int, default=120, help="maximum number of input frames to save per trial")
parser.add_argument("--input_obs_dir", type=str, default="/home/ubuntu/Desktop/Egovla/EgoVLA_Release/observe_image", help="optional root dir for saving input frames")
parser.add_argument("--debug_ik", type=int, default=0, help="print IK tracking diagnostics")
parser.add_argument("--debug_ik_stride", type=int, default=10, help="print diagnostics every N steps")
parser.add_argument("--debug_ik_csv", type=str, default=None, help="optional directory for saving per-trial IK diagnostics csv")
parser.add_argument(
    "--ik_ignore_orientation",
    type=int,
    default=1,
    help="if set to 1, ignore orientation targets in IK and prioritize EE position tracking",
)
parser.add_argument(
    "--ik_active_arm",
    type=str,
    default="both",
    help="active arm mode for IK: both, left, or right",
)
parser.add_argument("--robot_base_x", type=float, default=None, help="optional robot base x override in world frame")
parser.add_argument("--robot_base_y", type=float, default=None, help="optional robot base y override in world frame")
parser.add_argument("--robot_base_z", type=float, default=None, help="optional robot base z override in world frame")
parser.add_argument("--box_init_x", type=float, default=None, help="optional push-box initial x override")
parser.add_argument("--box_init_y", type=float, default=None, help="optional push-box initial y override")
parser.add_argument("--box_init_z", type=float, default=None, help="optional push-box initial z override")
parser.add_argument("--goal_x", type=float, default=None, help="optional push-box goal x override")
parser.add_argument("--goal_y", type=float, default=None, help="optional push-box goal y override")
parser.add_argument("--goal_z", type=float, default=None, help="optional push-box goal z override")
parser.add_argument("--ik_tcp_offset_enable", type=int, default=1, help="enable TCP->link7 translational offset compensation")
parser.add_argument("--ik_left_tcp_offset_x", type=float, default=0.0, help="left TCP offset x in link7 local frame")
parser.add_argument("--ik_left_tcp_offset_y", type=float, default=0.0, help="left TCP offset y in link7 local frame")
parser.add_argument("--ik_left_tcp_offset_z", type=float, default=0.1654, help="left TCP offset z in link7 local frame")
parser.add_argument("--ik_right_tcp_offset_x", type=float, default=0.0, help="right TCP offset x in link7 local frame")
parser.add_argument("--ik_right_tcp_offset_y", type=float, default=0.0, help="right TCP offset y in link7 local frame")
parser.add_argument("--ik_right_tcp_offset_z", type=float, default=0.1654, help="right TCP offset z in link7 local frame")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# launch omniverse app
app_launcher = AppLauncher(enable_cameras=True, device="cuda", headless=True)
simulation_app = app_launcher.app

from human_plan.ego_bench_eval.utils import (
    process_input,
    ik_step,
    ik_eval_single_step,
    get_language_instruction
)
import gymnasium as gym
import torch

from omni.isaac.lab_tasks.utils import parse_env_cfg
import torch

from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.markers import VisualizationMarkers
# from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
# from omni.isaac.lab.utils.math import subtract_frame_transforms
from humanoid.tasks.base_env import BaseEnv, BaseEnvCfg

import cv2
from human_plan.vila_eval.utils.load_model import load_model_eval

def main():

    model_args, data_args, training_args, task_args = parser.parse_args_into_dataclasses()

    model, tokenizer, model_args, data_args, training_args = load_model_eval(
      model_args, data_args, training_args
    )
    model.to("cuda")

    data_args.sep_query_token = model_args.sep_query_token

    import random
    assert task_args.task in seed_map
    print(f"Setting seed to {seed_map[task_args.task]}")
    random.seed(seed_map[task_args.task])
    randomize_idxes = list(range(10000))
    random.shuffle(randomize_idxes)

    # train data are using idxes 0-49, test start from 50
    set_selection = "Test"
    if set_selection == "Train":
      # curr_random_idx = 0 + task_args.room_idx * task_args.num_trials * task_args.num_episodes
      assert False
    elif set_selection == "Test":
      # We used the first 100 random idxes for training
      # Starting from 101th random ides for evaluation
      curr_random_idx = 100 + (
        task_args.room_idx * 5 + task_args.table_idx
      )  * task_args.num_trials * task_args.num_episodes

    # parse configuration
    env_cfg: BaseEnvCfg = parse_env_cfg(
        task_args.task,
        num_envs=1,
    )

    # Optional workspace overrides for fast calibration without editing task files.
    if any(v is not None for v in (task_args.robot_base_x, task_args.robot_base_y, task_args.robot_base_z)):
      default_robot_pos = list(env_cfg.robot.init_state.pos)
      robot_pos = (
        task_args.robot_base_x if task_args.robot_base_x is not None else default_robot_pos[0],
        task_args.robot_base_y if task_args.robot_base_y is not None else default_robot_pos[1],
        task_args.robot_base_z if task_args.robot_base_z is not None else default_robot_pos[2],
      )
      env_cfg.robot.init_state.pos = robot_pos
      print(f"[WS] robot base override: {robot_pos}")

    if hasattr(env_cfg, "box") and any(v is not None for v in (task_args.box_init_x, task_args.box_init_y, task_args.box_init_z)):
      default_box_pos = list(env_cfg.box.init_state.pos)
      box_pos = (
        task_args.box_init_x if task_args.box_init_x is not None else default_box_pos[0],
        task_args.box_init_y if task_args.box_init_y is not None else default_box_pos[1],
        task_args.box_init_z if task_args.box_init_z is not None else default_box_pos[2],
      )
      env_cfg.box.init_state.pos = box_pos
      print(f"[WS] box init override: {box_pos}")

    if hasattr(env_cfg, "goal_default_pos") and any(v is not None for v in (task_args.goal_x, task_args.goal_y, task_args.goal_z)):
      default_goal_pos = list(env_cfg.goal_default_pos)
      goal_pos = (
        task_args.goal_x if task_args.goal_x is not None else default_goal_pos[0],
        task_args.goal_y if task_args.goal_y is not None else default_goal_pos[1],
        task_args.goal_z if task_args.goal_z is not None else default_goal_pos[2],
      )
      env_cfg.goal_default_pos = goal_pos
      print(f"[WS] goal override: {goal_pos}")

    env_cfg.episode_length_s = 60 # 60 seconds episode length -> For long horizon tasks
    env_cfg.randomize = True
    # create environment
    env_cfg.spawn_background =True
    # select background
    room_idx = task_args.room_idx
    table_idx = task_args.table_idx
    env_cfg.room_idx = room_idx
    env_cfg.table_idx = table_idx
    env: BaseEnv = gym.make(
        task_args.task,
        cfg=env_cfg
    )
    env.cfg.randomize_idx = randomize_idxes[curr_random_idx]
    env.reset()

    ik_active_arm = str(task_args.ik_active_arm).lower()
    if ik_active_arm not in ("both", "left", "right"):
      print(f"[IK] invalid --ik_active_arm={task_args.ik_active_arm}, fallback to 'both'")
      ik_active_arm = "both"
    print(
      f"[IK] config: ignore_orientation={bool(task_args.ik_ignore_orientation)} "
      f"active_arm={ik_active_arm}"
    )
    left_joint_ids = env.cfg.left_arm_cfg.joint_ids
    right_joint_ids = env.cfg.right_arm_cfg.joint_ids
    left_joint_names = list(getattr(env.cfg.left_arm_cfg, "joint_names", []))
    right_joint_names = list(getattr(env.cfg.right_arm_cfg, "joint_names", []))
    if len(left_joint_names) != len(left_joint_ids):
      left_joint_names = [f"left_joint_{int(jid)}" for jid in left_joint_ids]
    if len(right_joint_names) != len(right_joint_ids):
      right_joint_names = [f"right_joint_{int(jid)}" for jid in right_joint_ids]
    left_tcp_offset = (
      float(task_args.ik_left_tcp_offset_x),
      float(task_args.ik_left_tcp_offset_y),
      float(task_args.ik_left_tcp_offset_z),
    )
    right_tcp_offset = (
      float(task_args.ik_right_tcp_offset_x),
      float(task_args.ik_right_tcp_offset_y),
      float(task_args.ik_right_tcp_offset_z),
    )
    print(
      f"[IK] tcp_offset_enable={bool(task_args.ik_tcp_offset_enable)} "
      f"left_offset={left_tcp_offset} right_offset={right_tcp_offset}"
    )

    # IK controllers
    command_type = "pose"
    left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)

    # Create buffers to store actions
    left_ik_commands_world = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    left_ik_commands_robot = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_world = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_robot = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    action = torch.zeros((env.scene.num_envs, env.num_actions), device=env.robot.device)

    save_path = os.path.join(
      task_args.video_saving_path,
      task_args.additional_label,
      f"inference_{task_args.smooth_weight}_{task_args.hand_smooth_weight}"
    )

    from pathlib import Path
    Path(save_path).mkdir(exist_ok=True, parents=True)

    import pickle
    with open("init_poses_fixed_set_100traj.pkl", "rb") as f:
       init_poses = pickle.load(f)

    task_name = task_args.task[9:-3]
    load_name = task_name

    # Collect Initial Hand and EE poses from data -> Only set the arm and hand for start
    from human_plan.ego_bench_eval.utils import TASK_INIT_EPISODE
    episode_list = TASK_INIT_EPISODE[task_name][:task_args.num_episodes]

    hist_len = data_args.predict_future_step * data_args.future_index

    import numpy as np
    def quat_angle_error_wxyz(q_curr, q_goal):
      """Return geodesic quaternion angle (radians)."""
      q_curr = np.asarray(q_curr, dtype=np.float64)
      q_goal = np.asarray(q_goal, dtype=np.float64)
      q_curr = q_curr / (np.linalg.norm(q_curr) + 1e-12)
      q_goal = q_goal / (np.linalg.norm(q_goal) + 1e-12)
      dot = np.clip(np.abs(np.dot(q_curr, q_goal)), -1.0, 1.0)
      return 2.0 * np.arccos(dot)

    cam_intrinsics = np.array([
      [488.6662,   0.0000, 640.0000],
      [  0.0000, 488.6662, 360.0000],
      [  0.0000,   0.0000,   1.0000]
    ])

    padding = 0

    # with torch.inference_mode():
    for episode_idx in episode_list:
      for trial_idx in range(task_args.num_trials):
        # seq_name = f"episode_{episode_idx}.hdf5"
        seq_name = episode_idx[0]

        # 30 Hz
        rgb_obs_hist = deque(maxlen=120)
        # original video is 15fps and env is 30 fps
        action_hist_left_ee = deque(maxlen=hist_len)
        action_hist_right_ee = deque(maxlen=hist_len)
        action_hist_left_hand = deque(maxlen=hist_len)
        action_hist_right_hand = deque(maxlen=hist_len)

        seq_save_path = os.path.join(
          save_path,
          task_name,
          f"room_{room_idx}",
          f"table_{table_idx}",
        )
        from pathlib import Path
        Path(seq_save_path).mkdir(exist_ok=True, parents=True)
        output_path = os.path.join(
          seq_save_path,
          f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}_{trial_idx}.mp4"
        )
        if task_args.save_frames:
          frames_output_path = os.path.join(
            seq_save_path,
            f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}_{trial_idx}"
          )
          Path(frames_output_path).mkdir(exist_ok=True, parents=True)

        #Test
        print("I have already run here")
        print(task_args.save_input_obs)
          
        #2.同样为了获取图像，20260318
        if task_args.save_input_obs:
          input_obs_root = task_args.input_obs_dir if task_args.input_obs_dir is not None else seq_save_path
          input_obs_output_path = os.path.join(
            input_obs_root,
            f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}_{trial_idx}_input_obs"
          )
          Path(input_obs_output_path).mkdir(exist_ok=True, parents=True)
          input_obs_saved_count = 0    

        fps = 15
        out = cv2.VideoWriter(
          output_path, 
          #  seq_save_path,
          cv2.VideoWriter_fourcc(*"mp4v"), 
          fps, (1280, 720)
        )

        # def init_env():
        if True:
            # reset
          curr_random_idx += 1
          env.cfg.randomize_idx = randomize_idxes[curr_random_idx]
          env_results = env.reset()
          left_ik_controller.reset()
          right_ik_controller.reset()
          padding_idx = padding
          # for padding_idx in range(padding):

          left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
          right_dof = init_poses[load_name][seq_name][padding]["right_dof"]

          for idx in range(100):
            left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
            right_dof = init_poses[load_name][seq_name][padding]["right_dof"]
            
            left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
            right_dof = init_poses[load_name][seq_name][padding]["right_dof"]
            
            left_ee_pose_traj_gt = init_poses[load_name][seq_name][padding]["left_ee"]
            right_ee_pose_traj_gt = init_poses[load_name][seq_name][padding]["right_ee"]

            ik_step(
              env,
              left_ik_controller,
              right_ik_controller,

              left_ik_commands_world, 
              right_ik_commands_world,
              
              left_ik_commands_robot,
              right_ik_commands_robot,

              left_ee_pose_traj_gt, right_ee_pose_traj_gt,
              left_dof, right_dof,
              action,
              ignore_orientation=bool(task_args.ik_ignore_orientation),
              active_arm=ik_active_arm,
              tcp_offset_enable=False,
              left_tcp_offset=(0.0, 0.0, 0.0),
              right_tcp_offset=(0.0, 0.0, 0.0),
            )
            env_results = env.step(action)
            rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
            rgb_obs = cv2.resize(rgb_obs, (384, 384))
        rgb_obs_hist.append(rgb_obs)
        count = padding

        result = False
        ik_debug_rows = []
        from human_plan.ego_bench_eval.utils import TASK_MAX_HORIZON
        max_horizon = TASK_MAX_HORIZON[task_args.task]

        for i in tqdm.tqdm(range(max_horizon)):
          # run everything in inference mode
          # obtain quantities from simulation
          rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
          
          #3.在主循环中获取图像，20260318
          if (
            task_args.save_input_obs
            and input_obs_saved_count < task_args.input_obs_max
            and (i % max(1, task_args.input_obs_stride) == 0)
          ):
            print("开始录像了！")
            rgb_obs_to_save = rgb_obs
            if rgb_obs_to_save.dtype != np.uint8:
              rgb_obs_to_save = np.clip(rgb_obs_to_save, 0.0, 255.0)
              if rgb_obs_to_save.max() <= 1.0:
                rgb_obs_to_save = rgb_obs_to_save * 255.0
              rgb_obs_to_save = rgb_obs_to_save.astype(np.uint8)
            cv2.imwrite(
              os.path.join(input_obs_output_path, f"step_{i:04d}.jpg"),
              rgb_obs_to_save[:, :, ::-1]
            )
            input_obs_saved_count += 1

          from human_plan.ego_bench_eval.utils import process_proprio_input

          proprio_input, raw_proprio_inputs = process_proprio_input(
            env_results[0]["left_finger_tip_pos"].cpu().numpy(),
            env_results[0]["right_finger_tip_pos"].cpu().numpy(),
            env_results[0]["left_ee_pose"].cpu().numpy(),
            env_results[0]["right_ee_pose"].cpu().numpy(),
            env_results[0]["qpos"],
            cam_intrinsics,
            input_hand_dof=data_args.input_hand_dof
          )

          rgb_obs = cv2.resize(rgb_obs, (384, 384))
          rgb_obs_hist.append(rgb_obs)

          raw_language_instruction = get_language_instruction(
              task_args.task
          )

          raw_data_dict = process_input(
              rgb_obs_hist, 
              proprio_input.to("cuda"),
              raw_language_instruction,
              data_args, model_args, tokenizer
          )

          raw_data_dict.update(raw_proprio_inputs)
          with torch.inference_mode():
            action_dict = ik_eval_single_step(
                raw_data_dict,
                model, tokenizer,
            )

          from human_plan.ego_bench_eval.utils import smooth_action, repeat_action
          action_hist_right_ee.append(
            repeat_action(action_dict["right_ee_pose"], data_args.future_index)
          )
          action_hist_left_ee.append(
            repeat_action(action_dict["left_ee_pose"], data_args.future_index)
          )

          action_hist_left_hand.append(
            repeat_action(action_dict["left_qpos_multi_step"], data_args.future_index)
          )
          action_hist_right_hand.append(
            repeat_action(action_dict["right_qpos_multi_step"], data_args.future_index)
          )

          action_left_ee = smooth_action(
            hist_len, task_args.smooth_weight, action_hist_left_ee
          )

          action_right_ee = smooth_action(
            hist_len, task_args.smooth_weight, action_hist_right_ee
          )

          action_left_hand = smooth_action(
            hist_len, task_args.hand_smooth_weight, action_hist_left_hand
          )
          action_right_hand = smooth_action(
            hist_len, task_args.hand_smooth_weight, action_hist_right_hand
          )

          ik_step(
              env,
              left_ik_controller,
              right_ik_controller,

              left_ik_commands_world,
              right_ik_commands_world,
              
              left_ik_commands_robot,
              right_ik_commands_robot,

              action_left_ee,
              action_right_ee,

              action_left_hand,
              action_right_hand,

              action,
              ignore_orientation=bool(task_args.ik_ignore_orientation),
              active_arm=ik_active_arm,
              tcp_offset_enable=bool(task_args.ik_tcp_offset_enable),
              left_tcp_offset=left_tcp_offset,
              right_tcp_offset=right_tcp_offset,
          )
          env_results = env.step(action)

          if task_args.debug_ik:
            left_curr = env_results[0]["left_ee_pose"][0].detach().cpu().numpy()
            right_curr = env_results[0]["right_ee_pose"][0].detach().cpu().numpy()
            left_goal = np.asarray(action_left_ee[:7], dtype=np.float64)
            right_goal = np.asarray(action_right_ee[:7], dtype=np.float64)

            left_pos_err = float(np.linalg.norm(left_curr[:3] - left_goal[:3]))
            right_pos_err = float(np.linalg.norm(right_curr[:3] - right_goal[:3]))
            left_rot_err_rad = float(quat_angle_error_wxyz(left_curr[3:7], left_goal[3:7]))
            right_rot_err_rad = float(quat_angle_error_wxyz(right_curr[3:7], right_goal[3:7]))

            qpos_now = env_results[0]["qpos"][0]
            left_joint_track_err = float(torch.mean(torch.abs(action[0, left_joint_ids] - qpos_now[left_joint_ids])).item())
            right_joint_track_err = float(torch.mean(torch.abs(action[0, right_joint_ids] - qpos_now[right_joint_ids])).item())

            lower = env.robot_dof_lower_limits
            upper = env.robot_dof_upper_limits
            near_limit_eps = 1e-2

            left_q = qpos_now[left_joint_ids]
            right_q = qpos_now[right_joint_ids]
            left_lower_margin = left_q - lower[left_joint_ids]
            left_upper_margin = upper[left_joint_ids] - left_q
            right_lower_margin = right_q - lower[right_joint_ids]
            right_upper_margin = upper[right_joint_ids] - right_q
            left_margin = torch.minimum(left_lower_margin, left_upper_margin)
            right_margin = torch.minimum(right_lower_margin, right_upper_margin)

            left_near_mask = (left_lower_margin <= near_limit_eps) | (left_upper_margin <= near_limit_eps)
            right_near_mask = (right_lower_margin <= near_limit_eps) | (right_upper_margin <= near_limit_eps)
            left_near_limit_ratio = float(torch.mean(left_near_mask.float()).item())
            right_near_limit_ratio = float(torch.mean(right_near_mask.float()).item())
            left_near_limit_count = int(left_near_mask.sum().item())
            right_near_limit_count = int(right_near_mask.sum().item())

            left_worst_idx = int(torch.argmin(left_margin).item())
            right_worst_idx = int(torch.argmin(right_margin).item())
            left_worst_joint = left_joint_names[left_worst_idx]
            right_worst_joint = right_joint_names[right_worst_idx]
            left_worst_side = "lower" if float(left_lower_margin[left_worst_idx].item()) <= float(left_upper_margin[left_worst_idx].item()) else "upper"
            right_worst_side = "lower" if float(right_lower_margin[right_worst_idx].item()) <= float(right_upper_margin[right_worst_idx].item()) else "upper"
            left_min_margin = float(left_margin[left_worst_idx].item())
            right_min_margin = float(right_margin[right_worst_idx].item())
            left_margin_values = ";".join([f"{v:.6f}" for v in left_margin.detach().cpu().tolist()])
            right_margin_values = ";".join([f"{v:.6f}" for v in right_margin.detach().cpu().tolist()])

            if i % max(1, task_args.debug_ik_stride) == 0:
              print(
                f"[IK-DEBUG] step={i} "
                f"L_pos={left_pos_err:.4f}m L_rot={np.degrees(left_rot_err_rad):.2f}deg "
                f"R_pos={right_pos_err:.4f}m R_rot={np.degrees(right_rot_err_rad):.2f}deg "
                f"L_qerr={left_joint_track_err:.4f} R_qerr={right_joint_track_err:.4f} "
                f"L_lim={left_near_limit_ratio:.2f}({left_near_limit_count}) "
                f"R_lim={right_near_limit_ratio:.2f}({right_near_limit_count}) "
                f"L_worst={left_worst_joint}@{left_worst_side}:{left_min_margin:.4f} "
                f"R_worst={right_worst_joint}@{right_worst_side}:{right_min_margin:.4f}"
              )

            if task_args.debug_ik_csv is not None:
              ik_debug_rows.append({
                "step": i,
                "left_pos_err_m": left_pos_err,
                "left_rot_err_deg": float(np.degrees(left_rot_err_rad)),
                "right_pos_err_m": right_pos_err,
                "right_rot_err_deg": float(np.degrees(right_rot_err_rad)),
                "left_joint_track_err": left_joint_track_err,
                "right_joint_track_err": right_joint_track_err,
                "left_near_limit_ratio": left_near_limit_ratio,
                "right_near_limit_ratio": right_near_limit_ratio,
                "left_near_limit_count": left_near_limit_count,
                "right_near_limit_count": right_near_limit_count,
                "left_worst_joint": left_worst_joint,
                "left_worst_side": left_worst_side,
                "left_min_margin_rad": left_min_margin,
                "right_worst_joint": right_worst_joint,
                "right_worst_side": right_worst_side,
                "right_min_margin_rad": right_min_margin,
                "left_joint_margin_rad_list": left_margin_values,
                "right_joint_margin_rad_list": right_margin_values,
                "ik_ignore_orientation": int(task_args.ik_ignore_orientation),
                "ik_active_arm": ik_active_arm,
              })

          # Success 
          if env_results[0]["success"].sum().item() == 1:
            result = True
            break

          result_img_3d = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, ::-1].copy()
          if task_args.project_trajs == 1:
            from human_plan.utils.visualization import (
              project_points
            )

            pred_3d = action_dict["pred_3d"]
            proj_2d = project_points(
              pred_3d, cam_intrinsics
            )
            proj_2d = proj_2d.reshape(-1, 2, 2)

            for fi in range(proj_2d.shape[0]-1):
              for j in range(2):
                result_img_3d = cv2.circle(
                  result_img_3d, 
                  (int(proj_2d[fi, j, 0]),int(proj_2d[fi, j, 1])),
                  5, (0, 255, 0), thickness=-1
                )
                if fi < proj_2d.shape[0] - 1:
                  result_img_3d = cv2.line(
                    result_img_3d, 
                  (int(proj_2d[fi, j, 0]),int(proj_2d[fi, j, 1])),
                  (int(proj_2d[fi + 1, j, 0]),int(proj_2d[fi + 1, j, 1])),
                    (0, 255, 0), thickness=2
                  )  

          out.write(result_img_3d)

          if task_args.save_frames:
            cv2.imwrite(
              os.path.join(frames_output_path, f"{i}.jpg"),
              result_img_3d
            )
          count += 1

        with open(task_args.result_saving_path, "a") as f:
          f.write(f"Task: {task_name}, Room Idx: {room_idx}, Table Idx: {table_idx}, Episode Label: {episode_idx[0]}, Trial Label: {trial_idx}, Result: {result} \n")
          subtask_string = ""
          for key in env_results[0].keys():
            if "success" in key:
              subtask_string += f"{key}: {env_results[0][key].sum().item()} "
          subtask_string += "\n"
          f.write(subtask_string)

        if task_args.debug_ik and task_args.debug_ik_csv is not None and len(ik_debug_rows) > 0:
          import csv
          from pathlib import Path
          Path(task_args.debug_ik_csv).mkdir(exist_ok=True, parents=True)
          debug_csv_path = os.path.join(
            task_args.debug_ik_csv,
            f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}_{trial_idx}_ik_debug.csv"
          )
          with open(debug_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ik_debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ik_debug_rows)
          print(f"[IK-DEBUG] saved csv: {debug_csv_path}")
          
        out.release()
        # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
