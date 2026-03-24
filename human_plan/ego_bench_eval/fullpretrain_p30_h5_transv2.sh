TASK=$1
ROOM_IDX=$2
TABLE_IDX=$3
SMOOTH_WEIGHT=$4
NUM_EPISODES=$5
NUM_TRIALS=$6
SAVING_PATH=$7
SAVE_FRAMES=$8
PROJ_TRAJS=$9
HAND_SMOOTH_WEIGHT=${10}
video_saving_path=${11}
additional_label=${12}
# Optional debug args for saving model input observations.
# Keep old behavior if these are not provided.
SAVE_INPUT_OBS=${13}
INPUT_OBS_STRIDE=${14}
INPUT_OBS_MAX=${15}
INPUT_OBS_DIR=${16}
use_per_step_instruction=${17}
DEBUG_IK=${18}
DEBUG_IK_STRIDE=${19}
DEBUG_IK_CSV=${20}
IK_IGNORE_ORIENTATION=${21}

DEBUG_IK_ARG=""
if [ -n "$DEBUG_IK" ]; then
  DEBUG_IK_ARG="--debug_ik $DEBUG_IK"
fi

DEBUG_IK_STRIDE_ARG=""
if [ -n "$DEBUG_IK_STRIDE" ]; then
  DEBUG_IK_STRIDE_ARG="--debug_ik_stride $DEBUG_IK_STRIDE"
fi

DEBUG_IK_CSV_ARG=""
if [ -n "$DEBUG_IK_CSV" ]; then
  DEBUG_IK_CSV_ARG="--debug_ik_csv $DEBUG_IK_CSV"
fi

IK_IGNORE_ORIENTATION_ARG=""
if [ -n "$IK_IGNORE_ORIENTATION" ]; then
  IK_IGNORE_ORIENTATION_ARG="--ik_ignore_orientation $IK_IGNORE_ORIENTATION"
fi

#source /home/rchal97/code/clean_egovla/isaacsim/setup_conda_env.sh

LOG_ROOT=logs

RUN_NAME=temp
OUTPUT_DIR=$LOG_ROOT/$RUN_NAME

bs=16
n_node=1

exp_id=ego_vla_checkpoint
checkpoint_xxx=$(find checkpoints/$exp_id -type d -name "ckpt-*" -print -quit)
echo $checkpoint_xxx

# deepspeed human_plan/train/train_vla_finetune_llava.py \

#1.改用isaaclab的启动脚本进行启动
#/home/ubuntu/Desktop/IsaacLab/isaaclab.sh -p -m debugpy --listen 5678 --wait-for-client human_plan/ego_bench_eval/ik_agent_30hz.py \
/home/ubuntu/Desktop/IsaacLab/isaaclab.sh -p human_plan/ego_bench_eval/ik_agent_30hz.py \
    --model_name_or_path $checkpoint_xxx \
    --version qwen2 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --data_mixture otv_sim_fixed_set_aug_AUG_SHIFT_30Hz_train \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --group_by_modality_length False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 100 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --gradient_accumulation_steps 1 \
    --eval_data_mixture otv_sim_fixed_set_aug_AUG_SHIFT_30Hz_train_sub100 \
    --evaluation_strategy "steps" \
    --eval_steps 250 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --future_index 1 \
    --predict_future_step 30 \
    --max_action 1 \
    --min_action 0 \
    --add_his_obs_step 5 \
    --add_his_imgs True \
    --add_his_img_skip 6 \
    --num_action_bins 256 \
    --action_tokenizer uniform \
    --invalid_token_weight 0.1 \
    --mask_input True \
    --add_current_language_description False \
    --traj_decoder_type transformer_split_action_v2 \
    --raw_action_label True \
    --traj_action_output_dim 48 \
    --input_placeholder_diff_index True \
    --ee_loss_coeff 20.0 \
    --hand_loss_coeff 5.0 \
    --hand_loss_dim 6 \
    --ee_2d_loss_coeff 0.0 \
    --ee_rot_loss_coeff 5.0 \
    --hand_kp_loss_coeff 0.0 \
    --next_token_loss_coeff 0.0 \
    --traj_action_output_ee_2d_dim 0 \
    --traj_action_output_ee_dim 6 \
    --traj_action_output_hand_dim 30  \
    --traj_action_output_ee_rot_dim 12 \
    --ee_rot_representation rot6d \
    --correct_transformation True \
    --include_2d_label True \
    --include_rot_label True \
    --use_short_language_label True \
    --no_norm_ee_label True \
    --lazy_preprocess True \
    --tf32 True \
    --merge_hand True \
    --use_mano True \
    --sep_proprio True \
    --sep_query_token True \
    --loss_use_l1 True \
    --task $TASK \
    --room_idx $ROOM_IDX \
    --table_idx $TABLE_IDX \
    --smooth_weight $SMOOTH_WEIGHT \
    --num_episodes $NUM_EPISODES \
    --num_trials $NUM_TRIALS \
    --result_saving_path $SAVING_PATH \
    --save_frames $SAVE_FRAMES \
    --project_trajs $PROJ_TRAJS \
    --hand_smooth_weight $HAND_SMOOTH_WEIGHT \
    --video_saving_path $video_saving_path \
    --additional_label $additional_label \
    --save_input_obs $SAVE_INPUT_OBS \
    --input_obs_stride $INPUT_OBS_STRIDE \
    --input_obs_max $INPUT_OBS_MAX \
    --input_obs_dir $INPUT_OBS_DIR \
    $DEBUG_IK_ARG \
    $DEBUG_IK_STRIDE_ARG \
    $DEBUG_IK_CSV_ARG \
    $IK_IGNORE_ORIENTATION_ARG
