export WANDB_API_KEY='WANDBAPIKEY'
python utils/compute_norm_stats.py --dataset_path='hungchiayu/lerobot_multi_task_1104' --delta_transform
accelerate launch --config_file='config.yaml' training/lerobot/train_lerobot.py