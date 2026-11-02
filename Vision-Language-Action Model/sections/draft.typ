== Conversation

创建一个新的 screen 会话，命名为 pi05_train：

```bash
screen -S pi05_train
```

可以使用 `Ctrl + A` 然后按 `D` 来分离当前的 screen 会话，之后可以使用以下命令重新连接：

```bash
screen -r pi05_train
```

可用 `screen -ls` 查看当前所有的 screen 会话。

== Environment Variables

设置环境变量（TODO）：

```bash
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TORCH_HOME=/root/autodl-tmp/cache/torch
export WANDB_DIR=/root/autodl-tmp/wandb
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export DATASET_ROOT=/root/autodl-tmp/cache/huggingface/hub/datasets--lerobot--aloha_sim_insertion_human/snapshots/cc571a3c661df81b566dbfde3d5c1e85fcdf7884
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

== Baseline

运行一步无效训练（极小学习率，但不能为零，会除以零报错），让 base 模型通过 normalization_mapping 适配 ALOHA 数据集，生成可用于评估的 checkpoint：

```bash
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-30 \
  --policy.scheduler_decay_lr=1e-30 \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --batch_size=1 \
  --num_workers=0 \
  --steps=1 \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq=1 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_base \
  --job_name=pi05_aloha_base
```

环境变量存储 base 模型 checkpoint 的路径：

```bash
export BASE_POLICY=/root/autodl-tmp/outputs/pi05_aloha_base/checkpoints/last/pretrained_model
```

数据集新旧导致训练和 eval 的语言 prompt 不一致，所以不用 `lerobot-eval`，改用 `lerobot_eval_insertion.py` 评估：

```python
from gym_aloha.env import AlohaEnv
from lerobot.scripts.lerobot_eval import main

PROMPT = "Insert the peg into the socket."
AlohaEnv.task_description = property(lambda self: PROMPT)

if __name__ == "__main__":
    print(f"Evaluation prompt: {PROMPT}")
    main()
```

推理一个 episode 测试一下：

```bash
python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$BASE_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_base_insertion_single \
  --job_name=pi05_base_insertion_single
```

后续评估时用固定 seed（例如 1000-1049）运行 50 个 episode：

```bash
python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$BASE_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_base_insertion_50eps \
  --job_name=pi05_base_insertion_50eps
```

还有 ACT（TODO）：

```bash
export ACT_POLICY=/root/autodl-tmp/outputs/act_aloha_insertion/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$ACT_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/act_reference_insertion \
  --job_name=act_reference_insertion
```

== Training

// 少量 step 测试训练命令：

// ```bash
// lerobot-train \
//   --dataset.repo_id=lerobot/aloha_sim_insertion_human \
//   --dataset.root="$DATASET_ROOT" \
//   --policy.type=pi05 \
//   --policy.pretrained_path=lerobot/pi05_base \
//   --policy.device=cuda \
//   --policy.dtype=bfloat16 \
//   --policy.train_expert_only=true \
//   --policy.gradient_checkpointing=true \
//   --policy.compile_model=false \
//   --policy.push_to_hub=false \
//   --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
//   --batch_size=1 \
//   --num_workers=0 \
//   --steps=10 \
//   --log_freq=1 \
//   --save_checkpoint=true \
//   --save_freq=10 \
//   --wandb.enable=false \
//   --output_dir=/root/autodl-tmp/outputs/pi05_aloha_smoke \
//   --job_name=pi05_aloha_smoke
// ```

// 调整 `--batch_size`、`--num_workers`、`--steps`、`--log_freq`、`--save_freq` 等参数进行训练：

// ```bash
// set -o pipefail

// lerobot-train \
//   --dataset.repo_id=lerobot/aloha_sim_insertion_human \
//   --dataset.root="$DATASET_ROOT" \
//   --policy.type=pi05 \
//   --policy.pretrained_path=lerobot/pi05_base \
//   --policy.device=cuda \
//   --policy.dtype=bfloat16 \
//   --policy.train_expert_only=true \
//   --policy.gradient_checkpointing=true \
//   --policy.compile_model=false \
//   --policy.push_to_hub=false \
//   --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
//   --batch_size=16 \
//   --num_workers=4 \
//   --steps=6000 \
//   --dataset.eval_split=0.1 \
//   --eval_steps=500 \
//   --log_freq=10 \
//   --save_checkpoint=true \
//   --save_freq=6000 \
//   --wandb.enable=false \
//   --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert \
//   --job_name=pi05_aloha_insertion_expert \
//   2>&1 | tee /root/autodl-tmp/outputs/pi05_train.log; status=${PIPESTATUS[0]}; echo "exit status: $status" | tee -a /root/autodl-tmp/outputs/pi05_train.log; sync; /usr/bin/shutdown
// ```

// 数据盘较小，`--save_freq` 只能设置大一些；结束/中断就关机，保留日志。

// 测试一下训练后的模型：

// ```bash
// export PI05_IE_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert/checkpoints/last/pretrained_model

// python /root/workspace/lerobot_eval_insertion.py \
//   --policy.path="$PI05_IE_POLICY" \
//   --policy.device=cuda \
//   --env.type=aloha \
//   --env.task=AlohaInsertion-v0 \
//   --env.episode_length=400 \
//   --env.render_mode=rgb_array \
//   --eval.batch_size=1 \
//   --eval.n_episodes=1 \
//   --seed=1000 \
//   --output_dir=/root/autodl-tmp/outputs/eval/pi05_expert_insertion_single \
//   --job_name=pi05_expert_insertion_single
// ```

// 多次运行：

// ```bash
// python /root/workspace/lerobot_eval_insertion.py \
//   --policy.path="$PI05_IE_POLICY" \
//   --policy.device=cuda \
//   --env.type=aloha \
//   --env.task=AlohaInsertion-v0 \
//   --env.episode_length=400 \
//   --env.render_mode=rgb_array \
//   --eval.batch_size=1 \
//   --eval.n_episodes=50 \
//   --seed=1000 \
//   --output_dir=/root/autodl-tmp/outputs/eval/pi05_expert_insertion_50eps \
//   --job_name=pi05_expert_insertion_50eps
// ```

// 一些测试：

// ```bash
// python /root/workspace/lerobot_eval_insertion.py \
//     --policy.path="$PI05_IE_POLICY" \
//     --policy.device=cuda \
//     --policy.n_action_steps=5 \
//     --env.type=aloha \
//     --env.task=AlohaInsertion-v0 \
//     --env.episode_length=400 \
//     --eval.batch_size=1 \
//     --eval.n_episodes=10 \
//     --seed=1000 \
//     --output_dir=/root/autodl-tmp/outputs/eval/pi05_ft_replan5_10eps \
//     --job_name=pi05_ft_replan5_10eps
// ```

// ```bash
// python /root/workspace/lerobot_eval_insertion.py \
//     --policy.path="$PI05_IE_POLICY" \
//     --policy.device=cuda \
//     --policy.n_action_steps=1 \
//     --env.type=aloha \
//     --env.task=AlohaInsertion-v0 \
//     --env.episode_length=400 \
//     --eval.batch_size=1 \
//     --eval.n_episodes=3 \
//     --seed=1000 \
//     --output_dir=/root/autodl-tmp/outputs/eval/pi05_ft_replan1_3eps \
//     --job_name=pi05_ft_replan1_3eps
// ```

复制数据集，补充分位数信息：

```python
from pathlib import Path

from lerobot.datasets import LeRobotDataset, write_stats
from lerobot.scripts.augment_dataset_quantile_stats import (
    compute_quantile_stats_for_dataset,
)

DATASET_ROOT = Path(
    "/root/autodl-tmp/datasets/"
    "aloha_sim_insertion_human_quantiles"
)

print(f"Loading dataset from {DATASET_ROOT}")

dataset = LeRobotDataset(
    repo_id="lerobot/aloha_sim_insertion_human",
    root=DATASET_ROOT,
    video_backend="torchcodec",
)

print("Computing quantile statistics...")
new_stats = compute_quantile_stats_for_dataset(dataset)

print("Writing updated meta/stats.json...")
write_stats(new_stats, DATASET_ROOT)

print("Finished.")
print("Features:", list(new_stats))

for key in ("observation.state", "action"):
    stats = new_stats[key]
    print(f"\n{key}")
    print("q01:", stats["q01"])
    print("q50:", stats["q50"])
    print("q99:", stats["q99"])
```

训练：

```bash
set -o pipefail

export DATASET_QUANTILES_ROOT=/root/autodl-tmp/datasets/aloha_sim_insertion_human_quantiles

lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root="$DATASET_QUANTILES_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --batch_size=16 \
  --num_workers=4 \
  --steps=6000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=6000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert \
  --job_name=pi05_aloha_insertion_expert \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_train_quantiles.log; status=${PIPESTATUS[0]}; echo "exit status: $status" | tee -a /root/autodl-tmp/outputs/pi05_train_quantiles.log; sync; /usr/bin/shutdown
```

测试一下训练后的模型：

```bash
export PI05_IE_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_IE_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_expert_insertion_single \
  --job_name=pi05_expert_insertion_single
```

多次运行：

```bash
python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_IE_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_expert_insertion_50eps \
  --job_name=pi05_expert_insertion_50eps
```

全量训练 smoke（32GB 显存不够）：

```bash
export DATASET_QUANTILES_ROOT=/root/autodl-tmp/datasets/aloha_sim_insertion_human_quantiles

lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root="$DATASET_QUANTILES_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --num_workers=4 \
  --steps=1 \
  --log_freq=10 \
  --save_checkpoint=false \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_full_smoke \
  --job_name=pi05_aloha_insertion_full_smoke
```

== Try LoRA (Including Visual Encoders)

对于 transformer 一般会调整注意力层中的 q 和 v 权重，一般不会同时调这二者和 k（一边是查询一边是索引，调其一就差不多了，同时调容易导致变化更剧烈）。

$pi_(0.5)$ 的 policy 架构大致包含：TODO。考虑微调其中的：TODO。

直接基于 base 的 smoke 和完整训练：

```bash
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root="$DATASET_QUANTILES_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --peft.target_modules='(.*vision_tower.*self_attn\.(q_proj|v_proj)|.*multi_modal_projector\.linear|.*gemma_expert.*self_attn\.(q_proj|v_proj)|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))' \
  --peft.full_training_modules='[]' \
  --batch_size=16 \
  --num_workers=4 \
  --steps=1 \
  --log_freq=10 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_insertion_visual_lora_smoke \
  --job_name=pi05_insertion_visual_lora_smoke
```

```bash
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root="$DATASET_QUANTILES_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --peft.target_modules='(.*vision_tower.*self_attn\.(q_proj|v_proj)|.*multi_modal_projector\.linear|.*gemma_expert.*self_attn\.(q_proj|v_proj)|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))' \
  --peft.full_training_modules='[]' \
  --batch_size=16 \
  --num_workers=4 \
  --steps=6000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=6000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_insertion_visual_lora_full50 \
  --job_name=pi05_insertion_visual_lora_full50
```

基于前面训练 action expert 后的：

```bash
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_human_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert/checkpoints/006000/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --peft.target_modules='(.*vision_tower.*self_attn\.(q_proj|v_proj)|.*multi_modal_projector\.linear|.*gemma_expert.*self_attn\.(q_proj|v_proj)|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))' \
  --peft.full_training_modules='[]' \
  --batch_size=16 \
  --num_workers=4 \
  --steps=1 \
  --log_freq=10 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_insertion_visual_lora_from_action_expert_smoke \
  --job_name=pi05_insertion_visual_lora_from_action_expert_smoke
```

```bash
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_human_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert/checkpoints/006000/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --peft.target_modules='(.*vision_tower.*self_attn\.(q_proj|v_proj)|.*multi_modal_projector\.linear|.*gemma_expert.*self_attn\.(q_proj|v_proj)|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))' \
  --peft.full_training_modules='[]' \
  --batch_size=16 \
  --num_workers=4 \
  --steps=6000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=6000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_insertion_visual_lora_from_action_expert_full50 \
  --job_name=pi05_insertion_visual_lora_from_action_expert_full50
```

多次运行：

```bash
export PI05_VL_POLICY=/root/autodl-tmp/outputs/pi05_insertion_visual_lora_from_action_expert_full50/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_VL_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_insertion_visual_lora_from_action_expert_full50 \
  --job_name=pi05_insertion_visual_lora_from_action_expert_full50
```

== LoRA v2

```bash
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_human_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert/checkpoints/006000/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --peft.method_type=LORA \
  --peft.r=64 \
  --peft.lora_alpha=128 \
  --peft.target_modules='(.*vision_tower.*self_attn\.(q_proj|k_proj|v_proj|out_proj)|.*multi_modal_projector\.linear|.*gemma_expert.*self_attn\.(q_proj|v_proj))' \
  --peft.full_training_modules='["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]' \
  --batch_size=32 \
  --num_workers=4 \
  --steps=2000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_insertion_visual_lora_from_action_expert_v2_2000 \
  --job_name=pi05_insertion_visual_lora_from_action_expert_v2_2000
```

== Longer (v3)

训练：

```bash
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.root="$DATASET_QUANTILES_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --batch_size=32 \
  --num_workers=4 \
  --steps=20000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v3 \
  --job_name=pi05_aloha_insertion_expert_v3
```

多次运行：

```bash
export PI05_V3_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v3/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_V3_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_v3 \
  --job_name=pi05_aloha_insertion_expert_v3
```

// ```bash
// python /root/workspace/lerobot_eval_insertion.py \
//   --policy.path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v3/checkpoints/020000/pretrained_model \
//   --env.type=aloha \
//   --env.task=AlohaInsertion-v0 \
//   --seed=1000 \
//   --eval.n_episodes=50 \
//   --eval.batch_size=1 \
//   --eval.use_async_envs=false \
//   --output_dir=/root/autodl-tmp/outputs/eval/pi05_v3_020000_seed1000
// ```

// 现在 loss 只是 action，如何把例如 “当前已经抓到手上” 这种状态，乃至 “这种状态下怎么做” 也加进去？

继续训练：

```bash
lerobot-train \
  --dataset.repo_id=local/aloha_sim_insertion_comb_quantiles \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_comb_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v3/checkpoints/020000/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --batch_size=32 \
  --num_workers=4 \
  --steps=5000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v4 \
  --job_name=pi05_aloha_insertion_expert_v4 \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v4_train.log
```

多次运行：

```bash
export PI05_V4_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v4/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_V4_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_v4 \
  --job_name=pi05_aloha_insertion_expert_v4
```

继续训练：

```bash
lerobot-train \
  --dataset.repo_id=local/aloha_sim_insertion_comb_quantiles \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_comb_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v4/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --batch_size=32 \
  --num_workers=4 \
  --steps=10000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v5 \
  --job_name=pi05_aloha_insertion_expert_v5 \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v5_train.log
```

多次运行：

```bash
export PI05_V5_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v5/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_V5_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_v5 \
  --job_name=pi05_aloha_insertion_expert_v5
```

继续训练：

```bash
lerobot-train \
  --dataset.repo_id=local/aloha_sim_insertion_comb_quantiles \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_comb_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v5/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --batch_size=32 \
  --num_workers=4 \
  --steps=10000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v6 \
  --job_name=pi05_aloha_insertion_expert_v6 \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v6_train.log
```

多次运行：

```bash
export PI05_V6_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v6/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_V6_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_v6 \
  --job_name=pi05_aloha_insertion_expert_v6
```

继续训练：

```bash
lerobot-train \
  --dataset.repo_id=local/aloha_sim_insertion_comb_quantiles \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_comb_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v6/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --batch_size=32 \
  --num_workers=4 \
  --steps=10000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v7 \
  --job_name=pi05_aloha_insertion_expert_v7 \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v7_train.log
```

多次运行：

```bash
export PI05_V7_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v7/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_V7_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_v7 \
  --job_name=pi05_aloha_insertion_expert_v7
```

继续训练：

```bash
lerobot-train \
  --dataset.repo_id=local/aloha_sim_insertion_comb_quantiles \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_comb_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v7/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --batch_size=32 \
  --num_workers=4 \
  --steps=10000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v8 \
  --job_name=pi05_aloha_insertion_expert_v8 \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v8_train.log
```

多次运行：

```bash
export PI05_V8_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v8/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_V8_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_v8 \
  --job_name=pi05_aloha_insertion_expert_v8
```

继续训练：

```bash
lerobot-train \
  --dataset.repo_id=local/aloha_sim_insertion_comb_quantiles \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_comb_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v8/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --batch_size=64 \
  --num_workers=4 \
  --steps=10000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v9 \
  --job_name=pi05_aloha_insertion_expert_v9 \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v9_train.log
```

多次运行：

```bash
export PI05_V9_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v9/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_V9_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_v9 \
  --job_name=pi05_aloha_insertion_expert_v9
```

== Loss Aware

```bash
lerobot-train-loss-aware \
  --dataset.repo_id=local/aloha_sim_insertion_comb_quantiles \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_comb_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_base/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --loss_aware.enable=true \
  --loss_aware.warmup_steps=3000 \
  --loss_aware.ramp_steps=3000 \
  --loss_aware.alpha=0.5 \
  --loss_aware.w_min=0.5 \
  --loss_aware.w_max=3.0 \
  --batch_size=32 \
  --num_workers=4 \
  --steps=10000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_loss_aware_v1 \
  --job_name=pi05_aloha_insertion_expert_loss_aware_v1 \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_loss_aware_v1_train.log
```

```bash
export PI05_LAV1_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_loss_aware_v1/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_LAV1_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_loss_aware_v1 \
  --job_name=pi05_aloha_insertion_expert_loss_aware_v1
```

……………………………………………………………………………………………………………………

```bash
lerobot-train-loss-aware \
  --dataset.repo_id=local/aloha_sim_insertion_comb_quantiles \
  --dataset.root=/root/autodl-tmp/datasets/aloha_sim_insertion_comb_quantiles \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_loss_aware_v1/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --loss_aware.enable=true \
  --loss_aware.warmup_steps=3000 \
  --loss_aware.ramp_steps=3000 \
  --loss_aware.alpha=0.5 \
  --loss_aware.w_min=0.5 \
  --loss_aware.w_max=3.0 \
  --batch_size=32 \
  --num_workers=4 \
  --steps=20000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=20000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_loss_aware_v2 \
  --job_name=pi05_aloha_insertion_expert_loss_aware_v2 \
  2>&1 | tee /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_loss_aware_v2_train.log
```

```bash
export PI05_LAV2_POLICY=/root/autodl-tmp/outputs/pi05_aloha_insertion_expert_loss_aware_v2/checkpoints/last/pretrained_model

python /root/workspace/lerobot_eval_insertion.py \
  --policy.path="$PI05_LAV2_POLICY" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --env.render_mode=rgb_array \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/root/autodl-tmp/outputs/eval/pi05_aloha_insertion_expert_loss_aware_v2 \
  --job_name=pi05_aloha_insertion_expert_loss_aware_v2
```

== RL

```bash
python /root/workspace/pi05_residual_ppo_insertion.py \
  --base-policy /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v8/checkpoints/last/pretrained_model \
  --output-dir /root/autodl-tmp/outputs/rl/pi05_v8_residual_ppo \
  --device cuda \
  --seed 1000 \
  --total-env-steps 100000 \
  --episode-length 400 \
  --rollout-episodes 8 \
  --residual-scale 0.05 \
  --residual-penalty 0.01 \
  --lr 3e-4 \
  --eval-every-updates 5 \
  --eval-episodes 50 \
  2>&1 | tee /root/autodl-tmp/outputs/rl/pi05_v8_residual_ppo_train.log
```

```bash
python /root/workspace/pi05_residual_ppo_insertion.py \
  --base-policy /root/autodl-tmp/outputs/pi05_aloha_insertion_expert_v8/checkpoints/last/pretrained_model \
  --residual-checkpoint /root/autodl-tmp/outputs/rl/pi05_v8_residual_ppo/checkpoints/best.pt \
  --output-dir /root/autodl-tmp/outputs/eval/pi05_v8_residual_ppo \
  --device cuda \
  --eval-only
```
