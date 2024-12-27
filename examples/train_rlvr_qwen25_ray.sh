set -x

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/aiscuser/code/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_rlvr_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain /home/aiscuser/qwen/Qwen2.5-7B-Instruct/ \
   --reward_pretrain /home/aiscuser/qwen/Qwen2.5-7B-Instruct/ \
   --save_path /home/aiscuser/qwen/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /home/aiscuser/code/OpenRLHF/data/math_prompt_v0 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --save_steps -1 \
   --ckpt_path openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --remote_rm_url 127.0.0.1 \
   --colocate_actor_ref \
   --apply_rlvr

# also supports --advantage_estimator rloo

