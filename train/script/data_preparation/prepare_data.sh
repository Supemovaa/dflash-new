python test.py \
  --model Qwen/Qwen3-8B \
  --temperature 0.7 \
  --max-tokens 4096 \
  --concurrency 64 \
  --input-file-path /home/jianc/cache/datasets/nemotron_train.jsonl \
  --output-file-path /home/jianc/cache/datasets/nemotron_regen_think.jsonl \
  --num-samples 128 \
  --server-address localhost:30000\
  --thinking