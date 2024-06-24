PYTHONPATH=.. python multiple_strategy_continuation.py --model_path llama2-7b-chat-hf\
  --output_path outputs/exp1_7b_full --get_attentions True --prompt_constructor full --n_iters 300\
  --min_turn 3 --max_turn 12
