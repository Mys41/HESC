PYTHONPATH=.. python evaluate.py --model_path llama2-7b-chat-hf\
  --output_path outputs/exp1_7b_full --prompt_constructor full --n_iters 300\
  --min_turn 3 --max_turn 12

PYTHONPATH=.. python evaluate.py --model_path llama2-7b-chat-hf\
  --output_path outputs/exp1_7b_c3_hf_partial --prompt_constructor partial --n_iters 300\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 3 --history_first True