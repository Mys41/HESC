torchrun --nproc_per_node=4 --master_port=5999 Trainer.py \  #指定gpu数量和训练时通信的端口号

  --model_name_or_path ../prompting/llama2-7b-chat-hf/ \

  --data_path ../data/finetune.json \

  --bf16 True \  # 使用BF16精度进行训练

  --output_dir output/llama-7b \

  --num_train_epochs 3 \  # 指定训练轮数

  --per_device_train_batch_size 8 \  # 指定单张GPU上的批次大小

  --gradient_accumulation_steps 8 \  # 指定梯度累计步数

  --evaluation_strategy "no" \  # 在训练过程中不进行模型评估

  --save_strategy "steps" \  # 每训练一定步数后就保存一次模型

  --save_steps 200 \  # 模型保存的步数

  --save_total_limit 200 \  # 在训练过程中，最多只会保存200个模型检查点

  --learning_rate 1e-5 \  # 微调的最大学习率

  --weight_decay 0. \  # 权重衰减的值被设置为0，这意味着在训练过程中，不使用权重衰减。也就是说，模型的训练不会受到L2正则化的影响

  --warmup_ratio 0.03 \  # 于指定学习率预热的数据比例

  --lr_scheduler_type "cosine" \  # 指定学习率衰减策略

  --logging_steps 1 \

  --deepspeed DeepSpeed.json \  # 指定使用DeepSpeed的参数文件

  --tf32 True \

  --gradient_checkpointing True  # 是否使用激活重计算技术