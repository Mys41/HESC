import os
import torch
from dataclasses import dataclass
import SFTDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.hf_argparser import HfArg
# 加载PEFT模块相关接口
from peft import (
    LoraConfig,
    TaskType,
    AutoPeftModelForCausalLM,
    get_peft_model,
)
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)
from typing import Optional

IGNORE_INDEX = -100


# 用户输入超参数
@dataclass
class Arguments(TrainingArguments):
    # 模型结构
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-chat`",
    )
    # 微调数据集
    dataset: str = HfArg(
        default="",
        help="Setting the names of data file.",
    )
    # 上下文窗口大小
    model_max_length: int = HfArg(
        default=2048,
        help="The maximum sequence length",
    )
    # 只保存模型参数（不保存优化器状态等中间结果）
    save_only_model: bool = HfArg(
        default=True,
        help="When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.",
    )
    # 使用BF16混合精度训练
    bf16: bool = HfArg(
        default=True,
        help="Whether to use bf16 (mixed) precision instead of 32-bit.",
    )
    # LoRA相关超参数
    lora: Optional[bool] = HfArg(default=False, help="whether to train with LoRA.")
    # LoRA注意力维度
    lora_r: Optional[int] = HfArg(default=16, help='Lora attention dimension (the "rank")')
    # alpha参数
    lora_alpha: Optional[int] = HfArg(default=16, help="The alpha parameter for Lora scaling.")
    # dropout
    lora_dropout: Optional[float] = HfArg(default=0.05, help="The dropout probability for Lora layers.")


# 批次化数据，并构建序列到序列损失(交叉熵损失)
@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # 返回输入id和标签，用于计算序列到序列损失
        return dict(
            input_ids=input_ids,
            labels=labels,
        )


def train():
    # 解析命令行参数
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=False,
    )
    # 加载LoRA配置并初始化LoRA模型
    if args.lora:
        # 加载模型，并使用FlashAttention
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2")
        # 配置LoRA微调参数
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        # 用于获取一个已经配置了PEFT方法的模型。这个函数需要两个参数：一个基础模型和一个PEFT配置
        model = get_peft_model(model, peft_config)
    else:
        # 直接加载模型，并使用FlashAttention
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2")

    # 初始化训练器、准备训练数据并开始训练
    kwargs = dict(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=SFTDataset(args, tokenizer),
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    trainer = Trainer(**kwargs)
    trainer.train()
    # 使用LoRA高效微调后，这里保存的模型检查点实际上包含PEFT参数和原始模型参数
    trainer.save_model(args.output_dir + "/checkpoint-final")
    trainer.save_state()

    # 将LoRA参数合并到原始模型中
    if args.lora:
        # 如果启用了Zero3，那么就需要调用unset_hf_deepspeed_config()函数来取消设置Hugging Face的DeepSpeed配置
        if is_deepspeed_zero3_enabled():
            unset_hf_deepspeed_config()
        # 获取args.output_dir目录下的所有子目录，并将它们存储在subdir_list列表中
        subdir_list = os.listdir(args.output_dir)
        for subdir in subdir_list:
            # 检查子目录的名称是否以"checkpoint"开头
            if subdir.startswith("checkpoint"):
                print("Merging model in ", args.output_dir + "/" + subdir)
                # 从模型检查点中加载PEFT模型，获得PEFT参数和原始模型参数
                peft_model = AutoPeftModelForCausalLM.from_pretrained(args.output_dir + "/" + subdir)
                # 将PEFT参数合并到原始模型中，并卸载PEFT模型
                merged_model = peft_model.merge_and_unload()
                # 定义了保存合并后模型的路径
                save_path = args.output_dir + "/" + subdir + "-merged"
                merged_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    train()
