import torch
from datasets import load_dataset
from itertools import chain


# 预训练数据集类
class PTDataset:
    def __init__(self, args, tokenizer):
        self.args = args
        self.block_size = self.args.model_max_length
        self.tokenizer = tokenizer
        self.input_ids = self.process()
        self.input_ids = self.group_texts(self.input_ids)
        self.labels = self.input_ids.copy()

    # 数据集长度
    def __len__(self):
        return len(self.input_ids)

    # 获取第i条数据
    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    # 数据分词
    def encode(self, examples):
        output = self.tokenizer(examples["text"], truncation=True)
        return output

    # 数据批次化处理
    def group_texts(self, examples):
        # 使用itertools.chain函数将样本中的所有元素连接成一个列表
        concatenated_examples = list(chain(*examples))
        total_length = len(concatenated_examples)
        # 将total_length调整为block_size的整数倍。这是为了确保后续的切片操作能够将样本完整地切分成多个长度为block_size的块
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # 将concatenated_examples切分成多个长度为block_size的块，并将这些块堆叠成一个新的张量
        result = [
            torch.stack(concatenated_examples[i:i + self.block_size]) for i
            in range(0, total_length, self.block_size)
        ]
        return result

    # 调用数据集加载、分词、批次化
    def process(self):
        input_ids = []
        list_data_dict = load_dataset('text', data_files=self.args.data_path)['train']
        tokenized_dataset = list_data_dict.map(
            self.encode,
            batched=True,
            remove_columns='text',
        )
        for example in tokenized_dataset:
            if len(example['input_ids']) > 0:
                input_ids.append(torch.tensor(example['input_ids']))
        return input_ids
