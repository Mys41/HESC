# HESC

该存储库包含论文 "Being Human Supporters: Guiding LLMs for Emotional Support Conversation via Chain-of-Thought" 的代码和数据

## 安装依赖
运行我们的模型需要安装的包在`requirement.txt`中给出。注意，由于使用了Flash Attention，您需要将CUDA版本升级至11.6或以上。
您可以使用以下命令运行脚本来安装依赖：

```shell
pip install requirement.txt
```

## 数据清洗
原始ESConv数据集位于`esconv/`目录下。您可以运行`quality_filter`以对数据进行质量过滤，过滤后的文件名为`qfilter.json`
我们使用预训练好的FastText语言分类器，为每个输入文本生成一个语言标签，不符合配置文件中语言类别的文本将被过滤：

```shell
python quality_filter.py
```

您还可以运行`unrepeat`以对数据进行去重，过滤后的文件名为`rfilter.json`
我们进行句子级去重。首先，对文本包含的所有句子（每行对应一个句子）计算𝑛元组， 
对于相邻的句子之间𝑛元组的Jaccard相似度(Jaccard相似度是一种用于比较两个集合相似度的指标，
定义为两个集合的交集大小除以两个集合的并集大小)超过设定阈值的都将会被过滤：

```shell
python unrepeat.py
```

运行`privacy_filter`以对数据进行隐私过滤，过滤后的文件名为`pfilter.json`。使用正则替换的方式将匹配到的身份证号、邮箱和电话号码替换为特定字符串：

```shell
python privacy_filter.py
```

## 预训练
模型预训练的相关文件在`pretrain/`目录下。您可以运行`pretrain.sh`以使用torchrun实现并行训练(单机多卡)
，注意我们仅使用了数据并行和ZeRO优化技术。我们基于Transformers和DeepSpeed进行训练，其中DeepSpeed的配置文件位于`pretrain/DeepSpeed.json`。您可以使用以下命令运行脚本：

```sh
bash pretrain.sh
```

## 指令微调
模型指令微调的相关文件在`finetune/`目录下，微调的数据是`data/finetune.json`。微调的配置和预训练的配置类似，也使用torchrun实现并行训练。 
和预训练一样，我们基于Transformers和DeepSpeed进行微调，使用数据并行、ZeRO阶段3、BF16和激活重计算技术。您可以使用以下命令运行脚本：

```sh
bash finetune.sh
```

## 转换数据格式
对于清洗后的数据`esconv/filter.json`，您可以运行`process_esconv.sh`以将数据转换为我们在实验中使用的格式。
它将在同一文件夹中创建一个json文件 ，名为`conversations.json`。您可以使用以下命令运行脚本：

```sh
bash process_esconv.sh
```

## 实验
对于我们的实验，我们使用具有4位量化的LLaMa v2 chat模型。您可以访问模型
[LLaMa2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)在huggingface上的链接。

所有实验都是使用“transformers”库进行的。我们使用bitsandbytes来4位量化模型。并且通过vLLM库对键值缓存进行分页存储，并结合PagedAttention技术，
显著提升了解码时注意力的计算效率。同时它还支持多种解码策略，引入了批次管理优化技术。

您可以使用以下命令运行论文中的实验：

```sh
cd prompting
bash llama7b.sh
```

然后，您可以使用`prompting/postprocess.py`脚本对生成的响应进行后处理。生成数据的示例
在`data/`目录中可用。每个文件都包含一个不完整的对话和一些使用不同的策略的延续。除了这些信息，我们还提供了用于生成每个延续的确切提示。