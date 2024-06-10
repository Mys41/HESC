import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List
import fire
from prompting.llama_prompt import strategy_descriptions, modified_extes_support_strategies
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from accelerate import Accelerator
import random

random.seed(11335577)

template1 = """You are a helpful, precise and accurate emotional support expert.\
 The user has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and to the point.\
 Do not provide additional info. only respond in one paragraph that satisfies {cur_strategy} strategy."""

template2 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and natural.\
 Do not provide additional information. only respond in one paragraph that satisfies {cur_strategy} strategy."""

template3 = """You are a helpful and caring friend.\ Your best friend has come to you with the following situation: 
"{situation}". continue the\ conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make 
your response short and to the point.\ Do not provide additional info. only respond in one or two short sentences 
that satisfies {cur_strategy} strategy. """

template4 = """System: You are an expert in emotional psychology, and you can provide effective emotional support. 
Your best friend has come to you with the following situation: "{situation}".\nGuideline: Understand the 
help-seeker's emotion, follow the help-seeker's point of view and intention, express sympathy for the help-seeker's 
negative situation or approval of the help-seeker’s positive situation. The response should not imply negative 
emotions toward anyone or anything, such as disgust, resentment, hatred, etc. Consider the potential impact of your 
response on the help-seeker, and offer encouragement, comfort, support.\nSupport strategy description: {
strategy_description}\nContext: {context}\nECoT: The above is a conversation between 'help-seeker' and 'supporter'. 
Now let's say you're the 'supporter' and you need to make an empathy response to the 'help-seeker' based on the 
context. You need to follow the Guideline. Let’s think about it step by step:\nStep 1: Describe the content of the 
conversation.\nStep 2: Identify the help-seeker’s emotions and think carefully why.\nStep 3: Select the most 
appropriate support strategy based on Support strategy description, context, and help-seeker's situation to generate 
a response.\nStep 4: You’re the 'supporter', think about how to reply to 'help-seeker' in empathy.\nResponse: Combine 
above thoughts give your response. Your response should start with the name of the selected support strategy, 
and be enclosed in square brackets followed by the actual response. Please note that the response should not include 
the thought process, but only the final response. """

template5 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy ({strategy_description}) make your response short and to the point.\
 Do not provide additional info. only respond in ONE SENTENCE in this format: assistant: <one sentence response>"""


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


# 获得模型和tokenizer
def get_model_and_tokenizer(model_name, cache_dir, load_in_4bit=True):
    # access_token = "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"
    # 通过tansformers库直接获取llama模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 使用BitsAndBytes量化模型
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=True
        )
        # 将模型复制到每个设备
        # device_map 一个字典，用于指定模型应该在哪个设备上加载。字典的键是模型的部分名称，值是设备的索引。
        # {"": Accelerator().local_process_index}这行代码的意思是，将模型的所有部分(由空字符串 "" 表示)加载到当前进程的设备上
        # Accelerator().local_process_index返回当前进程的设备索引
        device_map = (
            {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    # 从预训练模型中加载模型，并将上述的量化配置、设备映射、缓存目录和张量数据类型作为参数传入
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    )

    return model, tokenizer


# 构造原始提示
def convert_to_llama2_chat_format(conversations: List[str], tokenizer, **kwargs) -> str:
    labelled_conversations = ""
    for i, dialogue in enumerate(conversations):
        # 去掉句子前后的空格
        dialogue = dialogue.strip()
        # 如果句子不为空
        if dialogue:
            # 如果句子没有以标点符号结束，添加一个句号
            if not dialogue[-1] in ".!?":
                dialogue += "."
            # 根据句子的索引判断发言者
            speaker = "help-seeker: " if i % 2 == 0 else "supporter: "
            # 将发言者标签和句子添加到对话中
            labelled_conversations += speaker + dialogue + " "

    return labelled_conversations


# 将一系列对话转换为适应llama2模型且针对部分对话的提示
# 接受以下参数：sys_msg(系统消息)，conversations(对话列表)，tokenizer(分词器)，n_turns_as_conv(保留的用户消息的轮数，默认为3)，
# history_first(是对话历史优先还是策略指南优先，默认为True)
def convert_to_llama2_chat_partial_conv_format(sys_msg: str, conversations: List[str], tokenizer,
                                               n_turns_as_conv=3, history_first=True, **kwargs) -> str:
    # 检查n_turns_as_conv是否为奇数
    if n_turns_as_conv % 2 != 1:
        raise ValueError("n_turns_as_conv should be odd number")

    # 创建一个空列表conv_messages，并在该列表中添加对话消息。每个消息都是一个字典，包含 ‘role’(角色)和 ‘content’(内容)两个键。
    # conv_messages中只包含最后n_turns_as_conv轮的系统与用户对话
    conv_messages = []
    for i in range(max(len(conversations) - n_turns_as_conv, 0), len(conversations) - 1, 2):
        conv_messages.append({'role': 'user', 'content': conversations[i].strip()})
        conv_messages.append({'role': 'assistant', 'content': conversations[i + 1].strip()})
    conv_messages.append({'role': 'user', 'content': conversations[-1].strip()})

    # 如果conversations的长度大于n_turns_as_conv，那么删除conversations的最后n_turns_as_conv个元素
    if len(conversations) > n_turns_as_conv:
        conversations = conversations[:-n_turns_as_conv]

    # 将conversations作为对话历史，存储在conv_history_str中
    conv_history_str = "conversation history:\n"
    for i in range(0, len(conversations) - 1, 2):
        conv_history_str += "user: " + conversations[i].strip() + "\n"
        conv_history_str += "assistant: " + conversations[i + 1].strip() + "\n"

    # 如果对话历史优先，那么它会将conv_history_str添加到sys_msg的前面。否则，它会将conv_history_str添加到sys_msg的后面
    if history_first:
        sys_msg = f"{conv_history_str}\n{sys_msg.strip()}"
    else:
        sys_msg = f"{sys_msg.strip()}\n{conv_history_str}"

    # 最终的提示格式为：
    # 系统消息(前 n - n_turns_as_conv 轮的对话历史 + 策略指南、用户境况等信息) + 最后 n_turns_as_conv 轮的对话
    messages = [{'role': 'system', 'content': sys_msg}]
    messages.extend(conv_messages)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_prompt


# 获取连续提示
def get_continuation_prompt(conversation, model, tokenizer, model_type='llama', max_new_tokens=512,
                            prompt_constructor=convert_to_llama2_chat_format, sample_prob=0.3,
                            history_first=True, n_turns_as_conv=3):
    dialog = conversation['dialog_history']  # 对话历史
    speakers = conversation['prev_speakers']  # 先前的所有讲话者
    situation = conversation['situation']  # 用户境况
    # response = conversation['response']  # 真实回答

    # 保证第一句为用户语句
    if speakers[0] == 'supporter':
        speakers = speakers[1:]
        dialog = dialog[1:]

    assert speakers[0] == 'seeker'
    assert speakers[-1] == 'seeker'
    # prompts = {}

    # 从修改后的支持策略字典中获得策略名和描述
    for strategy, desc in tqdm(strategy_descriptions.items()):

        # 使用random.random()生成一个随机浮点数，如果这个数大于sample_prob，那么它会跳过当前的循环
        if random.random() > sample_prob:
            continue

        # 构造提示
        prompt = template4.format(situation=situation, cur_strategy=strategy, strategy_description=desc)

        input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)
        # print("prompt: ", prompt)
        # 提示长度如果大于内存限制，则跳过这个样本
        if len(input_ids[0]) > 1400:
            print(f"PROMPT LENGTH ({len(input_ids[0])}) exceeds the memory limit. skipping this instance")
            return

        # 通过模型获取logits
        outputs = model.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens,
                                 output_scores=True, return_dict_in_generate=True)
        # logits = torch.cat(outputs.scores, dim=0)
        # probabilities = F.softmax(logits, dim=-1)
        # predicted_token_ids = torch.argmax(probabilities, dim=-1)
        # predicted_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).strip()

        # labels = tokenizer(response, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)

        # 将构造的提示按策略加入提示字典
        # prompts[strategy] = prompt

        # 获取模型的输出序列，并去掉输入部分(input_ids[0]的长度)，得到模型的响应response
        response = outputs.sequences[0][len(input_ids[0]):]
        output_txt = tokenizer.decode(response, skip_special_tokens=True).strip()
        # 将output_txt添加到responses字典中
        # responses[strategy] = output_txt

        # 打印出提示prompt，以及策略strategy 和对应的响应output_txt
        print("\nprompt: ", prompt)
        print("\nstrategy: ", strategy)
        print("\noutputs: ", output_txt)

        return outputs


def run(data_path='../esconv/conversations.json', min_turn=3, max_turn=10, model_path='nickypro/tinyllama-15M',
        cache_dir=None, output_path='./outputs', load_in_4bit=True, max_new_tokens=512, n_iters=-1,
        prompt_constructor='partial', n_turns_as_conv=None, history_first=None, sample_prob=0.3):
    data = load_jsonl(data_path)
    data = [d for d in data if min_turn <= d['turn'] <= max_turn]

    # 获得模型和分词器
    model, tokenizer = get_model_and_tokenizer(model_path, cache_dir, load_in_4bit)
    tokenizer.padding_side = 'left'
    # 将结束符号作为pad符号
    tokenizer.pad_token = tokenizer.eos_token

    # 根据模型路径(model_path)和提示构造器类型(prompt_constructor)来选择适当的提示构造函数(prompt_constructor_func)
    if 'llama' in model_path:
        if prompt_constructor == 'partial':
            assert n_turns_as_conv is not None
            assert history_first is not None
            prompt_constructor_func = convert_to_llama2_chat_partial_conv_format
        elif prompt_constructor == 'full':
            prompt_constructor_func = convert_to_llama2_chat_format
        else:
            raise ValueError(f"prompt_constructor should be one of ['partial', 'full'], but got {prompt_constructor}")

        # 打印最终使用的提示构造函数
        print(f"using prompt constructor: {prompt_constructor}")

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    i = 0
    if n_iters == -1:
        n_iters = len(data)

    while i < len(data):
        # 如果i大于等于迭代次数，那么跳出循环
        if i >= n_iters:
            break

        # 随机采样数据内的样本
        rand_id = random.randint(0, len(data) - 1)
        if os.path.exists(os.path.join(output_path, f'{rand_id}.json')):
            i += 1
            continue

        # 一段随机对话
        conversation = data[rand_id]
        # 获得对话，境况，发言者，提示和注意力
        generated_conts = get_continuation_prompt(
            conversation, model, tokenizer,
            max_new_tokens=max_new_tokens,
            prompt_constructor=prompt_constructor_func,
            history_first=history_first,
            n_turns_as_conv=n_turns_as_conv,
            sample_prob=sample_prob
        )


if __name__ == '__main__':
    fire.Fire(run)
