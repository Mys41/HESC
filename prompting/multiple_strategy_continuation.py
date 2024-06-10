import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Tuple
import fire
from prompting.llama_prompt import modified_extes_support_strategies
from prompting.llama_prompt import B_SYS, B_INST, E_INST, E_SYS
import torch
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

template3 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and to the point.\
 Do not provide additional info. only respond in one or two short sentences that satisfies {cur_strategy} strategy."""

template4 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy ({strategy_description}) make your response short and to the point.\
 Do not provide additional info. only respond in one paragraph that satisfies {cur_strategy} strategy.\
 answer in this format: assistant: <response>
 You are an expert in emotional psychology, and your best friend has come to you with the following situation: "{situation}".\nNow your friend is a help-seeker, and you play the role of a supporter. Understand the help-seeker's emotion, follow the help-seeker's point of view and intention, express sympathy for the help-seeker's negative situation or approval of the help-seeker’s positive situation. The response should not imply negative emotions toward anyone or anything, such as disgust, resentment, hatred, etc. Consider the potential impact of your response on the help-seeker, and offer encouragement, comfort, support.\nYou can choose one of the following 7 support strategies to use: {strategy_description}\nThe following user message contains a conversation between 'help-seeker' and 'supporter'. Based on the context, you need to empathetically respond to the 'help-seeker'. Your response should be based on the entire context and situation. Let’s think about it step by step:\nStep 1: Understand the dialogue context.\nStep 2: Identify the help-seeker’s emotions and think carefully why.\nStep 3: Select the most appropriate support strategy based on Support strategy description, context, and help-seeker's situation to generate a response.\nStep 4: You’re the 'supporter', think about how to reply to 'help-seeker' in empathy.\nCombine above thoughts to give your response. Your response should start with the name of the selected support strategy, and be enclosed in square brackets followed by the actual response. The response should not include the thought process, only the final response.
 """

template5 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy ({strategy_description}) make your response short and to the point.\
 Do not provide additional info. only respond in ONE SENTENCE in this format: assistant: <one sentence response>"""


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


# 用于存储和处理对话、境况、发言者、响应、提示和注意力等信息
class ESPromptOutput:
    def __init__(self, dialog: List[str], situation: str, speakers: List[str], responses: Dict[str, str],
                 prompts: Dict[str, str] = None):
        self.dialog = dialog
        self.situation = situation
        self.speakers = speakers
        self.responses = responses
        self.prompts = prompts

    # 将类的实例变量转换为一个字典
    def to_dict(self):
        return {
            "dialog": self.dialog,
            "situation": self.situation,
            "speakers": self.speakers,
            "responses": self.responses,
            "prompts": self.prompts,
        }

    # 返回一个字符串，用于表示类的实例
    def __repr__(self):
        respones_str = ""
        for strategy, resp in self.responses.items():
            respones_str += f"{'*' * 300}\n\n strategy: {strategy}\n\n response: {resp}\n\n{'*' * 300}\n\n"
        return "_______________________".join([
            f"Situation: {self.situation}",
            f"Dialog: {self.dialog}",
            f"responses: {respones_str}"
        ])


def convert_to_llama2_chat_format_manually(sys_msg: str, conversations: List[str]) -> str:
    """
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    """
    conv_0 = conversations[0]
    conversations = conversations[1:]

    result = f"<s>{B_INST} {B_SYS}{sys_msg}{E_SYS} {conv_0} {E_INST} "
    i = 0
    while i < len(conversations):
        ai_msg = conversations[i]
        human_msg = conversations[i + 1]
        result += f"{ai_msg} </s><s>{B_INST} {human_msg} {E_INST} "
        i += 2

    return result


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


def convert_to_mistral_chat_partial_conv_format(sys_msg: str, conversations: List[str], tokenizer,
                                                n_turns_as_conv=3, history_first=True, **kwargs) -> str:
    if n_turns_as_conv % 2 != 1:
        raise ValueError("n_turns_as_conv should be odd number")

    conv_messages = []
    for i in range(max(len(conversations) - n_turns_as_conv, 0), len(conversations) - 1, 2):
        conv_messages.append({'role': 'user', 'content': conversations[i].strip()})
        conv_messages.append({'role': 'assistant', 'content': conversations[i + 1].strip()})
    conv_messages.append({'role': 'user', 'content': conversations[-1].strip()})

    if len(conversations) > n_turns_as_conv:
        conversations = conversations[:-n_turns_as_conv]

    conv_history_str = "conversation history:\n"
    for i in range(0, len(conversations) - 1, 2):
        conv_history_str += "user: " + conversations[i].strip() + "\n"
        conv_history_str += "assistant: " + conversations[i + 1].strip() + "\n"

    if history_first:
        sys_msg = f"{conv_history_str}\n{sys_msg.strip()}"
    else:
        sys_msg = f"{sys_msg.strip()}\n{conv_history_str}"

    assert conv_messages[0]['role'] == 'user'
    conv_messages[0]['content'] = f"{sys_msg}\n\n{conv_messages[0]['content']}"

    formatted_prompt = tokenizer.apply_chat_template(conv_messages, tokenize=False)
    return formatted_prompt


# 构造原始提示
def convert_to_llama2_chat_format(sys_msg: str, conversations: List[str], tokenizer, **kwargs) -> str:
    messages = [{'role': 'system', 'content': sys_msg}]
    for i in range(0, len(conversations) - 1, 2):
        messages.append({'role': 'user', 'content': conversations[i]})
        messages.append({'role': 'assistant', 'content': conversations[i + 1]})
    messages.append({'role': 'user', 'content': conversations[-1]})

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_prompt


def convert_to_mistral_chat_format(sys_msg: str, conversations: List[str], tokenizer, **kwargs) -> str:
    messages = []
    for i in range(0, len(conversations) - 1, 2):
        messages.append({'role': 'user', 'content': f"{sys_msg}\n\n{conversations[i]}"})
        messages.append({'role': 'assistant', 'content': conversations[i + 1]})
    messages.append({'role': 'user', 'content': conversations[-1]})

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_prompt


def convert_to_llama2_llm_format(sys_msg: str, conversations: List[str], tokenizer) -> str:
    formatted_prompt = f"{tokenizer.bos_token}{sys_msg}\n\n"

    for i in range(0, len(conversations) - 1, 2):
        formatted_prompt += f"seeker: {conversations[i].strip()}\n"
        formatted_prompt += f"supporter: {conversations[i + 1].strip()}\n"
    formatted_prompt += f"seeker: {conversations[-1].strip()}\n"
    formatted_prompt += f"supporter: "
    return formatted_prompt


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
            # bnb_4bit_quant_type="nf4",  使用NF4数据类型
            # bnb_4bit_use_double_quant=True,  使用嵌套量化技术
            # bnb_4bit_compute_dtype=torch.bfloat16   计算数据类型设置为 torch.bfloat16
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
        quantization_config=quantization_config,  # 设置模型的量化配置
        device_map=device_map,  # 设置模型在多个设备上的分布
        cache_dir=cache_dir,  # 设置模型和配置文件的缓存目录
        torch_dtype=torch_dtype,  # 设置模型的数据类型
    )

    return model, tokenizer


# 获取连续提示
def get_continuation_prompt(conversation, model, tokenizer, model_type='llama', max_new_tokens=512,
                            prompt_constructor=convert_to_llama2_chat_format, sample_prob=0.3,
                            history_first=True, n_turns_as_conv=3):
    dialog = conversation['dialog_history']   # 对话历史
    speakers = conversation['prev_speakers']   # 先前的所有讲话者
    situation = conversation['situation']   # 用户境况

    # 保证第一句为用户语句
    if speakers[0] == 'supporter':
        speakers = speakers[1:]
        dialog = dialog[1:]

    assert speakers[0] == 'seeker'
    assert speakers[-1] == 'seeker'
    responses = {}
    prompts = {}

    # 从修改后的支持策略字典中获得策略名和描述
    for strategy, desc in tqdm(modified_extes_support_strategies.items()):

        # 使用random.random()生成一个随机浮点数，如果这个数大于sample_prob，那么它会跳过当前的循环
        if random.random() > sample_prob:
            continue
        # 用模板4构造系统消息
        sys_msg = template4.format(situation=situation, cur_strategy=strategy, strategy_description=desc)

        # 构造提示
        if model_type == 'llama' or model_type == 'mistral':
            prompt = prompt_constructor(sys_msg, dialog, tokenizer, n_turns_as_conv=n_turns_as_conv,
                                        history_first=history_first)
            # 将构造好的提示输入分词器，获得input_ids
            input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)
            # print("prompt: ", prompt)
            # 提示长度如果大于内存限制，则跳过这个样本
            if len(input_ids[0]) > 1400:
                print(f"PROMPT LENGTH ({len(input_ids[0])}) exceeds the memory limit. skipping this instance")
                continue

        elif model_type == 'mistral':
            pass
        else:
            raise ValueError(f"model_type should be one of ['llama', 'mistral'], but got {model_type}")

        # todo: 当前聚合仅支持贪婪解码
        outputs = model.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens,
                                 return_dict_in_generate=True)

        # 将构造的提示按策略加入提示字典
        prompts[strategy] = prompt

        # 获取模型的输出序列，并去掉输入部分(input_ids[0]的长度)，得到模型的响应response
        response = outputs.sequences[0][len(input_ids[0]):]
        output_txt = tokenizer.decode(response, skip_special_tokens=True).strip()
        # 将output_txt添加到responses字典中
        responses[strategy] = output_txt
        # 打印出提示prompt，以及策略strategy 和对应的响应文本output_txt
        print("\nprompt: ", prompt)
        print(f"\n\nstrategy:\n{strategy}\n\nresponse:\n{output_txt}")

    res = ESPromptOutput(dialog=dialog, situation=situation, speakers=speakers, responses=responses, prompts=prompts)
    return res


def run(data_path='../esconv/conversations.json', min_turn=3, max_turn=10, model_path='/llama2-7b-chat-hf',
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

        with open(os.path.join(output_path, f'{rand_id}.json'), 'w') as f:
            json.dump(generated_conts.to_dict(), f)

        i += 1


if __name__ == '__main__':
    fire.Fire(run)
