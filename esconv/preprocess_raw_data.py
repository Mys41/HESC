import fire
import json
from typing import Dict, Tuple, List
from tqdm import tqdm


VALID_STRATEGIES = [
    'Question',
    'Restatement or Paraphrasing',
    'Reflection of feelings',
    'Self-disclosure',
    'Affirmation and Reassurance',
    'Providing Suggestions',
    'Information',
    'Others'
]


# 分解对话的函数，接收三个参数：conversation(一个字典，包含对话的详细信息)，starting_turn(一个整数，表示从哪个轮次开始分解对话)，
# turn_by_turn(一个布尔值，表示是否按轮次分解对话)。这个函数的返回值是一个字典列表，每个字典代表一个分解后的对话
def decompose_conversation(conversation: Dict, starting_turn: int, turn_by_turn=True) -> List[Dict]:

    # 获得原始ESConv数据集中的对话历史，情绪类型，问题类型和用户境况
    history = conversation['dialog']
    emotion_type = conversation['emotion_type']
    problem_type = conversation['problem_type']
    situation = conversation['situation']

    # 使用列表记录对话历史中每个轮次的内容，发言者和策略
    all_turns = []
    all_speakers = []
    all_strategies = []
    # 最后使用decomposed_examples列表来存储分解后的对话样本，列表中每个元素都是一个字典
    decomposed_examples = []

    # 遍历对话历史列表中的每个轮次
    for turn_obj in history:
        # 获得当前轮次的发言者，内容和注释
        speaker = turn_obj['speaker']
        content = turn_obj['content']
        annotation = turn_obj['annotation']

        # 如果当前轮次是系统话语，则获得使用的支持策略
        if 'strategy' in annotation:
            strategy = annotation['strategy']
        else:
            strategy = ''

        # 保证策略在策略列表中存在
        if speaker == 'supporter' and strategy != '':
            assert strategy in VALID_STRATEGIES, f"strategy {strategy} is not valid"

        # 将当前轮次的内容，发言者和使用的策略记录到列表中
        all_turns.append(content)
        all_speakers.append(speaker)
        all_strategies.append(strategy)

    # 按轮次分解对话。因为原始数据集中可能存在两轮不同的对话是同一发言者的情况，所以需要将这些同一发言者的对话轮次合并
    if turn_by_turn:
        # 初始化一些列表，用于存储合并后的内容、发言者和策略
        concat_turns = [all_turns[0]]
        concat_speakers = [all_speakers[0]]
        concat_strategies = [[all_strategies[0]]]

        # 从第2轮开始遍历对话历史中所有的轮次
        for i in range(1, len(all_turns)):
            prev_speaker = all_speakers[i-1]
            cur_speaker = all_speakers[i]

            # 如果当前发言者和前一个发言者不同，那么就将当前轮次的内容、发言者和策略添加到相应的列表中
            if cur_speaker != prev_speaker:
                concat_turns.append(all_turns[i])
                concat_speakers.append(all_speakers[i])
                concat_strategies.append([all_strategies[i]])
                continue
            # 否则，就只将当前轮次的内容与上一轮次的内容相连，也就是将当前轮次与上一轮次合并
            concat_turns[-1] += ' ' + all_turns[i]

            # 如果当前轮次和上一轮次都是系统轮次，则直接将内容合并，但可能使用了不同的支持策略，所以记录策略的列表需要使用二维列表，
            # 以记录系统在不同轮次中使用的不同策略
            prev_strategy = concat_strategies[-1][-1]
            cur_strategy = all_strategies[i]
            # 如果当前策略和前一个策略不同，那么就将当前策略添加到上一轮次的策略的列表中，完成当前轮次和上一轮次的合并
            if cur_strategy != prev_strategy:
                concat_strategies[-1].append(cur_strategy)
            # 如果当前发言者和前一个发言者相同，则发言者列表无需添加新的元素，不需要额外处理

        all_turns = concat_turns
        all_speakers = concat_speakers
        all_strategies = concat_strategies

    # 函数初始化一些列表，用于存储迄今为止的对话历史、发言者和策略
    conv_so_far = []
    speakers_so_far = []
    strategies_so_far = []

    # 计算支持者(系统)的最大轮次数
    max_turn = len([x for x in all_speakers if x == 'supporter']) - 1

    turn = 0
    # 遍历所有的对话轮次中的内容，发言者和策略
    for i, (turn_content, speaker, strategies) in enumerate(zip(all_turns, all_speakers, all_strategies)):
        # 如果发言者是支持者并且不是对话的开始，那么就增加轮次数
        if speaker == 'supporter' and i > 0:
            turn += 1
        # 如果轮次数达到了支持者的最大轮次数，那么就结束循环
        if turn == max_turn:
            break

        # 达到最大轮次数之前，将对话历史按轮次分解为多个样本
        if speaker == 'supporter' and turn >= starting_turn:
            decomposed_examples.append({
                'emotion_type': emotion_type,
                'problem_type': problem_type,
                'situation': situation,
                'dialog_history': conv_so_far.copy(),
                'prev_speakers': speakers_so_far.copy(),
                'prev_strategies': strategies_so_far.copy(),
                'strategy': strategies,
                'response': turn_content,
                'turn': turn,
            })

        # 记录目前为止的内容，发言者和策略
        conv_so_far.append(turn_content)
        speakers_so_far.append(speaker)
        strategies_so_far.append(strategies)

    return decomposed_examples


def preprocess(
        data_path: str = "ESConv.json",
        output_dir: str = ".",
        starting_turn: int = 1,

):
    with open(data_path, 'r') as f:
        data = json.load(f)

    conversations = []

    # 将原始ESConv数据集中的每份对话历史分解为多个样本，然后加入到conversations列表中
    for conversation in tqdm(data):
        conversations.extend(decompose_conversation(conversation, starting_turn=starting_turn))

    with open(f'{output_dir}/conversations.json', 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')


if __name__ == '__main__':
    # fire是一个Python库，它可以将任何Python组件(函数、类、模块、对象、字典、列表、元组等)转化为命令行接口(CLI)
    # 在程序的末尾调用fire.Fire()，可以将程序的所有内容暴露给命令行。下面只暴露了preprocess函数，让我们可以在命令行
    # 中指定函数的参数，让Bash和Python之间的转换变得更加容易
    fire.Fire(preprocess)
