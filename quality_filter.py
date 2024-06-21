import sys
sys.path.append("/home/taoran/HESC")
import json
from utils.evaluator import LangIdentifier


class FilterPassageByLangs():
    def __init__(self) -> None:
        self.language_identifier = LangIdentifier(model_path="utils/models/fasttext/lid.176.bin")
        self.reject_threshold = 0.5

    def filter_single_text(self, text: str, accept_lang_list: list) -> bool:
        labels, scores = self.language_identifier.evaluate_single_text(text)
        if scores[0] < self.reject_threshold:
            labels = ["uk"]
        accept_lang_list = [each.lower() for each in accept_lang_list]
        if labels[0] not in accept_lang_list:
            return True
        return False


# 创建一个FilterPassageByLangs对象
filter_passage = FilterPassageByLangs()

# 定义一个接受的语言列表
accept_lang_list = ["en"]

# 读取JSON文件
with open('esconv/ESConv.json', 'r') as f:
    data = json.load(f)

# 对每个样本应用过滤函数
filtered_data = [sample for sample in data if not filter_passage.filter_single_text(sample["situation"],
                                                                                    accept_lang_list)]

# 将过滤后的数据写回JSON文件
with open('esconv/qfilter.json', 'w') as f:
    json.dump(filtered_data, f)
