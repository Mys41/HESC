import sys
import json
sys.path.append("/home/taoran/HESC")
from utils.rules.regex import REGEX_IDCARD, REGEX_EMAIL, REGEX_ZHPHONE
from utils.cleaner.cleaner_base import CleanerBase


class CleanerSubstitutePassageIDCard(CleanerBase):
    def __init__(self):
        super().__init__()

    def clean_single_text(self, text: str, repl_text: str = "**MASKED**IDCARD**") -> str:
        # 使用正则表达式REGEX_IDCARD匹配身份证号，用repl_text代替
        text = self._sub_re(text=text, re_text=REGEX_IDCARD, repl_text=repl_text)

        # 使用正则表达式REGEX_IDCARD匹配邮箱，用repl_text代替
        text = self._sub_re(text=text, re_text=REGEX_EMAIL, repl_text=repl_text)

        # 使用正则表达式REGEX_IDCARD匹配手机号，用repl_text代替
        text = self._sub_re(text=text, re_text=REGEX_ZHPHONE, repl_text=repl_text)

        return text


# 创建一个CleanerDedupLineByNgram对象
cleaner = CleanerSubstitutePassageIDCard()

# 读取JSON文件
with open('esconv/ESConv.json', 'r') as f:
    data = json.load(f)

# 对每个样本应用过滤函数
for sample in data:
    for dialog in sample["dialog"]:
        # 使用CleanerDedupLineByNgram对象来清洗文本
        dialog["content"] = cleaner.clean_single_text(dialog["content"])

# 将过滤后的数据写回JSON文件
with open('esconv/pfilter.json', 'w') as f:
    json.dump(data, f)
