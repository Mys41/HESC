import json
import string
import re
from nltk.util import ngrams


class CleanerDedupLineByNgram():
    def __init__(self):
        self.line_delimiter = list("\n")
        chinese_punctuation = "，。！？：；“”‘’（）《》【】、|—"
        self.gram_delimiter = list(string.punctuation) + list(chinese_punctuation) + ['']

    def clean_single_text(self, text: str, n: int = 5, thre_sim: float = 0.95) -> str:
        lines = [each for each in re.split('|'.join(map(re.escape, self.line_delimiter)), text) if each != '']
        lineinfo, last = list(), {}
        for idx, line in enumerate(lines):
            grams = [each for each in re.split('|'.join(map(re.escape, self.gram_delimiter)), line) if each != '']
            computed_ngrams = list(ngrams(grams, min(len(grams), n)))
            lineinfo.append({
                "lineno": idx, "text": line, "n": min(len(grams), n),
                "ngrams": computed_ngrams, "keep": 0
            })

        for idx, each in enumerate(lineinfo):
            if last == {}:
                each["keep"], last = 1, each
            else:
                ngrams_last, ngrams_cur = set(last["ngrams"]), set(each["ngrams"])
                ngrams_intersection, ngrams_union = len(ngrams_last.intersection(ngrams_cur)), len(
                    ngrams_last.union(ngrams_cur))
                jaccard_sim = ngrams_intersection / ngrams_union if ngrams_union != 0 else 0
                if jaccard_sim < thre_sim:
                    each["keep"], last = 1, each
        text = self.line_delimiter[0].join([each["text"] for each in lineinfo if each["keep"] == 1])
        return text


# 创建一个CleanerDedupLineByNgram对象
cleaner = CleanerDedupLineByNgram()

# 读取JSON文件
with open('esconv/ESConv.json', 'r') as f:
    data = json.load(f)

# 对每个样本应用过滤函数
for sample in data:
    for dialog in sample["dialog"]:
        # 使用CleanerDedupLineByNgram对象来清洗文本
        dialog["content"] = cleaner.clean_single_text(dialog["content"])

# 将过滤后的数据写回JSON文件
with open('esconv/rfilter.json', 'w') as f:
    json.dump(data, f)
