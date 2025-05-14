import math
from collections import Counter
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class InfoEntropy():

    def __init__(self):
        pass

    def preprocess(self, text):
        """简单预处理：小写、去除标点、分词"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        return words

    def split_sentences(self, text):
        """按编号（1. 2. 3.）切分出每个结构块"""
        lines = re.split(r'\n\s*(?=\d+\.)', text.strip())
        # 移除开头可能没有编号的段落
        lines = [line for line in lines if re.match(r'^\d+\.', line.strip())]
        return lines
    
    def calculate_entropy(self, words):
        """根据词频计算信息熵"""
        if not words:
            return 0.0
        total_words = len(words)
        word_counts = Counter(words)
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            entropy -= p * math.log2(p)
        return entropy    

    def entropy_change(self, source_text, target_text, ratio_threshold=0.75, absolute_threshold=0.20):
        """信息熵变化检测"""
        source_words = self.preprocess(source_text)
        target_words = self.preprocess(target_text)
    
        source_entropy = self.calculate_entropy(source_words)
        target_entropy = self.calculate_entropy(target_words)
        change = target_entropy - source_entropy
        ratio = target_entropy / source_entropy if source_entropy != 0 else 0
    
        loss_flag = False
        if ratio < ratio_threshold:
            loss_flag = True
        elif abs(change) / source_entropy >= absolute_threshold and change < 0:
            loss_flag = True

        return loss_flag

    def batched_entropy_change(self, source_texts, target_texts):
        entropy_loss = []

        for source_text, target_text in zip(source_texts, target_texts):
            entropy_loss.append(self.entropy_change(source_text, target_text))

        return entropy_loss

if __name__ == "__main__":
    entropy_judger = InfoEntropy()
    text1 = ""
    text2 = ""

    entropy_judger.batched_entropy_change([text1], [text2])
