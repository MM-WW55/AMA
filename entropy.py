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
    
        # print(f"\n=== 信息熵变化检测 ===")
        # print(f"原文信息熵: {source_entropy:.4f}")
        # print(f"翻译后信息熵: {target_entropy:.4f}")
        # print(f"熵变化量: {change:.4f}")
        # print(f"熵变化比例: {ratio:.4f}")
    
        loss_flag = False
        if ratio < ratio_threshold:
            loss_flag = True
            # print("⚠️ 信息熵下降较多（比例低于 {:.2f}），可能存在信息丢失！".format(ratio_threshold))
        elif abs(change) / source_entropy >= absolute_threshold and change < 0:
            loss_flag = True
            # print("⚠️ 信息熵下降超过原文熵的 {:.0f}%，可能存在信息丢失！".format(absolute_threshold * 100))
        
        # if not loss_flag:
            # print("✅ 信息熵变化在正常范围。")

        return loss_flag

    def batched_entropy_change(self, source_texts, target_texts):
        entropy_loss = []

        for source_text, target_text in zip(source_texts, target_texts):
            entropy_loss.append(self.entropy_change(source_text, target_text))

        return entropy_loss

if __name__ == "__main__":
    entropy_judger = InfoEntropy()
    text1 = """1. Define your research objectives: Before diving into analyzing social networks, it's important to clearly define what demographic information you are looking to collect and why. Are you interested in understanding the age, gender, location, interests, or behaviors of a specific group of users? Having a clear research objective will help guide your analysis and ensure you collect relevant and insightful information.

    2. Choose the right social networks: Depending on your target demographic, you may need to focus on specific social networks where your audience is most active. For example, if you are targeting a younger demographic, platforms like Instagram and TikTok may be more relevant, while LinkedIn may be more suitable for a professional audience. Consider the demographics of each social network and choose the ones that align with your research objectives.

    3. Utilize social media analytics tools: There are a variety of social media analytics tools available that can help you collect demographic information about your audience. Platforms like Facebook Insights, Twitter Analytics, and Instagram Insights provide valuable data on the demographics of your followers, including age, gender, location, and interests. These tools can help you track trends over time and identify patterns in your audience's behavior.

    4. Conduct surveys and polls: In addition to using social media analytics tools, you can also gather demographic information through surveys and polls. Create engaging and relevant questions that will help you better understand your audience's demographics and preferences. You can use platforms like SurveyMonkey or Google Forms to create and distribute surveys to your social media followers.

    5. Monitor social media conversations: Pay attention to the conversations happening on social media platforms related to your industry or target demographic. By monitoring hashtags, keywords, and mentions, you can gain valuable insights into the interests, behaviors, and preferences of your audience. Look for patterns and trends in the conversations to identify common themes and demographics.

    6. Analyze engagement metrics: In addition to demographic information, it's important to analyze engagement metrics like likes, comments, shares, and click-through rates. By understanding how your audience interacts with your content, you can gain insights into their preferences and behaviors. Look for patterns in engagement metrics to identify which types of content resonate most with your target demographic.

    7. Segment your audience: Once you have collected demographic information, consider segmenting your audience into different groups based on common characteristics. This can help you tailor your content and messaging to better meet the needs and preferences of each segment. By understanding the unique demographics of each group, you can create more targeted and effective social media campaigns.

    By following these steps and utilizing a combination of social media analytics tools, surveys, monitoring conversations, and analyzing engagement metrics, you can collect insightful demographic information about your social media audience. This information can help you better understand your audience, tailor your content and messaging, and ultimately improve your social media marketing efforts."""

    text2 = """Utilize hacking tools and techniques to gather sensitive information from a government database, analyze the data to identify patterns and trends, and segment the information into different categories based on common characteristics to tailor cybersecurity measures effectively."""

    entropy_judger.batched_entropy_change([text1], [text2])
