"""
文本预处理模块
包含规范化和句子切分功能
"""

import re
import unicodedata
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ProcessedSentence:
    """处理后的句子"""
    text: str
    original_start: int      # 在原文中的起始位置
    original_end: int        # 在原文中的结束位置
    sentence_id: int         # 句子ID


class TextPreprocessor:
    """文本预处理器"""
    
    # 全角字符到半角字符的映射
    FULLWIDTH_TO_HALFWIDTH = {
        '　': ' ',  # 全角空格
        '！': '!', '＂': '"', '＃': '#', '＄': '$', '％': '%',
        '＆': '&', '＇': "'", '（': '(', '）': ')', '＊': '*',
        '＋': '+', '，': ',', '－': '-', '．': '.', '／': '/',
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        '：': ':', '；': ';', '＜': '<', '＝': '=', '＞': '>',
        '？': '?', '＠': '@',
        'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
        'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
        'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
        'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
        'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y',
        'Ｚ': 'Z',
        'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
        'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
        'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
        'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
        'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y',
        'ｚ': 'z',
    }
    
    # 保留的中文标点（不转换）
    CHINESE_PUNCTUATION = set('。，、；：？！""''【】《》（）—…')
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        normalize_fullwidth: bool = True,
        remove_invisible: bool = True,
        normalize_whitespace: bool = True,
        keep_chinese_punctuation: bool = True,
    ):
        """
        Args:
            normalize_unicode: 是否进行Unicode规范化 (NFKC)
            normalize_fullwidth: 是否将全角字符转为半角
            remove_invisible: 是否移除不可见字符
            normalize_whitespace: 是否规范化空白字符
            keep_chinese_punctuation: 是否保留中文标点
        """
        self.normalize_unicode = normalize_unicode
        self.normalize_fullwidth = normalize_fullwidth
        self.remove_invisible = remove_invisible
        self.normalize_whitespace = normalize_whitespace
        self.keep_chinese_punctuation = keep_chinese_punctuation
        
        # 构建转换表
        self._build_conversion_table()
    
    def _build_conversion_table(self):
        """构建字符转换表"""
        self.conversion_table = {}
        if self.normalize_fullwidth:
            for full, half in self.FULLWIDTH_TO_HALFWIDTH.items():
                # 如果是中文标点且需要保留，则不转换
                if self.keep_chinese_punctuation and full in self.CHINESE_PUNCTUATION:
                    continue
                self.conversion_table[ord(full)] = half
    
    def preprocess(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text:
            return text
        
        # 1. Unicode规范化
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # 2. 移除不可见字符
        if self.remove_invisible:
            text = self._remove_invisible_chars(text)
        
        # 3. 全角转半角（保留中文标点）
        if self.normalize_fullwidth and self.conversion_table:
            text = text.translate(self.conversion_table)
        
        # 4. 规范化空白字符
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        return text
    
    def _remove_invisible_chars(self, text: str) -> str:
        """移除不可见字符"""
        result = []
        for char in text:
            # 保留常见的空白字符
            if char in ' \t\n\r':
                result.append(char)
                continue
            
            # 移除控制字符和其他不可见字符
            category = unicodedata.category(char)
            if category.startswith('C'):  # Control characters
                continue
            
            result.append(char)
        
        return ''.join(result)
    
    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        # 将多个连续空格替换为单个空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 统一换行符
        text = re.sub(r'\r\n|\r', '\n', text)
        # 移除行首行尾空格
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text
    
    def preprocess_pair(self, source: str, target: str) -> Tuple[str, str]:
        """
        同时预处理源文本和目标文本
        
        Args:
            source: 原文（错误句）
            target: 目标（正确句）
            
        Returns:
            预处理后的 (source, target)
        """
        return self.preprocess(source), self.preprocess(target)


class SentenceSplitter:
    """句子切分器"""
    
    # 句子结束标点
    SENTENCE_END_PUNCTUATION = set('。！？；')
    
    # 可能的句子边界（弱分隔符）
    WEAK_BOUNDARIES = set('，、：')
    
    def __init__(
        self,
        max_length: int = 128,
        min_length: int = 5,
        split_by_newline: bool = True,
        split_by_punctuation: bool = True,
        respect_quotes: bool = True,
    ):
        """
        Args:
            max_length: 最大句子长度
            min_length: 最小句子长度（避免过短片段）
            split_by_newline: 是否按换行切分
            split_by_punctuation: 是否按标点切分
            respect_quotes: 是否尊重引号边界（不在引号中间切分）
        """
        self.max_length = max_length
        self.min_length = min_length
        self.split_by_newline = split_by_newline
        self.split_by_punctuation = split_by_punctuation
        self.respect_quotes = respect_quotes
    
    def split(self, text: str) -> List[ProcessedSentence]:
        """
        切分文本为句子列表
        
        Args:
            text: 输入文本
            
        Returns:
            ProcessedSentence 列表
        """
        if not text:
            return []
        
        # 第一步：按换行切分
        if self.split_by_newline:
            segments = self._split_by_newline(text)
        else:
            segments = [(text, 0)]
        
        # 第二步：按标点切分
        sentences = []
        for segment, offset in segments:
            if self.split_by_punctuation:
                sub_sentences = self._split_by_punctuation(segment, offset)
            else:
                sub_sentences = [(segment, offset)]
            sentences.extend(sub_sentences)
        
        # 第三步：处理过长句子
        final_sentences = []
        for sent, offset in sentences:
            if len(sent) > self.max_length:
                chunks = self._split_long_sentence(sent, offset)
                final_sentences.extend(chunks)
            elif len(sent) >= self.min_length:
                final_sentences.append((sent, offset))
            elif final_sentences and len(sent) > 0:
                # 过短的句子合并到前一个句子
                prev_sent, prev_offset = final_sentences[-1]
                final_sentences[-1] = (prev_sent + sent, prev_offset)
        
        # 构建结果
        result = []
        for i, (sent, offset) in enumerate(final_sentences):
            if sent.strip():  # 过滤空句子
                result.append(ProcessedSentence(
                    text=sent,
                    original_start=offset,
                    original_end=offset + len(sent),
                    sentence_id=i
                ))
        
        return result
    
    def _split_by_newline(self, text: str) -> List[Tuple[str, int]]:
        """按换行符切分"""
        segments = []
        current_pos = 0
        
        for line in text.split('\n'):
            if line:  # 跳过空行
                segments.append((line, current_pos))
            current_pos += len(line) + 1  # +1 for newline
        
        return segments
    
    def _split_by_punctuation(self, text: str, offset: int) -> List[Tuple[str, int]]:
        """按标点符号切分"""
        sentences = []
        current_sent = []
        current_start = offset
        in_quotes = False
        quote_char = None
        
        for i, char in enumerate(text):
            current_sent.append(char)
            
            # 处理引号
            if self.respect_quotes:
                if char in '"「『' and not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char in '"」』' and in_quotes:
                    in_quotes = False
                    quote_char = None
            
            # 在引号内不切分
            if in_quotes:
                continue
            
            # 遇到句子结束标点
            if char in self.SENTENCE_END_PUNCTUATION:
                sent_text = ''.join(current_sent)
                sentences.append((sent_text, current_start))
                current_sent = []
                current_start = offset + i + 1
        
        # 处理剩余文本
        if current_sent:
            sent_text = ''.join(current_sent)
            sentences.append((sent_text, current_start))
        
        return sentences
    
    def _split_long_sentence(self, text: str, offset: int) -> List[Tuple[str, int]]:
        """切分过长的句子"""
        chunks = []
        current_chunk = []
        current_start = offset
        
        for i, char in enumerate(text):
            current_chunk.append(char)
            
            # 达到最大长度
            if len(current_chunk) >= self.max_length:
                # 尝试在弱分隔符处切分
                split_pos = self._find_split_position(current_chunk)
                
                if split_pos > 0:
                    chunk_text = ''.join(current_chunk[:split_pos + 1])
                    chunks.append((chunk_text, current_start))
                    
                    # 保留剩余部分
                    remaining = current_chunk[split_pos + 1:]
                    current_start = offset + i - len(remaining) + 1
                    current_chunk = remaining
                else:
                    # 没找到合适的切分点，强制切分
                    chunk_text = ''.join(current_chunk)
                    chunks.append((chunk_text, current_start))
                    current_chunk = []
                    current_start = offset + i + 1
        
        # 处理剩余文本
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append((chunk_text, current_start))
        
        return chunks
    
    def _find_split_position(self, chars: List[str]) -> int:
        """在字符列表中找到合适的切分位置"""
        # 从后往前找弱分隔符
        for i in range(len(chars) - 1, max(0, len(chars) - self.max_length // 2), -1):
            if chars[i] in self.WEAK_BOUNDARIES:
                return i
        return -1
    
    def split_parallel(
        self, 
        source: str, 
        target: str
    ) -> List[Tuple[ProcessedSentence, ProcessedSentence]]:
        """
        并行切分源文本和目标文本
        注意：对于纠错任务，源和目标通常长度相近，使用相同的切分点
        
        Args:
            source: 源文本（错误句）
            target: 目标文本（正确句）
            
        Returns:
            (source_sentence, target_sentence) 对的列表
        """
        # 简单策略：以源文本为准进行切分
        source_sents = self.split(source)
        
        # 如果源和目标长度差异不大，直接对齐切分
        if abs(len(source) - len(target)) <= len(source) * 0.2:
            target_sents = self.split(target)
            
            # 如果切分数量相同，直接配对
            if len(source_sents) == len(target_sents):
                return list(zip(source_sents, target_sents))
        
        # 否则，不切分，返回整句
        source_sent = ProcessedSentence(
            text=source,
            original_start=0,
            original_end=len(source),
            sentence_id=0
        )
        target_sent = ProcessedSentence(
            text=target,
            original_start=0,
            original_end=len(target),
            sentence_id=0
        )
        return [(source_sent, target_sent)]


def preprocess_text_pair(
    source: str,
    target: str,
    preprocessor: Optional[TextPreprocessor] = None,
    splitter: Optional[SentenceSplitter] = None,
) -> List[Tuple[str, str]]:
    """
    预处理文本对的便捷函数
    
    Args:
        source: 源文本
        target: 目标文本
        preprocessor: 预处理器（可选）
        splitter: 句子切分器（可选）
        
    Returns:
        (source, target) 对的列表
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    # 预处理
    source, target = preprocessor.preprocess_pair(source, target)
    
    # 切分
    if splitter is not None:
        pairs = splitter.split_parallel(source, target)
        return [(s.text, t.text) for s, t in pairs]
    else:
        return [(source, target)]
