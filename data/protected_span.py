"""
保护约束模块
识别并保护不应被修改的文本片段（术语、格式等）
"""

import re
from typing import List, Tuple, Set, Optional, Dict, Pattern
from dataclasses import dataclass
from enum import Enum


class ProtectedType(Enum):
    """保护类型枚举"""
    DOC_NUMBER = "doc_number"           # 文号 如：国发〔2024〕1号
    DATE = "date"                       # 日期 如：2024年12月13日
    AMOUNT = "amount"                   # 金额 如：100万元、1,234.56元
    CLAUSE_NUMBER = "clause_number"     # 条款编号 如：第一条、第3款
    ORGANIZATION = "organization"       # 机构名称
    LAW_NAME = "law_name"              # 法规名称
    FIXED_PHRASE = "fixed_phrase"       # 固定格式短语
    CUSTOM = "custom"                   # 自定义保护


@dataclass
class ProtectedSpan:
    """受保护的文本片段"""
    start: int          # 起始位置（包含）
    end: int            # 结束位置（不包含）
    text: str           # 片段文本
    span_type: ProtectedType  # 保护类型
    confidence: float = 1.0   # 置信度


class ProtectedSpanDetector:
    """
    保护片段检测器
    
    检测并标记不应被修改的文本片段：
    - 文号：国发〔2024〕1号、京政发〔2023〕第1号
    - 日期：2024年12月13日、2024/12/13、2024-12-13
    - 金额：100万元、1,234.56元、￥100
    - 条款编号：第一条、第3款、第（一）项
    - 机构名称：国务院、xxx局、xxx委员会
    - 法规名称：《xxx法》、《xxx条例》
    - 固定格式：根据...、现将...通知如下
    """
    
    # 预定义正则表达式
    PATTERNS: Dict[ProtectedType, List[Pattern]] = {
        ProtectedType.DOC_NUMBER: [
            # 国发〔2024〕1号、京政发〔2023〕第1号
            re.compile(r'[\u4e00-\u9fa5]+[发办函议通批指令][\[〔\(（]\d{4}[\]〕\)）]第?[\d一二三四五六七八九十百]+号'),
            # 简化格式：x字〔2024〕第x号
            re.compile(r'[\u4e00-\u9fa5]+字[\[〔\(（]\d{4}[\]〕\)）]第?[\d一二三四五六七八九十百]+号'),
        ],
        ProtectedType.DATE: [
            # 2024年12月13日
            re.compile(r'\d{4}年\d{1,2}月\d{1,2}日'),
            # 2024年12月
            re.compile(r'\d{4}年\d{1,2}月'),
            # 2024/12/13 或 2024-12-13
            re.compile(r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}'),
            # 12月13日
            re.compile(r'\d{1,2}月\d{1,2}日'),
        ],
        ProtectedType.AMOUNT: [
            # 1,234.56元、100万元、5000亿元
            re.compile(r'[\d,]+\.?\d*[万亿]?元'),
            # ￥100、¥1,234.56
            re.compile(r'[￥¥][\d,]+\.?\d*'),
            # 百分比
            re.compile(r'\d+\.?\d*%'),
            # 100万、5亿
            re.compile(r'\d+\.?\d*[万亿千百十]'),
        ],
        ProtectedType.CLAUSE_NUMBER: [
            # 第一条、第3款、第（一）项
            re.compile(r'第[一二三四五六七八九十百千\d]+[条款项章节]'),
            # 第（一）款
            re.compile(r'第[（\(][一二三四五六七八九十百千\d]+[）\)][条款项章节]'),
            # (一)、（1）
            re.compile(r'[（\(][一二三四五六七八九十\d]+[）\)]'),
            # 1.、2.、3. （标题编号）
            re.compile(r'^\d+[\.\、]'),
        ],
        ProtectedType.LAW_NAME: [
            # 《xxx法》、《xxx条例》、《xxx办法》
            re.compile(r'《[\u4e00-\u9fa5\s]+(?:法|条例|规定|办法|细则|准则|规范|标准|意见|通知|决定)》'),
            # 简称形式：xxx法
            re.compile(r'[\u4e00-\u9fa5]{2,}(?:法|条例|规定|办法)(?![》\u4e00-\u9fa5])'),
        ],
        ProtectedType.FIXED_PHRASE: [
            # 公文固定格式
            re.compile(r'根据[\u4e00-\u9fa5《》]+[，,]'),
            re.compile(r'依据[\u4e00-\u9fa5《》]+[，,]'),
            re.compile(r'按照[\u4e00-\u9fa5《》]+[，,]'),
            re.compile(r'现将[\u4e00-\u9fa5]+(?:通知|印发|转发|公布)如下'),
            re.compile(r'特此(?:通知|公告|说明|函复)'),
            re.compile(r'此复[。]?'),
        ],
    }
    
    # 常见机构名称后缀
    ORG_SUFFIXES = [
        '部', '委', '局', '厅', '处', '科', '室', '中心',
        '委员会', '办公室', '研究院', '研究所', '大学', '学院',
        '公司', '集团', '银行', '协会', '学会', '基金会',
        '人民政府', '国务院', '全国人大', '全国政协',
    ]
    
    def __init__(
        self,
        enable_doc_number: bool = True,
        enable_date: bool = True,
        enable_amount: bool = True,
        enable_clause_number: bool = True,
        enable_organization: bool = True,
        enable_law_name: bool = True,
        enable_fixed_phrase: bool = True,
        custom_patterns: Optional[List[Tuple[str, str]]] = None,
        custom_protected_words: Optional[Set[str]] = None,
    ):
        """
        初始化保护片段检测器
        
        Args:
            enable_doc_number: 是否保护文号
            enable_date: 是否保护日期
            enable_amount: 是否保护金额
            enable_clause_number: 是否保护条款编号
            enable_organization: 是否保护机构名称
            enable_law_name: 是否保护法规名称
            enable_fixed_phrase: 是否保护固定格式
            custom_patterns: 自定义正则模式 [(pattern_str, type_name), ...]
            custom_protected_words: 自定义保护词汇集合
        """
        self.enabled_types: Set[ProtectedType] = set()
        
        if enable_doc_number:
            self.enabled_types.add(ProtectedType.DOC_NUMBER)
        if enable_date:
            self.enabled_types.add(ProtectedType.DATE)
        if enable_amount:
            self.enabled_types.add(ProtectedType.AMOUNT)
        if enable_clause_number:
            self.enabled_types.add(ProtectedType.CLAUSE_NUMBER)
        if enable_organization:
            self.enabled_types.add(ProtectedType.ORGANIZATION)
        if enable_law_name:
            self.enabled_types.add(ProtectedType.LAW_NAME)
        if enable_fixed_phrase:
            self.enabled_types.add(ProtectedType.FIXED_PHRASE)
        
        # 自定义正则模式
        self.custom_patterns: List[Tuple[Pattern, str]] = []
        if custom_patterns:
            for pattern_str, type_name in custom_patterns:
                try:
                    compiled = re.compile(pattern_str)
                    self.custom_patterns.append((compiled, type_name))
                except re.error as e:
                    print(f"Warning: Invalid regex pattern '{pattern_str}': {e}")
        
        # 自定义保护词汇
        self.custom_protected_words: Set[str] = custom_protected_words or set()
        
        # 构建机构名称正则
        self._build_org_pattern()
    
    def _build_org_pattern(self):
        """构建机构名称正则表达式"""
        # 机构名称模式：xxx + 后缀
        suffix_pattern = '|'.join(re.escape(s) for s in self.ORG_SUFFIXES)
        self.org_pattern = re.compile(
            rf'[\u4e00-\u9fa5]{{2,}}(?:{suffix_pattern})'
        )
    
    def detect(self, text: str) -> List[ProtectedSpan]:
        """
        检测文本中的受保护片段
        
        Args:
            text: 输入文本
            
        Returns:
            受保护片段列表（按起始位置排序）
        """
        spans = []
        
        # 检测预定义模式
        for ptype, patterns in self.PATTERNS.items():
            if ptype not in self.enabled_types:
                continue
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    spans.append(ProtectedSpan(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        span_type=ptype,
                        confidence=1.0
                    ))
        
        # 检测机构名称
        if ProtectedType.ORGANIZATION in self.enabled_types:
            for match in self.org_pattern.finditer(text):
                # 过滤掉过短的匹配
                if len(match.group()) >= 3:
                    spans.append(ProtectedSpan(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        span_type=ProtectedType.ORGANIZATION,
                        confidence=0.8
                    ))
        
        # 检测自定义正则模式
        for pattern, type_name in self.custom_patterns:
            for match in pattern.finditer(text):
                spans.append(ProtectedSpan(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    span_type=ProtectedType.CUSTOM,
                    confidence=1.0
                ))
        
        # 检测自定义保护词汇
        for word in self.custom_protected_words:
            start = 0
            while True:
                pos = text.find(word, start)
                if pos == -1:
                    break
                spans.append(ProtectedSpan(
                    start=pos,
                    end=pos + len(word),
                    text=word,
                    span_type=ProtectedType.CUSTOM,
                    confidence=1.0
                ))
                start = pos + 1
        
        # 合并重叠的片段并排序
        spans = self._merge_overlapping_spans(spans)
        
        return spans
    
    def _merge_overlapping_spans(
        self,
        spans: List[ProtectedSpan]
    ) -> List[ProtectedSpan]:
        """合并重叠的保护片段"""
        if not spans:
            return []
        
        # 按起始位置排序
        spans.sort(key=lambda x: (x.start, -x.end))
        
        merged = [spans[0]]
        for span in spans[1:]:
            last = merged[-1]
            
            # 检查是否重叠
            if span.start < last.end:
                # 重叠：取较大的范围
                if span.end > last.end:
                    # 扩展当前片段
                    merged[-1] = ProtectedSpan(
                        start=last.start,
                        end=span.end,
                        text=last.text[:span.end - last.start],  # 近似
                        span_type=last.span_type,
                        confidence=max(last.confidence, span.confidence)
                    )
            else:
                merged.append(span)
        
        return merged
    
    def get_protected_positions(self, text: str) -> Set[int]:
        """
        获取所有受保护的位置索引
        
        Args:
            text: 输入文本
            
        Returns:
            受保护位置索引的集合
        """
        spans = self.detect(text)
        positions = set()
        for span in spans:
            for i in range(span.start, span.end):
                positions.add(i)
        return positions
    
    def is_position_protected(
        self,
        text: str,
        position: int,
        spans: Optional[List[ProtectedSpan]] = None
    ) -> bool:
        """
        检查特定位置是否受保护
        
        Args:
            text: 输入文本
            position: 位置索引
            spans: 预先检测的保护片段（可选，避免重复检测）
            
        Returns:
            是否受保护
        """
        if spans is None:
            spans = self.detect(text)
        
        for span in spans:
            if span.start <= position < span.end:
                return True
        return False
    
    def get_editable_positions(self, text: str) -> List[int]:
        """
        获取所有可编辑的位置索引
        
        Args:
            text: 输入文本
            
        Returns:
            可编辑位置索引的列表
        """
        protected = self.get_protected_positions(text)
        return [i for i in range(len(text)) if i not in protected]
    
    def add_protected_word(self, word: str):
        """添加自定义保护词汇"""
        self.custom_protected_words.add(word)
    
    def add_protected_words(self, words: List[str]):
        """批量添加自定义保护词汇"""
        self.custom_protected_words.update(words)
    
    def add_custom_pattern(self, pattern: str, type_name: str = "custom"):
        """添加自定义正则模式"""
        try:
            compiled = re.compile(pattern)
            self.custom_patterns.append((compiled, type_name))
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{pattern}': {e}")
    
    def visualize(self, text: str) -> str:
        """
        可视化受保护片段（用于调试）
        
        Args:
            text: 输入文本
            
        Returns:
            带标记的文本
        """
        spans = self.detect(text)
        if not spans:
            return text
        
        # 从后往前标记，避免位置偏移
        result = list(text)
        for span in reversed(spans):
            result.insert(span.end, f"</{span.span_type.value}>")
            result.insert(span.start, f"<{span.span_type.value}>")
        
        return ''.join(result)


# 便捷函数
def create_default_detector() -> ProtectedSpanDetector:
    """创建默认保护检测器"""
    return ProtectedSpanDetector()


def create_document_detector() -> ProtectedSpanDetector:
    """创建公文场景的保护检测器（全部启用）"""
    return ProtectedSpanDetector(
        enable_doc_number=True,
        enable_date=True,
        enable_amount=True,
        enable_clause_number=True,
        enable_organization=True,
        enable_law_name=True,
        enable_fixed_phrase=True,
    )


def create_minimal_detector() -> ProtectedSpanDetector:
    """创建最小保护检测器（仅保护日期和金额）"""
    return ProtectedSpanDetector(
        enable_doc_number=False,
        enable_date=True,
        enable_amount=True,
        enable_clause_number=False,
        enable_organization=False,
        enable_law_name=False,
        enable_fixed_phrase=False,
    )
