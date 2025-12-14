"""
规则造错模块
支持删字(Skip)、重复字(Multiply)、错字(Replace)三种错误类型
"""

import random
import math
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .confusion_set import ConfusionSet, create_default_confusion_set
from .protected_span import ProtectedSpanDetector, ProtectedSpan, create_default_detector


class ErrorType(Enum):
    """错误类型枚举
    
    S (Skip): 删字 - 模拟漏打字符
    M (Multiply): 重复字 - 模拟多打字符
    R (Replace): 错字 - 替换为混淆字符
    """
    SKIP = "S"          # 删字
    MULTIPLY = "M"      # 重复字（多字）
    REPLACE = "R"       # 错字


@dataclass
class ErrorEdit:
    """单个错误编辑"""
    position: int           # 在原文中的位置
    error_type: ErrorType   # 错误类型
    original_char: str      # 原字符
    error_char: str         # 错误后的字符（删除时为空，重复时为重复字符，替换时为混淆字符）
    
    def __repr__(self):
        if self.error_type == ErrorType.SKIP:
            return f"SKIP({self.original_char})@{self.position}"
        elif self.error_type == ErrorType.MULTIPLY:
            return f"MULTIPLY({self.original_char}→{self.original_char}{self.error_char})@{self.position}"
        else:
            return f"REPLACE({self.original_char}→{self.error_char})@{self.position}"


@dataclass
class CorruptionResult:
    """造错结果"""
    original: str                           # 原始正确句子
    corrupted: str                          # 造错后的句子
    edits: List[ErrorEdit] = field(default_factory=list)  # 错误编辑列表
    is_corrupted: bool = False              # 是否进行了造错
    
    def to_training_pair(self) -> Tuple[str, str]:
        """转换为训练数据对 (错误句, 正确句)"""
        return (self.corrupted, self.original)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'original': self.original,
            'corrupted': self.corrupted,
            'is_corrupted': self.is_corrupted,
            'num_edits': len(self.edits),
            'edits': [
                {
                    'position': e.position,
                    'type': e.error_type.value,
                    'original': e.original_char,
                    'error': e.error_char
                }
                for e in self.edits
            ]
        }


class TruncatedPoisson:
    """截断泊松分布
    
    用于采样编辑数量，支持截断到 [min_val, max_val]
    """
    
    def __init__(
        self,
        lambda_: float = 1.5,
        min_val: int = 1,
        max_val: int = 4
    ):
        self.lambda_ = lambda_
        self.min_val = min_val
        self.max_val = max_val
    
    def sample(self) -> int:
        """采样一个值"""
        while True:
            val = self._poisson_sample()
            if self.min_val <= val <= self.max_val:
                return val
    
    def _poisson_sample(self) -> int:
        """标准泊松采样（Knuth算法）"""
        L = math.exp(-self.lambda_)
        k = 0
        p = 1.0
        
        while p > L:
            k += 1
            p *= random.random()
        
        return k - 1


class ErrorGenerator:
    """
    规则造错器
    
    按照可调参数化流程生成训练数据：
    1. 以概率 p_corrupt 决定是否造错
    2. 若造错，从截断泊松分布采样编辑数量 n_edits
    3. 对每个编辑，按概率 π 抽取错误类型
    4. 执行对应规则，遵守保护约束
    
    主要参数 (四个旋钮)：
    - p_corrupt: 造错概率 (0.5-0.8)
    - lambda_: 编辑数量的泊松参数 (1-2)
    - pi: 各类型概率 [π_S, π_M, π_R]
    - max_edits: 单句最大编辑数 (默认4)
    """
    
    def __init__(
        self,
        # 主要参数
        p_corrupt: float = 0.7,
        lambda_: float = 1.5,
        pi_skip: float = 0.2,           # 删字概率 π_S
        pi_multiply: float = 0.3,       # 重复字概率 π_M
        pi_replace: float = 0.5,        # 错字概率 π_R
        max_edits_per_sent: int = 4,
        max_insert_k: int = 3,          # 单次最大重复数 K
        
        # 混淆集
        confusion_set: Optional[ConfusionSet] = None,
        
        # 保护约束
        protected_detector: Optional[ProtectedSpanDetector] = None,
        enable_protection: bool = True,
        
        # 其他选项
        min_sentence_length: int = 5,   # 最小句子长度（太短不造错）
        skip_punct: bool = True,        # 跳过标点符号
        seed: Optional[int] = None,
    ):
        """
        初始化造错器
        
        Args:
            p_corrupt: 造错概率
            lambda_: 泊松分布参数（控制平均编辑数）
            pi_skip: 删字概率
            pi_multiply: 重复字概率
            pi_replace: 错字概率
            max_edits_per_sent: 单句最大编辑数
            max_insert_k: 单次最大重复数
            confusion_set: 混淆集（用于替换错误）
            protected_detector: 保护检测器
            enable_protection: 是否启用保护约束
            min_sentence_length: 最小句子长度
            skip_punct: 是否跳过标点符号
            seed: 随机种子
        """
        # 验证概率参数
        pi_total = pi_skip + pi_multiply + pi_replace
        if abs(pi_total - 1.0) > 1e-6:
            # 归一化
            pi_skip /= pi_total
            pi_multiply /= pi_total
            pi_replace /= pi_total
        
        self.p_corrupt = p_corrupt
        self.lambda_ = lambda_
        self.pi = {
            ErrorType.SKIP: pi_skip,
            ErrorType.MULTIPLY: pi_multiply,
            ErrorType.REPLACE: pi_replace,
        }
        self.max_edits = max_edits_per_sent
        self.max_insert_k = max_insert_k
        self.min_sentence_length = min_sentence_length
        self.skip_punct = skip_punct
        
        # 截断泊松分布
        self.poisson_sampler = TruncatedPoisson(
            lambda_=lambda_,
            min_val=1,
            max_val=max_edits_per_sent
        )
        
        # 混淆集
        self.confusion_set = confusion_set or create_default_confusion_set()
        
        # 保护检测器
        self.protected_detector = protected_detector or create_default_detector()
        self.enable_protection = enable_protection
        
        # 中文标点
        self.punctuation = set('。，、；：？！""''【】《》（）—…·「」『』〈〉〔〕')
        
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
    
    def corrupt(self, sentence: str) -> CorruptionResult:
        """
        对单个句子进行造错
        
        Args:
            sentence: 正确的原始句子
            
        Returns:
            CorruptionResult
        """
        # 检查是否太短
        if len(sentence) < self.min_sentence_length:
            return CorruptionResult(
                original=sentence,
                corrupted=sentence,
                edits=[],
                is_corrupted=False
            )
        
        # 以概率 p_corrupt 决定是否造错
        if random.random() > self.p_corrupt:
            return CorruptionResult(
                original=sentence,
                corrupted=sentence,
                edits=[],
                is_corrupted=False
            )
        
        # 获取可编辑位置
        editable_positions = self._get_editable_positions(sentence)
        
        if not editable_positions:
            return CorruptionResult(
                original=sentence,
                corrupted=sentence,
                edits=[],
                is_corrupted=False
            )
        
        # 采样编辑数量
        n_edits = min(self.poisson_sampler.sample(), len(editable_positions))
        
        # 随机选择编辑位置（不重复）
        edit_positions = random.sample(editable_positions, n_edits)
        edit_positions.sort(reverse=True)  # 从后往前处理，避免位置偏移
        
        # 生成错误
        edits = []
        corrupted = list(sentence)
        
        for pos in edit_positions:
            # 采样错误类型
            error_type = self._sample_error_type()
            
            # 执行错误
            edit = self._apply_error(corrupted, pos, error_type)
            if edit:
                edits.append(edit)
        
        # 反转edits以保持原始顺序
        edits.reverse()
        
        return CorruptionResult(
            original=sentence,
            corrupted=''.join(corrupted),
            edits=edits,
            is_corrupted=len(edits) > 0
        )
    
    def corrupt_batch(
        self,
        sentences: List[str],
        show_progress: bool = False
    ) -> List[CorruptionResult]:
        """
        批量造错
        
        Args:
            sentences: 句子列表
            show_progress: 是否显示进度
            
        Returns:
            造错结果列表
        """
        results = []
        iterator = sentences
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(sentences, desc="Corrupting")
            except ImportError:
                pass
        
        for sentence in iterator:
            results.append(self.corrupt(sentence))
        
        return results
    
    def _get_editable_positions(self, sentence: str) -> List[int]:
        """获取可编辑的位置"""
        # 获取保护位置
        protected_positions = set()
        if self.enable_protection:
            protected_positions = self.protected_detector.get_protected_positions(sentence)
        
        # 筛选可编辑位置
        editable = []
        for i, char in enumerate(sentence):
            # 跳过保护位置
            if i in protected_positions:
                continue
            
            # 跳过标点
            if self.skip_punct and char in self.punctuation:
                continue
            
            # 跳过空白字符
            if char.isspace():
                continue
            
            editable.append(i)
        
        return editable
    
    def _sample_error_type(self) -> ErrorType:
        """按概率采样错误类型"""
        r = random.random()
        cumsum = 0.0
        
        for error_type, prob in self.pi.items():
            cumsum += prob
            if r < cumsum:
                return error_type
        
        return ErrorType.REPLACE  # 默认
    
    def _apply_error(
        self,
        chars: List[str],
        pos: int,
        error_type: ErrorType
    ) -> Optional[ErrorEdit]:
        """
        应用单个错误
        
        Args:
            chars: 字符列表（原地修改）
            pos: 位置
            error_type: 错误类型
            
        Returns:
            ErrorEdit 或 None（如果无法应用）
        """
        original_char = chars[pos]
        
        if error_type == ErrorType.SKIP:
            # 删字：删除该位置的字符
            chars.pop(pos)
            return ErrorEdit(
                position=pos,
                error_type=ErrorType.SKIP,
                original_char=original_char,
                error_char=""
            )
        
        elif error_type == ErrorType.MULTIPLY:
            # 重复字：在该位置后插入重复字符
            # 采样重复次数（1 到 max_insert_k）
            k = random.randint(1, self.max_insert_k)
            insert_chars = original_char * k
            chars.insert(pos + 1, insert_chars)
            return ErrorEdit(
                position=pos,
                error_type=ErrorType.MULTIPLY,
                original_char=original_char,
                error_char=insert_chars
            )
        
        elif error_type == ErrorType.REPLACE:
            # 错字：替换为混淆字符
            confusion_char = self.confusion_set.get_random_confusion(original_char)
            
            if confusion_char:
                chars[pos] = confusion_char
                return ErrorEdit(
                    position=pos,
                    error_type=ErrorType.REPLACE,
                    original_char=original_char,
                    error_char=confusion_char
                )
            else:
                # 没有混淆字符，尝试其他类型
                # 优先尝试重复
                k = random.randint(1, self.max_insert_k)
                insert_chars = original_char * k
                chars.insert(pos + 1, insert_chars)
                return ErrorEdit(
                    position=pos,
                    error_type=ErrorType.MULTIPLY,
                    original_char=original_char,
                    error_char=insert_chars
                )
        
        return None
    
    def set_params(
        self,
        p_corrupt: Optional[float] = None,
        lambda_: Optional[float] = None,
        pi_skip: Optional[float] = None,
        pi_multiply: Optional[float] = None,
        pi_replace: Optional[float] = None,
        max_edits: Optional[int] = None,
    ):
        """
        更新参数（用于网格搜索调参）
        
        Args:
            p_corrupt: 造错概率
            lambda_: 泊松参数
            pi_skip: 删字概率
            pi_multiply: 重复字概率
            pi_replace: 错字概率
            max_edits: 最大编辑数
        """
        if p_corrupt is not None:
            self.p_corrupt = p_corrupt
        
        if lambda_ is not None:
            self.lambda_ = lambda_
            self.poisson_sampler = TruncatedPoisson(
                lambda_=lambda_,
                min_val=1,
                max_val=self.max_edits
            )
        
        if any([pi_skip, pi_multiply, pi_replace]):
            new_pi_skip = pi_skip if pi_skip is not None else self.pi[ErrorType.SKIP]
            new_pi_multiply = pi_multiply if pi_multiply is not None else self.pi[ErrorType.MULTIPLY]
            new_pi_replace = pi_replace if pi_replace is not None else self.pi[ErrorType.REPLACE]
            
            # 归一化
            total = new_pi_skip + new_pi_multiply + new_pi_replace
            self.pi = {
                ErrorType.SKIP: new_pi_skip / total,
                ErrorType.MULTIPLY: new_pi_multiply / total,
                ErrorType.REPLACE: new_pi_replace / total,
            }
        
        if max_edits is not None:
            self.max_edits = max_edits
            self.poisson_sampler = TruncatedPoisson(
                lambda_=self.lambda_,
                min_val=1,
                max_val=max_edits
            )
    
    def get_params(self) -> Dict[str, Any]:
        """获取当前参数"""
        return {
            'p_corrupt': self.p_corrupt,
            'lambda': self.lambda_,
            'pi_skip': self.pi[ErrorType.SKIP],
            'pi_multiply': self.pi[ErrorType.MULTIPLY],
            'pi_replace': self.pi[ErrorType.REPLACE],
            'max_edits': self.max_edits,
            'max_insert_k': self.max_insert_k,
        }
    
    def stats(self, results: List[CorruptionResult]) -> Dict[str, Any]:
        """
        统计造错结果
        
        Args:
            results: 造错结果列表
            
        Returns:
            统计信息字典
        """
        total = len(results)
        corrupted = sum(1 for r in results if r.is_corrupted)
        
        type_counts = defaultdict(int)
        edit_counts = []
        
        for r in results:
            if r.is_corrupted:
                edit_counts.append(len(r.edits))
                for edit in r.edits:
                    type_counts[edit.error_type.value] += 1
        
        avg_edits = sum(edit_counts) / len(edit_counts) if edit_counts else 0
        
        return {
            'total_sentences': total,
            'corrupted_sentences': corrupted,
            'corruption_rate': corrupted / total if total > 0 else 0,
            'avg_edits_per_corrupted': avg_edits,
            'error_type_distribution': dict(type_counts),
            'edit_count_distribution': dict(sorted(
                defaultdict(int, {str(k): edit_counts.count(k) for k in set(edit_counts)}).items()
            )),
        }


# 便捷函数
def create_default_error_generator() -> ErrorGenerator:
    """创建默认造错器"""
    return ErrorGenerator()


def create_balanced_error_generator() -> ErrorGenerator:
    """创建平衡的造错器（三种错误类型等概率）"""
    return ErrorGenerator(
        pi_skip=0.33,
        pi_multiply=0.33,
        pi_replace=0.34
    )


def create_insert_focused_error_generator() -> ErrorGenerator:
    """创建侧重插删的造错器（增加 M 和 S 的比例）"""
    return ErrorGenerator(
        pi_skip=0.35,
        pi_multiply=0.35,
        pi_replace=0.30
    )


def create_conservative_error_generator() -> ErrorGenerator:
    """创建保守的造错器（低造错率，少编辑）"""
    return ErrorGenerator(
        p_corrupt=0.5,
        lambda_=1.0,
        max_edits_per_sent=2
    )


def create_aggressive_error_generator() -> ErrorGenerator:
    """创建激进的造错器（高造错率，多编辑）"""
    return ErrorGenerator(
        p_corrupt=0.8,
        lambda_=2.0,
        max_edits_per_sent=4
    )
