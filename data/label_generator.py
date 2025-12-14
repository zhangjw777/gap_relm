"""
标签生成模块
从对齐结果生成 Planner 标签 (op, k) 和 Gold Template
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import json

from .alignment import AlignmentResult, EditOperation, EditType


@dataclass
class PlannerLabels:
    """Planner 标签
    
    对于源序列中的每个字符 x_i:
    - op_labels[i]: 操作类型 (KEEP=0, DELETE=1, REPLACE=2)
    - insert_labels[i]: 在 x_i 后插入的字符数量 (0 ~ max_insert_num)
    """
    op_labels: List[int]           # 长度 = len(source)
    insert_labels: List[int]       # 长度 = len(source)
    source: str
    target: str


@dataclass
class GoldTemplate:
    """Gold Template (用于训练 Infiller)
    
    模板结构: 根据 op 和 insert 标签构建的 token 序列
    - template_tokens: 模板 token 序列 (包含 [MASK] 和保留的字符)
    - gold_tokens: 每个 [MASK] 位置对应的正确 token
    - mask_positions: [MASK] 在模板中的位置列表
    """
    template_tokens: List[str]     # 模板序列
    gold_tokens: List[str]         # [MASK] 位置的正确答案
    mask_positions: List[int]      # [MASK] 的位置索引
    source: str
    target: str


@dataclass
class ProcessedSample:
    """完整的处理后样本"""
    source: str                    # 原始源文本
    target: str                    # 目标文本
    planner_labels: PlannerLabels  # Planner 标签
    gold_template: GoldTemplate    # Gold Template
    alignment_result: AlignmentResult  # 对齐结果（用于调试和可视化）
    
    # 元信息
    sample_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LabelGenerator:
    """标签生成器
    
    从 (source, target) 对齐结果生成 Planner 标签
    遵循 Left-Association Rule: 插入操作归属到左侧字符
    """
    
    def __init__(
        self,
        max_insert_num: int = 3,
        enable_insert: bool = True,
        enable_delete: bool = True,
    ):
        """
        Args:
            max_insert_num: 最大插入数量 K
            enable_insert: 是否启用插入标签
            enable_delete: 是否启用删除标签
        """
        self.max_insert_num = max_insert_num
        self.enable_insert = enable_insert
        self.enable_delete = enable_delete
    
    def generate(self, alignment: AlignmentResult) -> PlannerLabels:
        """
        从对齐结果生成 Planner 标签
        
        Args:
            alignment: 字符级对齐结果
            
        Returns:
            PlannerLabels
        """
        source = alignment.source
        n = len(source)
        
        # 初始化标签
        op_labels = [0] * n       # 默认 KEEP
        insert_labels = [0] * n   # 默认不插入
        
        # 用于跟踪每个源位置的插入计数
        # 注意：插入归属到左侧字符（Left-Association Rule）
        pending_inserts = [[] for _ in range(n + 1)]  # n+1 用于处理句首插入
        
        # 遍历对齐操作，分配到源位置
        current_src_pos = 0
        last_src_pos = -1  # 用于处理连续插入
        
        for op in alignment.operations:
            if op.op_type == EditType.KEEP:
                current_src_pos = op.source_pos + 1
                last_src_pos = op.source_pos
            
            elif op.op_type == EditType.DELETE:
                if self.enable_delete:
                    op_labels[op.source_pos] = 1  # DELETE
                # 如果不启用删除，保持为 KEEP
                current_src_pos = op.source_pos + 1
                last_src_pos = op.source_pos
            
            elif op.op_type == EditType.REPLACE:
                op_labels[op.source_pos] = 2  # REPLACE
                current_src_pos = op.source_pos + 1
                last_src_pos = op.source_pos
            
            elif op.op_type == EditType.INSERT:
                if self.enable_insert:
                    # Left-Association: 插入归属到左侧字符
                    # 如果是句首插入 (last_src_pos == -1)，特殊处理
                    if last_src_pos >= 0:
                        pending_inserts[last_src_pos].append(op.target_char)
                    else:
                        # 句首插入，归属到位置 0
                        pending_inserts[0].append(op.target_char)
        
        # 将 pending_inserts 转换为 insert_labels
        for i in range(n):
            if pending_inserts[i]:
                # 限制最大插入数
                insert_count = min(len(pending_inserts[i]), self.max_insert_num)
                insert_labels[i] = insert_count
        
        # 处理句首插入的特殊情况
        # 如果有句首插入但第一个字符被删除，需要调整
        if pending_inserts[0] and op_labels[0] == 1 and n > 1:
            # 将句首插入移到第一个非删除字符
            for i in range(1, n):
                if op_labels[i] != 1:  # 找到第一个非删除字符
                    insert_labels[i] = min(
                        len(pending_inserts[0]) + insert_labels[i],
                        self.max_insert_num
                    )
                    break
        
        return PlannerLabels(
            op_labels=op_labels,
            insert_labels=insert_labels,
            source=source,
            target=alignment.target
        )
    
    def validate_labels(self, labels: PlannerLabels) -> bool:
        """验证标签的有效性"""
        n = len(labels.source)
        
        # 检查长度一致性
        if len(labels.op_labels) != n or len(labels.insert_labels) != n:
            return False
        
        # 检查值范围
        for op in labels.op_labels:
            if op not in [0, 1, 2]:
                return False
        
        for ins in labels.insert_labels:
            if ins < 0 or ins > self.max_insert_num:
                return False
        
        return True


class GoldTemplateBuilder:
    """Gold Template 构建器
    
    根据 Planner 标签构建训练 Infiller 所需的 Gold Template
    """
    
    MASK_TOKEN = "[MASK]"
    
    def __init__(self, max_insert_num: int = 3):
        """
        Args:
            max_insert_num: 最大插入数量
        """
        self.max_insert_num = max_insert_num
    
    def build(
        self, 
        labels: PlannerLabels, 
        alignment: AlignmentResult
    ) -> GoldTemplate:
        """
        根据标签和对齐结果构建 Gold Template
        
        模板构建规则:
        - KEEP: 输出原字符
        - REPLACE: 输出 [MASK]
        - DELETE: 不输出
        - 在每个字符后，根据 insert_label 输出相应数量的 [MASK]
        
        Args:
            labels: Planner 标签
            alignment: 对齐结果（用于获取正确的 gold tokens）
            
        Returns:
            GoldTemplate
        """
        source = labels.source
        n = len(source)
        
        template_tokens = []
        gold_tokens = []
        mask_positions = []
        
        # 构建从对齐操作中提取插入字符的映射
        insert_chars = self._extract_insert_chars(alignment)
        
        for i in range(n):
            op = labels.op_labels[i]
            insert_count = labels.insert_labels[i]
            
            # 处理当前字符
            if op == 0:  # KEEP
                template_tokens.append(source[i])
            elif op == 1:  # DELETE
                # 不输出任何字符
                pass
            elif op == 2:  # REPLACE
                mask_positions.append(len(template_tokens))
                template_tokens.append(self.MASK_TOKEN)
                
                # 获取替换的正确字符
                replace_char = self._get_replace_char(alignment, i)
                gold_tokens.append(replace_char)
            
            # 处理插入（在当前字符后）
            if insert_count > 0 and op != 1:  # 删除的字符后不插入
                insert_chars_for_pos = insert_chars.get(i, [])
                for j in range(insert_count):
                    mask_positions.append(len(template_tokens))
                    template_tokens.append(self.MASK_TOKEN)
                    
                    # 获取插入的正确字符
                    if j < len(insert_chars_for_pos):
                        gold_tokens.append(insert_chars_for_pos[j])
                    else:
                        # 如果没有对应的插入字符，使用占位符
                        gold_tokens.append("[UNK]")
        
        return GoldTemplate(
            template_tokens=template_tokens,
            gold_tokens=gold_tokens,
            mask_positions=mask_positions,
            source=source,
            target=labels.target
        )
    
    def _extract_insert_chars(
        self, 
        alignment: AlignmentResult
    ) -> Dict[int, List[str]]:
        """
        从对齐结果中提取每个源位置后的插入字符
        
        Returns:
            字典: 源位置 -> 插入字符列表
        """
        insert_chars = {}
        last_src_pos = -1
        
        for op in alignment.operations:
            if op.op_type == EditType.KEEP:
                last_src_pos = op.source_pos
            elif op.op_type == EditType.DELETE:
                last_src_pos = op.source_pos
            elif op.op_type == EditType.REPLACE:
                last_src_pos = op.source_pos
            elif op.op_type == EditType.INSERT:
                # 归属到左侧位置
                pos = max(0, last_src_pos) if last_src_pos >= 0 else 0
                if pos not in insert_chars:
                    insert_chars[pos] = []
                insert_chars[pos].append(op.target_char)
        
        return insert_chars
    
    def _get_replace_char(self, alignment: AlignmentResult, src_pos: int) -> str:
        """获取指定源位置的替换字符"""
        for op in alignment.operations:
            if op.op_type == EditType.REPLACE and op.source_pos == src_pos:
                return op.target_char
        return "[UNK]"
    
    def template_to_string(self, template: GoldTemplate) -> str:
        """将模板转换为字符串（用于调试）"""
        return ''.join(template.template_tokens)
    
    def verify_template(self, template: GoldTemplate) -> bool:
        """验证模板的正确性"""
        # 检查 mask 位置和 gold tokens 数量匹配
        if len(template.mask_positions) != len(template.gold_tokens):
            return False
        
        # 检查 mask 位置有效
        for pos in template.mask_positions:
            if pos >= len(template.template_tokens):
                return False
            if template.template_tokens[pos] != self.MASK_TOKEN:
                return False
        
        return True


class SampleProcessor:
    """样本处理器 - 完整的数据处理流水线"""
    
    def __init__(
        self,
        aligner,
        label_generator: LabelGenerator,
        template_builder: GoldTemplateBuilder,
    ):
        """
        Args:
            aligner: CharacterAligner 实例
            label_generator: LabelGenerator 实例
            template_builder: GoldTemplateBuilder 实例
        """
        self.aligner = aligner
        self.label_generator = label_generator
        self.template_builder = template_builder
    
    def process(
        self, 
        source: str, 
        target: str,
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedSample:
        """
        处理单个 (source, target) 对
        
        Args:
            source: 源文本（错误句）
            target: 目标文本（正确句）
            sample_id: 样本ID（可选）
            metadata: 元信息（可选）
            
        Returns:
            ProcessedSample
        """
        # 1. 字符级对齐
        alignment = self.aligner.align(source, target)
        
        # 2. 生成 Planner 标签
        planner_labels = self.label_generator.generate(alignment)
        
        # 3. 构建 Gold Template
        gold_template = self.template_builder.build(planner_labels, alignment)
        
        return ProcessedSample(
            source=source,
            target=target,
            planner_labels=planner_labels,
            gold_template=gold_template,
            alignment_result=alignment,
            sample_id=sample_id,
            metadata=metadata or {}
        )
    
    def process_batch(
        self,
        pairs: List[Tuple[str, str]],
        sample_ids: Optional[List[str]] = None
    ) -> List[ProcessedSample]:
        """批量处理"""
        results = []
        for i, (source, target) in enumerate(pairs):
            sample_id = sample_ids[i] if sample_ids else f"sample_{i}"
            result = self.process(source, target, sample_id)
            results.append(result)
        return results


def create_sample_processor(
    max_insert_num: int = 3,
    enable_insert: bool = True,
    enable_delete: bool = True,
    alignment_algorithm: str = "levenshtein"
) -> SampleProcessor:
    """
    创建样本处理器的便捷函数
    """
    from .alignment import CharacterAligner
    
    aligner = CharacterAligner(algorithm=alignment_algorithm)
    label_generator = LabelGenerator(
        max_insert_num=max_insert_num,
        enable_insert=enable_insert,
        enable_delete=enable_delete
    )
    template_builder = GoldTemplateBuilder(max_insert_num=max_insert_num)
    
    return SampleProcessor(aligner, label_generator, template_builder)


def visualize_sample(sample: ProcessedSample) -> str:
    """可视化处理后的样本"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Sample ID: {sample.sample_id}")
    lines.append(f"Source: {sample.source}")
    lines.append(f"Target: {sample.target}")
    lines.append("")
    
    # Planner 标签
    lines.append("Planner Labels:")
    op_names = ['KEEP', 'DELETE', 'REPLACE']
    for i, char in enumerate(sample.source):
        op = sample.planner_labels.op_labels[i]
        ins = sample.planner_labels.insert_labels[i]
        lines.append(f"  [{i}] '{char}': {op_names[op]}, insert={ins}")
    lines.append("")
    
    # Gold Template
    lines.append("Gold Template:")
    template_str = ''.join(sample.gold_template.template_tokens)
    lines.append(f"  Template: {template_str}")
    lines.append(f"  Mask positions: {sample.gold_template.mask_positions}")
    lines.append(f"  Gold tokens: {sample.gold_template.gold_tokens}")
    lines.append("")
    
    # 对齐统计
    from .alignment import CharacterAligner
    aligner = CharacterAligner()
    stats = aligner.get_alignment_stats(sample.alignment_result)
    lines.append(f"Alignment Stats: {stats}")
    lines.append("=" * 60)
    
    return '\n'.join(lines)
