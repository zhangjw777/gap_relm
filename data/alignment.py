"""
字符级对齐模块
使用 Levenshtein 或 difflib 实现 (source, target) 的字符级编辑对齐
"""

from enum import IntEnum
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import difflib

# 尝试导入 python-Levenshtein，如果不存在则使用 difflib 作为后备
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("Warning: python-Levenshtein not installed, falling back to difflib")


class EditType(IntEnum):
    """编辑操作类型"""
    KEEP = 0      # 保持不变
    DELETE = 1    # 删除字符
    REPLACE = 2   # 替换字符
    INSERT = 3    # 插入字符


@dataclass
class EditOperation:
    """单个编辑操作"""
    op_type: EditType
    source_pos: int           # 在源序列中的位置 (-1 表示插入)
    target_pos: int           # 在目标序列中的位置 (-1 表示删除)
    source_char: Optional[str] = None  # 源字符
    target_char: Optional[str] = None  # 目标字符
    
    def __repr__(self):
        if self.op_type == EditType.KEEP:
            return f"KEEP({self.source_char})"
        elif self.op_type == EditType.DELETE:
            return f"DELETE({self.source_char})"
        elif self.op_type == EditType.REPLACE:
            return f"REPLACE({self.source_char}->{self.target_char})"
        elif self.op_type == EditType.INSERT:
            return f"INSERT({self.target_char})"
        return f"Unknown({self.op_type})"


class AlignmentResult(NamedTuple):
    """对齐结果"""
    operations: List[EditOperation]  # 编辑操作序列
    source: str                      # 源字符串
    target: str                      # 目标字符串
    edit_distance: int               # 编辑距离


class CharacterAligner:
    """字符级对齐器"""
    
    def __init__(self, algorithm: str = "levenshtein"):
        """
        Args:
            algorithm: 对齐算法，"levenshtein" 或 "difflib"
        """
        self.algorithm = algorithm
        
        if algorithm == "levenshtein" and not HAS_LEVENSHTEIN:
            print("Warning: Levenshtein not available, using difflib instead")
            self.algorithm = "difflib"
    
    def align(self, source: str, target: str) -> AlignmentResult:
        """
        对齐源字符串和目标字符串
        
        Args:
            source: 源字符串（错误句）
            target: 目标字符串（正确句）
            
        Returns:
            AlignmentResult 包含编辑操作序列
        """
        if self.algorithm == "levenshtein":
            return self._align_levenshtein(source, target)
        else:
            return self._align_difflib(source, target)
    
    def _align_levenshtein(self, source: str, target: str) -> AlignmentResult:
        """使用 Levenshtein 库进行对齐"""
        # 获取编辑操作
        editops = Levenshtein.editops(source, target)
        edit_distance = len(editops)
        
        # 转换为我们的操作格式
        operations = []
        
        # 构建操作索引集合
        delete_positions = set()
        replace_ops = {}  # source_pos -> target_pos
        insert_ops = {}   # target_pos -> source_pos (插入点)
        
        for op, src_pos, tgt_pos in editops:
            if op == 'delete':
                delete_positions.add(src_pos)
            elif op == 'replace':
                replace_ops[src_pos] = tgt_pos
            elif op == 'insert':
                # 插入操作：在 source 的某个位置之后插入
                insert_ops[tgt_pos] = src_pos
        
        # 遍历源字符串，生成操作序列
        src_idx = 0
        tgt_idx = 0
        
        while src_idx < len(source) or tgt_idx < len(target):
            # 检查是否需要在当前位置插入
            while tgt_idx in insert_ops and tgt_idx < len(target):
                operations.append(EditOperation(
                    op_type=EditType.INSERT,
                    source_pos=-1,
                    target_pos=tgt_idx,
                    source_char=None,
                    target_char=target[tgt_idx]
                ))
                tgt_idx += 1
            
            if src_idx >= len(source):
                break
            
            # 处理当前源字符
            if src_idx in delete_positions:
                operations.append(EditOperation(
                    op_type=EditType.DELETE,
                    source_pos=src_idx,
                    target_pos=-1,
                    source_char=source[src_idx],
                    target_char=None
                ))
                src_idx += 1
            elif src_idx in replace_ops:
                operations.append(EditOperation(
                    op_type=EditType.REPLACE,
                    source_pos=src_idx,
                    target_pos=tgt_idx,
                    source_char=source[src_idx],
                    target_char=target[tgt_idx]
                ))
                src_idx += 1
                tgt_idx += 1
            else:
                # KEEP
                operations.append(EditOperation(
                    op_type=EditType.KEEP,
                    source_pos=src_idx,
                    target_pos=tgt_idx,
                    source_char=source[src_idx],
                    target_char=target[tgt_idx]
                ))
                src_idx += 1
                tgt_idx += 1
        
        # 处理剩余的插入
        while tgt_idx < len(target):
            operations.append(EditOperation(
                op_type=EditType.INSERT,
                source_pos=-1,
                target_pos=tgt_idx,
                source_char=None,
                target_char=target[tgt_idx]
            ))
            tgt_idx += 1
        
        return AlignmentResult(
            operations=operations,
            source=source,
            target=target,
            edit_distance=edit_distance
        )
    
    def _align_difflib(self, source: str, target: str) -> AlignmentResult:
        """使用 difflib 进行对齐"""
        matcher = difflib.SequenceMatcher(None, source, target)
        opcodes = matcher.get_opcodes()
        
        operations = []
        edit_distance = 0
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                # KEEP 操作
                for src_pos, tgt_pos in zip(range(i1, i2), range(j1, j2)):
                    operations.append(EditOperation(
                        op_type=EditType.KEEP,
                        source_pos=src_pos,
                        target_pos=tgt_pos,
                        source_char=source[src_pos],
                        target_char=target[tgt_pos]
                    ))
            
            elif tag == 'replace':
                # 处理替换：可能长度不等
                src_len = i2 - i1
                tgt_len = j2 - j1
                
                # 先处理替换部分（较短的长度）
                min_len = min(src_len, tgt_len)
                for k in range(min_len):
                    operations.append(EditOperation(
                        op_type=EditType.REPLACE,
                        source_pos=i1 + k,
                        target_pos=j1 + k,
                        source_char=source[i1 + k],
                        target_char=target[j1 + k]
                    ))
                    edit_distance += 1
                
                # 如果源更长，剩余部分是删除
                for k in range(min_len, src_len):
                    operations.append(EditOperation(
                        op_type=EditType.DELETE,
                        source_pos=i1 + k,
                        target_pos=-1,
                        source_char=source[i1 + k],
                        target_char=None
                    ))
                    edit_distance += 1
                
                # 如果目标更长，剩余部分是插入
                for k in range(min_len, tgt_len):
                    operations.append(EditOperation(
                        op_type=EditType.INSERT,
                        source_pos=-1,
                        target_pos=j1 + k,
                        source_char=None,
                        target_char=target[j1 + k]
                    ))
                    edit_distance += 1
            
            elif tag == 'delete':
                # DELETE 操作
                for src_pos in range(i1, i2):
                    operations.append(EditOperation(
                        op_type=EditType.DELETE,
                        source_pos=src_pos,
                        target_pos=-1,
                        source_char=source[src_pos],
                        target_char=None
                    ))
                    edit_distance += 1
            
            elif tag == 'insert':
                # INSERT 操作
                for tgt_pos in range(j1, j2):
                    operations.append(EditOperation(
                        op_type=EditType.INSERT,
                        source_pos=-1,
                        target_pos=tgt_pos,
                        source_char=None,
                        target_char=target[tgt_pos]
                    ))
                    edit_distance += 1
        
        return AlignmentResult(
            operations=operations,
            source=source,
            target=target,
            edit_distance=edit_distance
        )
    
    def verify_alignment(self, result: AlignmentResult) -> bool:
        """
        验证对齐结果是否正确
        
        通过应用编辑操作重建目标字符串，检查是否与原目标一致
        """
        reconstructed = []
        
        for op in result.operations:
            if op.op_type == EditType.KEEP:
                reconstructed.append(op.source_char)
            elif op.op_type == EditType.REPLACE:
                reconstructed.append(op.target_char)
            elif op.op_type == EditType.INSERT:
                reconstructed.append(op.target_char)
            # DELETE 不添加任何字符
        
        reconstructed_str = ''.join(reconstructed)
        return reconstructed_str == result.target
    
    def get_alignment_stats(self, result: AlignmentResult) -> dict:
        """
        获取对齐统计信息
        """
        stats = {
            'keep_count': 0,
            'delete_count': 0,
            'replace_count': 0,
            'insert_count': 0,
            'source_length': len(result.source),
            'target_length': len(result.target),
            'edit_distance': result.edit_distance,
        }
        
        for op in result.operations:
            if op.op_type == EditType.KEEP:
                stats['keep_count'] += 1
            elif op.op_type == EditType.DELETE:
                stats['delete_count'] += 1
            elif op.op_type == EditType.REPLACE:
                stats['replace_count'] += 1
            elif op.op_type == EditType.INSERT:
                stats['insert_count'] += 1
        
        return stats


def visualize_alignment(result: AlignmentResult) -> str:
    """
    可视化对齐结果
    
    Returns:
        可视化字符串
    """
    lines = []
    lines.append(f"Source: {result.source}")
    lines.append(f"Target: {result.target}")
    lines.append(f"Edit distance: {result.edit_distance}")
    lines.append("")
    lines.append("Operations:")
    
    for i, op in enumerate(result.operations):
        lines.append(f"  {i}: {op}")
    
    return '\n'.join(lines)


# 便捷函数
def align_texts(
    source: str, 
    target: str, 
    algorithm: str = "levenshtein"
) -> AlignmentResult:
    """
    对齐两个文本的便捷函数
    
    Args:
        source: 源文本
        target: 目标文本
        algorithm: 对齐算法
        
    Returns:
        AlignmentResult
    """
    aligner = CharacterAligner(algorithm=algorithm)
    return aligner.align(source, target)
