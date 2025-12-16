"""
生成预计算 tokenize 的训练数据

关键优化：将 tokenize 阶段从训练时移到数据生成阶段
- 存储格式：二进制分片文件 (.bin) + 索引文件 (.idx)
- 训练时直接读取 tensor，无需任何 CPU 计算
- 显著减少 DataLoader 的 CPU 瓶颈，提升 GPU 利用率

优化版本：流式处理 + 批量 tokenize，避免内存爆炸
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import struct
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass
import random

from transformers import AutoTokenizer


@dataclass
class TokenizedSample:
    """预计算的 tokenized 样本"""
    # 源序列
    input_ids: np.ndarray           # [max_seq_length], dtype=int16
    attention_mask: np.ndarray      # [max_seq_length], dtype=int8
    
    # Planner 标签
    op_labels: np.ndarray           # [max_seq_length], dtype=int8
    insert_labels: np.ndarray       # [max_seq_length], dtype=int8
    
    # 模板序列
    template_input_ids: np.ndarray  # [max_seq_length], dtype=int16
    template_attention_mask: np.ndarray  # [max_seq_length], dtype=int8
    
    # Infiller 标签
    infill_labels: np.ndarray       # [max_seq_length], dtype=int16 (-100 需要 int16)


class TokenizedDataWriter:
    """
    二进制数据写入器
    
    文件格式:
    - .bin: 连续存储的二进制 tensor 数据
    - .idx: 每个样本在 .bin 文件中的偏移和大小
    
    每个样本的二进制布局 (固定大小):
    - input_ids: max_seq_length * 2 bytes (int16)
    - attention_mask: max_seq_length * 1 byte (int8)
    - op_labels: max_seq_length * 1 byte (int8)
    - insert_labels: max_seq_length * 1 byte (int8)
    - template_input_ids: max_seq_length * 2 bytes (int16)
    - template_attention_mask: max_seq_length * 1 byte (int8)
    - infill_labels: max_seq_length * 2 bytes (int16)
    
    总大小 = max_seq_length * (2+1+1+1+2+1+2) = max_seq_length * 10 bytes
    """
    
    def __init__(self, output_prefix: str, max_seq_length: int):
        self.output_prefix = output_prefix
        self.max_seq_length = max_seq_length
        self.sample_size = max_seq_length * 10  # 每样本字节数
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.bin_file = open(f"{output_prefix}.bin", 'wb')
        self.offsets: List[int] = []
        self.current_offset = 0
    
    def write_sample(self, sample: TokenizedSample):
        """写入单个样本"""
        # 确保数据类型正确
        input_ids = sample.input_ids.astype(np.int16)
        attention_mask = sample.attention_mask.astype(np.int8)
        op_labels = sample.op_labels.astype(np.int8)
        insert_labels = sample.insert_labels.astype(np.int8)
        template_input_ids = sample.template_input_ids.astype(np.int16)
        template_attention_mask = sample.template_attention_mask.astype(np.int8)
        infill_labels = sample.infill_labels.astype(np.int16)
        
        # 写入二进制数据
        self.offsets.append(self.current_offset)
        
        self.bin_file.write(input_ids.tobytes())
        self.bin_file.write(attention_mask.tobytes())
        self.bin_file.write(op_labels.tobytes())
        self.bin_file.write(insert_labels.tobytes())
        self.bin_file.write(template_input_ids.tobytes())
        self.bin_file.write(template_attention_mask.tobytes())
        self.bin_file.write(infill_labels.tobytes())
        
        self.current_offset += self.sample_size
    
    def close(self):
        """关闭文件并写入索引"""
        self.bin_file.close()
        
        # 写入索引文件
        with open(f"{self.output_prefix}.idx", 'wb') as f:
            # 头部: max_seq_length (4 bytes) + num_samples (8 bytes)
            f.write(struct.pack('<I', self.max_seq_length))
            f.write(struct.pack('<Q', len(self.offsets)))
            # 偏移数组
            for offset in self.offsets:
                f.write(struct.pack('<Q', offset))
        
        print(f"Written {len(self.offsets)} samples to {self.output_prefix}.*")


def tokenize_batch(
    samples: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    full_mask_mode: bool = False,
) -> List[Optional[TokenizedSample]]:
    """
    批量 tokenize 样本（利用 tokenizer 的批处理能力加速）
    
    Args:
        samples: 样本字典列表
        tokenizer: HuggingFace tokenizer
        max_seq_length: 最大序列长度
        full_mask_mode: 是否使用全 MASK 模式
    
    Returns:
        TokenizedSample 列表（无效样本为 None）
    """
    if full_mask_mode:
        return tokenize_batch_full_mask(samples, tokenizer, max_seq_length)
    else:
        return tokenize_batch_sparse_mask(samples, tokenizer, max_seq_length)


def tokenize_batch_sparse_mask(
    samples: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
) -> List[Optional[TokenizedSample]]:
    """稀疏 MASK 模式的批量 tokenize（原有逻辑）"""
    # 过滤有效样本并收集文本
    valid_indices = []
    sources = []
    templates = []
    sample_data = []
    
    for i, sample in enumerate(samples):
        source = sample.get('source', '')
        if not source or len(source) > max_seq_length - 2:
            continue
        
        template_tokens = sample.get('template_tokens', list(source))
        template_str = ''.join(template_tokens).replace('[MASK]', tokenizer.mask_token)
        
        valid_indices.append(i)
        sources.append(source)
        templates.append(template_str)
        sample_data.append(sample)
    
    if not sources:
        return [None] * len(samples)
    
    # 批量 tokenize（这是最耗时的操作，批处理可以显著加速）
    source_encodings = tokenizer(
        sources,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    template_encodings = tokenizer(
        templates,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    # 构建结果
    results = [None] * len(samples)
    mask_token_id = tokenizer.mask_token_id
    
    for j, orig_idx in enumerate(valid_indices):
        sample = sample_data[j]
        op_labels = sample['op_labels']
        insert_labels = sample['insert_labels']
        gold_tokens = sample.get('gold_tokens', [])
        
        # 处理 Planner 标签
        op_labels_padded = [-100] + op_labels[:max_seq_length-2] + [-100]
        insert_labels_padded = [0] + insert_labels[:max_seq_length-2] + [0]
        
        # Padding
        op_labels_padded = op_labels_padded + [-100] * (max_seq_length - len(op_labels_padded))
        insert_labels_padded = insert_labels_padded + [0] * (max_seq_length - len(insert_labels_padded))
        
        # 构建 Infiller 标签
        template_input_ids = template_encodings['input_ids'][j]
        infill_labels = np.full(max_seq_length, -100, dtype=np.int16)
        
        mask_positions = np.where(template_input_ids == mask_token_id)[0]
        for k, pos in enumerate(mask_positions):
            if k < len(gold_tokens):
                token_id = tokenizer.convert_tokens_to_ids(gold_tokens[k])
                infill_labels[pos] = token_id
        
        results[orig_idx] = TokenizedSample(
            input_ids=source_encodings['input_ids'][j].astype(np.int16),
            attention_mask=source_encodings['attention_mask'][j].astype(np.int8),
            op_labels=np.array(op_labels_padded, dtype=np.int8),
            insert_labels=np.array(insert_labels_padded, dtype=np.int8),
            template_input_ids=template_input_ids.astype(np.int16),
            template_attention_mask=template_encodings['attention_mask'][j].astype(np.int8),
            infill_labels=infill_labels,
        )
    
    return results


def tokenize_batch_full_mask(
    samples: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
) -> List[Optional[TokenizedSample]]:
    """全 MASK 模式的批量 tokenize
    
    模板格式: [CLS] source [SEP] [MASK]*N [SEP]
    Labels: source 部分 -100，[MASK]*N 部分为 target token IDs
    """
    valid_indices = []
    sources = []
    targets = []
    sample_data = []
    
    for i, sample in enumerate(samples):
        source = sample.get('source', '')
        target = sample.get('target', '')
        if not source or len(source) > max_seq_length - 2:
            continue
        
        valid_indices.append(i)
        sources.append(source)
        targets.append(target)
        sample_data.append(sample)
    
    if not sources:
        return [None] * len(samples)
    
    # 批量 tokenize source
    source_encodings = tokenizer(
        sources,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    # 批量 tokenize target (不加特殊 token)
    target_encodings = tokenizer(
        targets,
        add_special_tokens=False,
        max_length=max_seq_length,
        padding=False,
        truncation=True,
        return_tensors='np'
    )
    
    results = [None] * len(samples)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    
    for j, orig_idx in enumerate(valid_indices):
        sample = sample_data[j]
        source = sources[j]
        op_labels = sample['op_labels']
        insert_labels = sample['insert_labels']
        
        # 处理 Planner 标签
        op_labels_padded = [-100] + op_labels[:max_seq_length-2] + [-100]
        insert_labels_padded = [0] + insert_labels[:max_seq_length-2] + [0]
        op_labels_padded = op_labels_padded + [-100] * (max_seq_length - len(op_labels_padded))
        insert_labels_padded = insert_labels_padded + [0] * (max_seq_length - len(insert_labels_padded))
        
        # 计算 target 长度
        target_length = compute_target_length(op_labels, insert_labels)
        
        # 获取 source tokens
        src_input_ids = source_encodings['input_ids'][j].tolist()
        source_tokens = [tid for tid in src_input_ids if tid not in [cls_id, sep_id, pad_id]]
        src_len = len(source)
        
        # 计算可用空间
        max_target_len = max_seq_length - src_len - 3
        actual_target_len = min(target_length, max(0, max_target_len))
        
        # 构建模板: [CLS] source [SEP] [MASK]*N [SEP]
        template_ids = [cls_id] + source_tokens[:src_len] + [sep_id]
        target_start_pos = len(template_ids)
        template_ids += [mask_id] * actual_target_len
        template_ids += [sep_id]
        
        total_len = len(template_ids)
        template_ids += [pad_id] * (max_seq_length - total_len)
        template_ids = template_ids[:max_seq_length]
        
        # attention mask
        template_mask = [1] * min(total_len, max_seq_length)
        template_mask += [0] * (max_seq_length - len(template_mask))
        template_mask = template_mask[:max_seq_length]
        
        # 构建 infill_labels
        infill_labels = np.full(max_seq_length, -100, dtype=np.int16)
        target_token_ids = target_encodings['input_ids'][j]
        if isinstance(target_token_ids, np.ndarray):
            target_token_ids = target_token_ids.tolist()
        
        for k, tid in enumerate(target_token_ids[:actual_target_len]):
            pos = target_start_pos + k
            if pos < max_seq_length:
                infill_labels[pos] = tid
        
        results[orig_idx] = TokenizedSample(
            input_ids=source_encodings['input_ids'][j].astype(np.int16),
            attention_mask=source_encodings['attention_mask'][j].astype(np.int8),
            op_labels=np.array(op_labels_padded, dtype=np.int8),
            insert_labels=np.array(insert_labels_padded, dtype=np.int8),
            template_input_ids=np.array(template_ids, dtype=np.int16),
            template_attention_mask=np.array(template_mask, dtype=np.int8),
            infill_labels=infill_labels,
        )
    
    return results


def compute_target_length(op_labels: List[int], insert_labels: List[int]) -> int:
    """计算 target 长度"""
    length = 0
    for i, op in enumerate(op_labels):
        if op == 0:  # KEEP
            length += 1
        elif op == 2:  # REPLACE
            length += 1
        if op != 1 and i < len(insert_labels):
            length += insert_labels[i]
    return length


def convert_jsonl_to_tokenized_streaming(
    input_file: str,
    output_prefix: str,
    tokenizer_name: str = "hfl/chinese-macbert-base",
    max_seq_length: int = 128,
    batch_size: int = 1000,
    shuffle: bool = True,
    seed: int = 42,
    full_mask_mode: bool = True,
):
    """
    流式转换 JSONL 文件为预计算 tokenize 的二进制格式
    
    优化点：
    1. 流式读取，不一次性加载全部数据到内存
    2. 批量 tokenize，利用 HuggingFace tokenizer 的并行能力
    3. 单进程避免进程间通信开销
    
    Args:
        input_file: 输入的 JSONL 文件
        output_prefix: 输出文件前缀 (会生成 .bin 和 .idx 文件)
        tokenizer_name: tokenizer 名称
        max_seq_length: 最大序列长度
        batch_size: 批处理大小（每批 tokenize 的样本数）
        shuffle: 是否打乱数据（需要先统计行数）
        seed: 随机种子
        full_mask_mode: 是否使用 full MASK 模式 (True: ReLM style, False: sparse MASK)
    """
    mode_str = "full MASK (ReLM style)" if full_mask_mode else "sparse MASK"
    print(f"Converting {input_file} to tokenized binary format (streaming mode)")
    print(f"  MASK mode: {mode_str}")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Batch size: {batch_size}")
    
    # 加载 tokenizer（只加载一次）
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 统计行数（用于进度条和打乱）
    print("Counting lines...")
    total_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"Total lines: {total_lines}")
    
    # 如果需要打乱，生成打乱的索引
    line_indices = list(range(total_lines))
    if shuffle:
        random.seed(seed)
        random.shuffle(line_indices)
        print("Shuffle indices generated")
    
    # 创建写入器
    writer = TokenizedDataWriter(output_prefix, max_seq_length)
    
    # 如果打乱，需要读取所有行的偏移（用于随机访问）
    if shuffle:
        print("Building line offset index for random access...")
        line_offsets = []
        with open(input_file, 'rb') as f:
            offset = 0
            for _ in tqdm(range(total_lines), desc="Indexing"):
                line_offsets.append(offset)
                line = f.readline()
                offset += len(line)
        print("Index built")
        
        # 按打乱顺序处理
        total_written = 0
        batch = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i in tqdm(line_indices, desc="Processing"):
                # 跳到指定行
                f.seek(line_offsets[i])
                line = f.readline().strip()
                
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    if 'source' in sample and 'op_labels' in sample:
                        batch.append(sample)
                except json.JSONDecodeError:
                    continue
                
                # 批量处理
                if len(batch) >= batch_size:
                    results = tokenize_batch(batch, tokenizer, max_seq_length, full_mask_mode)
                    for result in results:
                        if result is not None:
                            writer.write_sample(result)
                            total_written += 1
                    batch = []
        
        # 处理剩余批次
        if batch:
            results = tokenize_batch(batch, tokenizer, max_seq_length, full_mask_mode)
            for result in results:
                if result is not None:
                    writer.write_sample(result)
                    total_written += 1
    else:
        # 顺序处理（更快，不需要索引）
        total_written = 0
        batch = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Processing"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    if 'source' in sample and 'op_labels' in sample:
                        batch.append(sample)
                except json.JSONDecodeError:
                    continue
                
                # 批量处理
                if len(batch) >= batch_size:
                    results = tokenize_batch(batch, tokenizer, max_seq_length, full_mask_mode)
                    for result in results:
                        if result is not None:
                            writer.write_sample(result)
                            total_written += 1
                    batch = []
        
        # 处理剩余批次
        if batch:
            results = tokenize_batch(batch, tokenizer, max_seq_length, full_mask_mode)
            for result in results:
                if result is not None:
                    writer.write_sample(result)
                    total_written += 1
    
    writer.close()
    
    # 计算文件大小
    bin_size = os.path.getsize(f"{output_prefix}.bin")
    idx_size = os.path.getsize(f"{output_prefix}.idx")
    
    print(f"\nConversion complete!")
    print(f"  Total samples written: {total_written}")
    print(f"  Binary file: {output_prefix}.bin ({bin_size / 1024 / 1024:.2f} MB)")
    print(f"  Index file: {output_prefix}.idx ({idx_size / 1024:.2f} KB)")
    print(f"  Bytes per sample: {writer.sample_size}")
    
    return total_written


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成预计算 tokenize 的训练数据（流式处理版）")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入的 JSONL 文件（由 generate_training_data.py 生成）")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="输出文件前缀（会生成 .bin 和 .idx 文件）")
    parser.add_argument("--tokenizer", type=str, default="hfl/chinese-macbert-base",
                        help="Tokenizer 名称或路径")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="批处理大小（每批 tokenize 的样本数）")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="不打乱数据（顺序处理更快）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # MASK mode 参数
    mask_mode_group = parser.add_mutually_exclusive_group()
    mask_mode_group.add_argument("--full_mask_mode", action="store_true", default=True,
                                  help="使用 full MASK 模式（ReLM 风格，模板: [CLS] src [SEP] [MASK]*N [SEP]）（默认）")
    mask_mode_group.add_argument("--sparse_mask_mode", action="store_true",
                                  help="使用 sparse MASK 模式（只在编辑位置放 MASK）")
    
    args = parser.parse_args()
    
    # 确定 MASK 模式
    full_mask_mode = not args.sparse_mask_mode
    
    convert_jsonl_to_tokenized_streaming(
        input_file=args.input_file,
        output_prefix=args.output_prefix,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        full_mask_mode=full_mask_mode,
    )


if __name__ == "__main__":
    main()

