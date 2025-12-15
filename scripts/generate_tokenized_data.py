"""
生成预计算 tokenize 的训练数据

关键优化：将 tokenize 阶段从训练时移到数据生成阶段
- 存储格式：二进制分片文件 (.bin) + 索引文件 (.idx)
- 训练时直接读取 tensor，无需任何 CPU 计算
- 显著减少 DataLoader 的 CPU 瓶颈，提升 GPU 利用率
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
from multiprocessing import Pool, cpu_count
from functools import partial
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


def tokenize_sample(
    sample: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
) -> Optional[TokenizedSample]:
    """
    将单个样本 tokenize 为二进制格式
    
    Args:
        sample: 包含 source, target, op_labels, insert_labels, template_tokens, gold_tokens 的字典
        tokenizer: HuggingFace tokenizer
        max_seq_length: 最大序列长度
    
    Returns:
        TokenizedSample 或 None（如果样本无效）
    """
    source = sample['source']
    op_labels = sample['op_labels']
    insert_labels = sample['insert_labels']
    template_tokens = sample.get('template_tokens', list(source))
    gold_tokens = sample.get('gold_tokens', [])
    
    # 检查长度
    if len(source) > max_seq_length - 2:
        return None
    
    # 1. 编码源序列
    source_encoding = tokenizer(
        source,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    input_ids = source_encoding['input_ids'].squeeze(0)
    attention_mask = source_encoding['attention_mask'].squeeze(0)
    
    # 2. 处理 Planner 标签
    # [CLS] + tokens + [SEP] + padding
    op_labels_padded = [-100] + op_labels[:max_seq_length-2] + [-100]
    insert_labels_padded = [0] + insert_labels[:max_seq_length-2] + [0]
    
    # Padding
    op_labels_padded = op_labels_padded + [-100] * (max_seq_length - len(op_labels_padded))
    insert_labels_padded = insert_labels_padded + [0] * (max_seq_length - len(insert_labels_padded))
    
    # 3. 构建模板输入
    template_str = ''.join(template_tokens).replace('[MASK]', tokenizer.mask_token)
    
    template_encoding = tokenizer(
        template_str,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    template_input_ids = template_encoding['input_ids'].squeeze(0)
    template_attention_mask = template_encoding['attention_mask'].squeeze(0)
    
    # 4. 构建 Infiller 标签
    mask_token_id = tokenizer.mask_token_id
    infill_labels = np.full(max_seq_length, -100, dtype=np.int16)
    
    mask_positions = np.where(template_input_ids == mask_token_id)[0]
    
    for i, pos in enumerate(mask_positions):
        if i < len(gold_tokens):
            token_id = tokenizer.convert_tokens_to_ids(gold_tokens[i])
            infill_labels[pos] = token_id
    
    return TokenizedSample(
        input_ids=input_ids.astype(np.int16),
        attention_mask=attention_mask.astype(np.int8),
        op_labels=np.array(op_labels_padded, dtype=np.int8),
        insert_labels=np.array(insert_labels_padded, dtype=np.int8),
        template_input_ids=template_input_ids.astype(np.int16),
        template_attention_mask=template_attention_mask.astype(np.int8),
        infill_labels=infill_labels,
    )


def process_chunk(
    chunk_data: Tuple[int, List[str]],
    tokenizer_name: str,
    max_seq_length: int,
) -> List[TokenizedSample]:
    """
    处理一个数据块（用于多进程）
    
    Args:
        chunk_data: (chunk_id, lines)
        tokenizer_name: tokenizer 名称
        max_seq_length: 最大序列长度
    
    Returns:
        TokenizedSample 列表
    """
    chunk_id, lines = chunk_data
    
    # 每个进程创建自己的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    results = []
    for line in lines:
        try:
            sample = json.loads(line.strip())
            tokenized = tokenize_sample(sample, tokenizer, max_seq_length)
            if tokenized is not None:
                results.append(tokenized)
        except (json.JSONDecodeError, KeyError) as e:
            continue
    
    return results


def convert_jsonl_to_tokenized(
    input_file: str,
    output_prefix: str,
    tokenizer_name: str = "hfl/chinese-macbert-base",
    max_seq_length: int = 128,
    num_workers: int = None,
    chunk_size: int = 10000,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    将 JSONL 文件转换为预计算 tokenize 的二进制格式
    
    Args:
        input_file: 输入的 JSONL 文件
        output_prefix: 输出文件前缀 (会生成 .bin 和 .idx 文件)
        tokenizer_name: tokenizer 名称
        max_seq_length: 最大序列长度
        num_workers: 并行处理的工作进程数
        chunk_size: 每个工作进程处理的样本数
        shuffle: 是否打乱数据
        seed: 随机种子
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    print(f"Converting {input_file} to tokenized binary format")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Workers: {num_workers}")
    print(f"  Chunk size: {chunk_size}")
    
    # 读取所有行
    print("Reading input file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"Total samples: {total_lines}")
    
    # 打乱数据
    if shuffle:
        random.seed(seed)
        random.shuffle(lines)
        print("Data shuffled")
    
    # 分块
    chunks = []
    for i in range(0, total_lines, chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        chunks.append((i // chunk_size, chunk_lines))
    
    print(f"Split into {len(chunks)} chunks")
    
    # 创建写入器
    writer = TokenizedDataWriter(output_prefix, max_seq_length)
    
    # 多进程处理
    process_fn = partial(
        process_chunk,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
    )
    
    total_written = 0
    
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for results in tqdm(
                pool.imap(process_fn, chunks),
                total=len(chunks),
                desc="Processing chunks"
            ):
                for sample in results:
                    writer.write_sample(sample)
                    total_written += 1
    else:
        # 单进程模式
        for chunk in tqdm(chunks, desc="Processing chunks"):
            results = process_fn(chunk)
            for sample in results:
                writer.write_sample(sample)
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
    
    parser = argparse.ArgumentParser(description="生成预计算 tokenize 的训练数据")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入的 JSONL 文件（由 generate_training_data.py 生成）")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="输出文件前缀（会生成 .bin 和 .idx 文件）")
    parser.add_argument("--tokenizer", type=str, default="hfl/chinese-macbert-base",
                        help="Tokenizer 名称或路径")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="并行工作进程数（默认: CPU核心数-2）")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="每个进程处理的样本数")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="不打乱数据")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    convert_jsonl_to_tokenized(
        input_file=args.input_file,
        output_prefix=args.output_prefix,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
