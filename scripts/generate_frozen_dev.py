#!/usr/bin/env python
"""
生成 Frozen-Dev-Synth（固定验证集）

用于在线动态数据增强训练时的稳定评估。
每次评估使用相同的错误数据，确保训练曲线可比。

使用示例:
    # 基础用法：从干净句子生成固定验证集
    python scripts/generate_frozen_dev.py \
        --clean_file data/clean_dev.txt \
        --output_file data/frozen_dev.jsonl \
        --num_samples 20000

    # 指定造错参数
    python scripts/generate_frozen_dev.py \
        --clean_file data/clean_dev.txt \
        --output_file data/frozen_dev.jsonl \
        --p_corrupt 0.7 \
        --lambda_ 1.5 \
        --seed 42

    # 从 JSONL 文件读取
    python scripts/generate_frozen_dev.py \
        --clean_file data/corpus.jsonl \
        --file_format jsonl \
        --text_field sentence \
        --output_file data/frozen_dev.jsonl
"""

import sys
import os
import json
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gap_relm.data.augmentation import DataAugmentor, AugmentationConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Frozen-Dev-Synth for stable evaluation during online augmentation training"
    )
    
    # 输入输出
    parser.add_argument("--clean_file", type=str, required=True,
                        help="Path to clean sentences file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output frozen dev file")
    parser.add_argument("--file_format", type=str, default="txt",
                        choices=["txt", "json", "jsonl"],
                        help="Format of clean sentences file")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Text field name in JSON/JSONL file")
    
    # 数据量控制
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to generate (default: all)")
    parser.add_argument("--max_samples", type=int, default=100000,
                        help="Maximum number of samples to load from clean file")
    
    # 造错参数
    parser.add_argument("--p_corrupt", type=float, default=0.7,
                        help="Probability of corrupting a sentence")
    parser.add_argument("--lambda_", type=float, default=1.5,
                        help="Poisson lambda for number of edits")
    parser.add_argument("--pi_skip", type=float, default=0.2,
                        help="Probability of skip (delete) error type")
    parser.add_argument("--pi_multiply", type=float, default=0.3,
                        help="Probability of multiply (insert) error type")
    parser.add_argument("--pi_replace", type=float, default=0.5,
                        help="Probability of replace error type")
    parser.add_argument("--max_edits_per_sent", type=int, default=4,
                        help="Maximum edits per sentence")
    parser.add_argument("--max_insert_k", type=int, default=3,
                        help="Maximum characters to insert")
    
    # 保护约束
    parser.add_argument("--no_protection", action="store_true",
                        help="Disable protected span detection")
    
    # 随机种子
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # 输出格式
    parser.add_argument("--output_format", type=str, default="jsonl",
                        choices=["jsonl", "json", "tsv"],
                        help="Output file format")
    
    return parser.parse_args()


def load_clean_sentences(file_path: str, file_format: str, text_field: str, max_samples: int):
    """加载干净句子"""
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_format == "txt":
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
                    if max_samples and len(sentences) >= max_samples:
                        break
        
        elif file_format == "json":
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        sentences.append(item)
                    elif isinstance(item, dict) and text_field in item:
                        sentences.append(item[text_field])
                    if max_samples and len(sentences) >= max_samples:
                        break
        
        elif file_format == "jsonl":
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if isinstance(data, str):
                            sentences.append(data)
                        elif isinstance(data, dict):
                            # 支持多种字段名
                            text = (data.get(text_field) or 
                                   data.get('text') or 
                                   data.get('sentence') or 
                                   data.get('content') or
                                   data.get('target'))  # 如果是已有的训练数据
                            if text:
                                sentences.append(text)
                    except json.JSONDecodeError:
                        continue
                if max_samples and len(sentences) >= max_samples:
                    break
    
    return sentences


def save_results(results, output_file: str, output_format: str):
    """保存结果"""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if output_format == "jsonl":
            for r in results:
                line = json.dumps({
                    'source': r.corrupted,      # 错误句（模型输入）
                    'target': r.original,       # 正确句（模型目标）
                    'is_corrupted': r.is_corrupted,
                    'num_edits': len(r.edits),
                }, ensure_ascii=False)
                f.write(line + '\n')
        
        elif output_format == "json":
            data = [
                {
                    'source': r.corrupted,
                    'target': r.original,
                    'is_corrupted': r.is_corrupted,
                    'num_edits': len(r.edits),
                }
                for r in results
            ]
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif output_format == "tsv":
            for r in results:
                f.write(f"{r.corrupted}\t{r.original}\n")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Frozen-Dev-Synth Generator")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载干净句子
    print(f"\nLoading clean sentences from: {args.clean_file}")
    sentences = load_clean_sentences(
        args.clean_file,
        args.file_format,
        args.text_field,
        args.max_samples
    )
    print(f"Loaded {len(sentences)} sentences")
    
    # 如果指定了 num_samples，随机采样
    if args.num_samples and args.num_samples < len(sentences):
        sentences = random.sample(sentences, args.num_samples)
        print(f"Sampled {len(sentences)} sentences")
    
    # 创建造错配置
    config = AugmentationConfig(
        p_corrupt=args.p_corrupt,
        lambda_=args.lambda_,
        pi_skip=args.pi_skip,
        pi_multiply=args.pi_multiply,
        pi_replace=args.pi_replace,
        max_edits_per_sent=args.max_edits_per_sent,
        max_insert_k=args.max_insert_k,
        enable_protection=not args.no_protection,
        seed=args.seed,
    )
    
    # 创建数据增强器
    print("\nCreating augmentor with config:")
    print(f"  p_corrupt: {args.p_corrupt}")
    print(f"  lambda: {args.lambda_}")
    print(f"  pi_skip: {args.pi_skip}")
    print(f"  pi_multiply: {args.pi_multiply}")
    print(f"  pi_replace: {args.pi_replace}")
    print(f"  seed: {args.seed}")
    
    augmentor = DataAugmentor(config)
    
    # 批量造错
    print(f"\nGenerating errors for {len(sentences)} sentences...")
    results = augmentor.augment_batch(sentences, show_progress=True)
    
    # 统计信息
    stats = augmentor.get_stats(results)
    print(f"\nStatistics:")
    print(f"  Total samples: {stats['total_sentences']}")
    print(f"  Corrupted samples: {stats['corrupted_sentences']}")
    print(f"  Corruption rate: {stats['corruption_rate']:.2%}")
    print(f"  Avg edits per corrupted: {stats['avg_edits_per_corrupted']:.2f}")
    print(f"  Error type distribution: {stats['error_type_distribution']}")
    
    # 保存结果
    print(f"\nSaving to: {args.output_file}")
    save_results(results, args.output_file, args.output_format)
    
    # 保存配置和统计信息
    stats_file = args.output_file.rsplit('.', 1)[0] + '_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config.to_dict(),
            'stats': stats,
            'num_samples': len(results),
        }, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")
    
    print("\n" + "=" * 60)
    print("  ✅ Frozen-Dev-Synth generated successfully!")
    print("=" * 60)
    print(f"\nUse this file as --frozen_dev_file when training with --online_augment:")
    print(f"  python scripts/train.py \\")
    print(f"      --train_file data/clean_train.txt \\")
    print(f"      --frozen_dev_file {args.output_file} \\")
    print(f"      --online_augment")


if __name__ == "__main__":
    main()
