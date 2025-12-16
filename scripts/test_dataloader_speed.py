#!/usr/bin/env python
"""
数据加载器速度诊断脚本
用于测试不同 num_workers 配置下的数据加载性能

使用方法:
    python scripts/test_dataloader_speed.py --data_prefix ./tokenized_data/train --num_workers 0
    python scripts/test_dataloader_speed.py --data_prefix ./tokenized_data/train --num_workers 4
    python scripts/test_dataloader_speed.py --data_prefix ./tokenized_data/train --num_workers 8
"""

import os
import sys
import time
import argparse

# 设置 multiprocessing 启动方法（必须在其他 import 之前）
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
    print("✓ Set multiprocessing start method to 'spawn'")
except RuntimeError as e:
    print(f"Note: {e}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import TokenizedBinaryDataset, GapReLMCollator


def benchmark_dataloader(
    data_prefix: str,
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    num_batches: int = 100,
    warmup_batches: int = 10,
):
    """测试数据加载器性能"""
    
    print(f"\n{'='*60}")
    print(f"Testing DataLoader with num_workers={num_workers}")
    print(f"{'='*60}")
    
    # 创建数据集
    print("Creating dataset...")
    start = time.time()
    dataset = TokenizedBinaryDataset(
        data_prefix=data_prefix,
        enable_aux_mlm=True,
        aux_mlm_prob=0.15,
    )
    print(f"Dataset created in {time.time() - start:.2f}s")
    print(f"Total samples: {len(dataset):,}")
    
    # 创建 tokenizer 和 collator
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
    collator = GapReLMCollator(tokenizer, include_aux_mlm=True)
    
    # 创建 DataLoader
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'collate_fn': collator,
        'pin_memory': True,
        'drop_last': True,
    }
    
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = True
    
    print("Creating DataLoader...")
    loader = DataLoader(dataset, **loader_kwargs)
    
    # Warmup
    print(f"Warming up ({warmup_batches} batches)...")
    warmup_start = time.time()
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break
        # 模拟 GPU 传输
        batch['input_ids'].cuda()
        batch['attention_mask'].cuda()
    warmup_time = time.time() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f}s ({warmup_batches / warmup_time:.1f} batches/s)")
    
    # Benchmark
    print(f"Benchmarking ({num_batches} batches)...")
    times = []
    batch_times = []
    
    data_iter = iter(loader)
    for i in range(num_batches):
        batch_start = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        # 模拟 GPU 传输
        batch['input_ids'].cuda()
        batch['attention_mask'].cuda()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if (i + 1) % 20 == 0:
            avg_time = sum(batch_times[-20:]) / 20
            print(f"  Batch {i+1}/{num_batches}: {avg_time*1000:.2f} ms/batch")
    
    # 统计
    total_time = sum(batch_times)
    avg_time = total_time / num_batches
    throughput = batch_size * num_batches / total_time
    
    print(f"\n{'='*60}")
    print(f"Results for num_workers={num_workers}:")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average batch time: {avg_time*1000:.2f} ms")
    print(f"Min batch time: {min(batch_times)*1000:.2f} ms")
    print(f"Max batch time: {max(batch_times)*1000:.2f} ms")
    print(f"Throughput: {throughput:.0f} samples/s")
    print(f"Effective batch rate: {num_batches / total_time:.1f} batches/s")
    
    return {
        'num_workers': num_workers,
        'avg_time': avg_time,
        'throughput': throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="DataLoader speed benchmark")
    parser.add_argument("--data_prefix", type=str, required=True,
                        help="Data file prefix (without .bin/.idx)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers (if not set, test 0, 4, 8, 16)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="Prefetch factor")
    parser.add_argument("--num_batches", type=int, default=100,
                        help="Number of batches to benchmark")
    
    args = parser.parse_args()
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU")
    else:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    if args.num_workers is not None:
        # 测试单个配置
        benchmark_dataloader(
            data_prefix=args.data_prefix,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            num_batches=args.num_batches,
        )
    else:
        # 测试多个配置
        results = []
        for num_workers in [0, 4, 8, 16]:
            try:
                result = benchmark_dataloader(
                    data_prefix=args.data_prefix,
                    batch_size=args.batch_size,
                    num_workers=num_workers,
                    prefetch_factor=args.prefetch_factor,
                    num_batches=args.num_batches,
                )
                results.append(result)
            except Exception as e:
                print(f"Error with num_workers={num_workers}: {e}")
                results.append({
                    'num_workers': num_workers,
                    'avg_time': float('inf'),
                    'throughput': 0,
                    'error': str(e),
                })
        
        # 总结
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        print(f"{'num_workers':<15} {'avg_time(ms)':<15} {'throughput(s/s)':<15}")
        for r in results:
            if 'error' in r:
                print(f"{r['num_workers']:<15} ERROR: {r['error']}")
            else:
                print(f"{r['num_workers']:<15} {r['avg_time']*1000:<15.2f} {r['throughput']:<15.0f}")
        
        # 推荐
        best = min([r for r in results if 'error' not in r], key=lambda x: x['avg_time'])
        print(f"\n推荐配置: num_workers={best['num_workers']}")


if __name__ == "__main__":
    main()
