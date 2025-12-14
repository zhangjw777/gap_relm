"""
Gap-ReLM 推理脚本
"""

import os
import sys
import argparse
import json
from typing import List
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from inference import GapReLMPipeline, CorrectionResult


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Gap-ReLM Inference")
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--input", type=str, required=True,
                        help="Input file path or text")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    
    # 推理参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda, cpu)")
    
    # 可选功能
    parser.add_argument("--use_refinement", action="store_true",
                        help="Use iterative refinement")
    parser.add_argument("--refinement_rounds", type=int, default=2,
                        help="Number of refinement rounds")
    parser.add_argument("--use_verifier", action="store_true",
                        help="Use verifier")
    
    # 输出格式
    parser.add_argument("--output_format", type=str, default="json",
                        choices=["json", "txt", "parallel"],
                        help="Output format")
    parser.add_argument("--include_edits", action="store_true",
                        help="Include edit operations in output")
    parser.add_argument("--only_changed", action="store_true",
                        help="Only output changed sentences")
    
    # 输入格式
    parser.add_argument("--input_format", type=str, default="txt",
                        choices=["txt", "json"],
                        help="Input format")
    
    return parser.parse_args()


def load_input(input_path: str, input_format: str) -> List[str]:
    """加载输入数据"""
    texts = []
    
    if os.path.isfile(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            if input_format == "json":
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict):
                            text = item.get('text', item.get('source', item.get('input', '')))
                            if text:
                                texts.append(text)
            else:  # txt
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    else:
        # 直接输入文本
        texts = [input_path]
    
    return texts


def save_output(
    results: List[CorrectionResult],
    output_path: str,
    output_format: str,
    include_edits: bool,
    only_changed: bool
):
    """保存输出结果"""
    
    # 过滤
    if only_changed:
        results = [r for r in results if r.is_changed]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if output_format == "json":
            output_data = []
            for r in results:
                item = {
                    'source': r.source,
                    'prediction': r.prediction,
                    'is_changed': r.is_changed,
                    'confidence': r.confidence,
                }
                if include_edits:
                    item['edits'] = r.edits
                if r.verifier_accepted is not None:
                    item['verifier_accepted'] = r.verifier_accepted
                output_data.append(item)
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        elif output_format == "parallel":
            for r in results:
                f.write(f"{r.source}\t{r.prediction}\n")
        
        else:  # txt
            for r in results:
                f.write(f"{r.prediction}\n")


def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    pipeline = GapReLMPipeline(
        model_path=args.model_path,
        device=args.device
    )
    
    print(f"Loading input from {args.input}...")
    texts = load_input(args.input, args.input_format)
    print(f"Loaded {len(texts)} texts")
    
    # 更新推理配置
    pipeline.predictor.inference_config.use_iterative_refinement = args.use_refinement
    pipeline.predictor.inference_config.use_verifier = args.use_verifier
    if args.use_refinement:
        pipeline.predictor.ablation_config.refinement_rounds = args.refinement_rounds
    
    print("Running inference...")
    results = pipeline.predictor.predict_batch(
        texts,
        batch_size=args.batch_size,
        use_iterative_refinement=args.use_refinement,
        use_verifier=args.use_verifier,
        show_progress=True
    )
    
    # 统计
    changed_count = sum(1 for r in results if r.is_changed)
    print(f"Changed: {changed_count}/{len(results)} ({changed_count/len(results)*100:.1f}%)")
    
    # 输出
    if args.output:
        save_output(
            results,
            args.output,
            args.output_format,
            args.include_edits,
            args.only_changed
        )
        print(f"Results saved to {args.output}")
    else:
        # 打印到控制台
        for r in results:
            if r.is_changed:
                print(f"Source: {r.source}")
                print(f"Output: {r.prediction}")
                print(f"Confidence: {r.confidence:.4f}")
                print("-" * 40)


if __name__ == "__main__":
    main()
