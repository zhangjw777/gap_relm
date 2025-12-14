"""
将 pycorrect 的混淆集格式转换为 Gap-ReLM 兼容格式

pycorrect 格式:
    #汉字	同音同调	同音异调
    一	壹	漪
    七	戚漆栖凄溪欺柒妻	泣迄畦稽脐只鳍气奇砌契企祈骑枝启旗歧起器乞棋弃汽齐其崎岂期

Gap-ReLM 格式:
    JSON: {"一": ["壹", "漪"], "七": ["戚", "漆", ...]}
    或 TSV: 一\t壹\t漪
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def parse_pycorrect_line(line: str) -> tuple[str, List[str]]:
    """
    解析 pycorrect 的一行数据
    
    Args:
        line: "一	壹	漪"
        
    Returns:
        (原字符, [混淆字列表])
    """
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, []
    
    char = parts[0].strip()
    confusions = []
    
    # 遍历每个字段（跳过第一个原字符）
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        # 将连续字符串拆分为单个字符
        for c in part:
            if c and c != char:  # 排除原字符本身
                confusions.append(c)
    
    return char, confusions


def convert_pycorrect_file(
    input_file: str,
    output_format: str = "json",
    confusion_type: str = "custom"
) -> Dict[str, List[str]]:
    """
    转换单个 pycorrect 文件
    
    Args:
        input_file: pycorrect 混淆集文件路径
        output_format: 输出格式 (json/tsv/jsonl)
        confusion_type: 混淆类型标签
        
    Returns:
        混淆字典
    """
    confusion_dict = defaultdict(set)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            char, confusions = parse_pycorrect_line(line)
            if char and confusions:
                confusion_dict[char].update(confusions)
    
    # 转换为列表格式
    result = {char: list(confs) for char, confs in confusion_dict.items()}
    
    print(f"✅ 从 {input_file} 加载了 {len(result)} 个字符的混淆集")
    total_pairs = sum(len(confs) for confs in result.values())
    print(f"   总共 {total_pairs} 个混淆对")
    
    return result


def merge_confusion_dicts(
    dict1: Dict[str, List[str]],
    dict2: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    合并两个混淆字典
    """
    merged = defaultdict(set)
    
    for char, confs in dict1.items():
        merged[char].update(confs)
    
    for char, confs in dict2.items():
        merged[char].update(confs)
    
    return {char: list(confs) for char, confs in merged.items()}


def save_as_json(confusion_dict: Dict[str, List[str]], output_file: str):
    """保存为 JSON 格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(confusion_dict, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存为 JSON: {output_file}")


def save_as_tsv(confusion_dict: Dict[str, List[str]], output_file: str):
    """保存为 TSV 格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 原字符\t混淆字1\t混淆字2\t...\n")
        for char, confusions in sorted(confusion_dict.items()):
            line = char + '\t' + '\t'.join(confusions) + '\n'
            f.write(line)
    print(f"✅ 已保存为 TSV: {output_file}")


def save_as_jsonl(confusion_dict: Dict[str, List[str]], output_file: str, confusion_type: str = "custom"):
    """保存为 JSON Lines 格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for char, confusions in sorted(confusion_dict.items()):
            entry = {
                'char': char,
                'confusions': confusions,
                'type': confusion_type
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"✅ 已保存为 JSONL: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="转换 pycorrect 混淆集为 Gap-ReLM 格式"
    )
    parser.add_argument(
        '--pinyin_file',
        type=str,
        default='./same_pinyin.txt',
        help='pycorrect 的 same_pinyin.txt 文件路径'
    )
    parser.add_argument(
        '--stroke_file',
        type=str,
        default='./same_stroke.txt',
        help='pycorrect 的 same_stroke.txt 文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./confusion_sets',
        help='输出目录'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='json',
        choices=['json', 'tsv', 'jsonl'],
        help='输出格式'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='是否合并音近字和形近字到一个文件'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("pycorrect 混淆集转换工具")
    print("="*60 + "\n")
    
    # 转换音近字
    pinyin_dict = {}
    if Path(args.pinyin_file).exists():
        print(f"📖 读取音近字文件: {args.pinyin_file}")
        pinyin_dict = convert_pycorrect_file(args.pinyin_file, args.format, "pinyin")
    else:
        print(f"⚠️  音近字文件不存在: {args.pinyin_file}")
    
    # 转换形近字
    stroke_dict = {}
    if Path(args.stroke_file).exists():
        print(f"\n📖 读取形近字文件: {args.stroke_file}")
        stroke_dict = convert_pycorrect_file(args.stroke_file, args.format, "shape")
    else:
        print(f"⚠️  形近字文件不存在: {args.stroke_file}")
    
    if not pinyin_dict and not stroke_dict:
        print("\n❌ 没有找到任何输入文件，请检查路径")
        return
    
    # 保存文件
    print("\n" + "="*60)
    print("保存转换结果")
    print("="*60 + "\n")
    
    if args.merge:
        # 合并保存
        merged_dict = merge_confusion_dicts(pinyin_dict, stroke_dict)
        output_file = output_dir / f"pycorrect_merged.{args.format}"
        
        if args.format == 'json':
            save_as_json(merged_dict, str(output_file))
        elif args.format == 'tsv':
            save_as_tsv(merged_dict, str(output_file))
        elif args.format == 'jsonl':
            save_as_jsonl(merged_dict, str(output_file), "pycorrect")
    else:
        # 分别保存
        if pinyin_dict:
            output_file = output_dir / f"pycorrect_pinyin.{args.format}"
            if args.format == 'json':
                save_as_json(pinyin_dict, str(output_file))
            elif args.format == 'tsv':
                save_as_tsv(pinyin_dict, str(output_file))
            elif args.format == 'jsonl':
                save_as_jsonl(pinyin_dict, str(output_file), "pinyin")
        
        if stroke_dict:
            output_file = output_dir / f"pycorrect_stroke.{args.format}"
            if args.format == 'json':
                save_as_json(stroke_dict, str(output_file))
            elif args.format == 'tsv':
                save_as_tsv(stroke_dict, str(output_file))
            elif args.format == 'jsonl':
                save_as_jsonl(stroke_dict, str(output_file), "shape")
    
    print("\n" + "="*60)
    print("✅ 转换完成！")
    print("="*60)
    print("\n【使用方式】")
    print("在 AugmentationConfig 中添加:")
    print(f"  custom_confusion_files=['{output_dir}/pycorrect_*.{args.format}']")
    print("\n或者在代码中:")
    print("  from gap_relm.data import ConfusionSet")
    print(f"  cs = ConfusionSet(custom_confusion_files=['{output_dir}/pycorrect_merged.{args.format}'])")


if __name__ == '__main__':
    # 示例用法
    print("\n" + "="*60)
    print("使用示例")
    print("="*60)
    print("\n1. 基本用法:")
    print("   python scripts/convert_pycorrect_confusion.py \\")
    print("     --pinyin_file ./same_pinyin.txt \\")
    print("     --stroke_file ./same_stroke.txt \\")
    print("     --output_dir ./confusion_sets")
    print("\n2. 合并为一个文件:")
    print("   python scripts/convert_pycorrect_confusion.py \\")
    print("     --pinyin_file ./same_pinyin.txt \\")
    print("     --stroke_file ./same_stroke.txt \\")
    print("     --merge")
    print("\n3. 输出为 TSV 格式:")
    print("   python scripts/convert_pycorrect_confusion.py \\")
    print("     --pinyin_file ./same_pinyin.txt \\")
    print("     --format tsv")
    print("\n" + "="*60 + "\n")
    
    main()
