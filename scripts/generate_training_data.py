"""
生成静态训练数据的脚本
从干净句子生成：1个正例 + N个不同类型的负例（每个负例只有1个错误）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augmentation import StaticDataGenerator, StaticSampleConfig


def generate_static_training_data(
    clean_file: str = "data/clean_sentences.txt",
    output_dir: str = "./static_training_data",
    confusion_file: str = "data/confusion_sets/pycorrect_merged.json",
    num_negative: int = 2,
    pi_skip: float = 0.2,
    pi_multiply: float = 0.3,
    pi_replace: float = 0.5,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    生成静态训练数据
    
    Args:
        clean_file: 干净句子文件（每行一句）
        output_dir: 输出目录
        confusion_file: 混淆集文件
        num_negative: 每句生成的负例数量
        pi_skip: 删字概率
        pi_multiply: 重复字概率  
        pi_replace: 替换概率
        train_ratio/dev_ratio/test_ratio: 划分比例
        seed: 随机种子
    """
    print("=" * 60)
    print("生成静态训练数据")
    print("=" * 60)
    
    # 配置
    custom_files = [confusion_file] if os.path.exists(confusion_file) else []
    config = StaticSampleConfig(
        num_negative_per_sentence=num_negative,
        pi_skip=pi_skip,
        pi_multiply=pi_multiply,
        pi_replace=pi_replace,
        enable_protection=True,
        min_sentence_length=5,
        max_insert_k=3,
        seed=seed,
        custom_confusion_files=custom_files,
    )
    
    print(f"\n配置:")
    print(f"  干净句子: {clean_file}")
    print(f"  混淆集: {confusion_file}")
    print(f"  每句: 1正例 + {num_negative}负例")
    print(f"  错误类型概率: S={pi_skip:.1%}, M={pi_multiply:.1%}, R={pi_replace:.1%}")
    print(f"  划分: {train_ratio}:{dev_ratio}:{test_ratio}")
    
    # 生成
    generator = StaticDataGenerator(config=config)
    sentences = generator.load_sentences(clean_file)
    print(f"\n加载 {len(sentences)} 句子")
    
    output_files = generator.generate_and_save(
        clean_sentences=sentences,
        output_dir=output_dir,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    
    print(f"\n输出目录: {output_dir}")
    print("\n使用方法:")
    print(f"  python scripts/train.py --train_file {output_files['train']} --dev_file {output_files['dev']}")
    
    return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成静态训练数据")
    parser.add_argument("--clean_file", default="data/clean_sentences.txt", help="干净句子文件")
    parser.add_argument("--output_dir", default="./static_training_data", help="输出目录")
    parser.add_argument("--confusion_file", default="data/confusion_sets/pycorrect_merged.json")
    parser.add_argument("--num_negative", type=int, default=2, help="每句负例数")
    parser.add_argument("--pi_skip", type=float, default=0.2, help="删字概率")
    parser.add_argument("--pi_multiply", type=float, default=0.3, help="重复字概率")
    parser.add_argument("--pi_replace", type=float, default=0.5, help="替换概率")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    generate_static_training_data(**vars(args))
