"""
从干净句子生成训练数据的完整脚本
适合新手使用，包含详细注释和示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augmentation import DataAugmentor, AugmentationConfig, TrainingDataGenerator


def example_1_basic_usage():
    """
    示例1: 基础使用 - 对单个句子造错
    """
    print("=" * 60)
    print("示例1: 对单个句子造错")
    print("=" * 60)
    
    # 1. 配置造错参数
    config = AugmentationConfig(
        p_corrupt=0.7,          # 70%的句子会被造错
        lambda_=1.5,            # 平均每句造1.5个错误
        pi_skip=0.2,            # 20%概率删字
        pi_multiply=0.3,        # 30%概率重复字
        pi_replace=0.5,         # 50%概率错字
        max_edits_per_sent=4,   # 每句最多4个错误
        max_insert_k=3,         # 单个位置最多重复3次
        seed=42                 # 固定随机种子（可复现）
    )
    
    # 2. 创建数据增强器
    augmentor = DataAugmentor(config)
    
    # 3. 对单个句子造错
    clean_sentence = "这是一个正确的中文句子，没有任何错误。"
    
    result = augmentor.augment(clean_sentence)
    
    # 4. 查看结果
    print(f"\n原始句子: {result.original}")
    print(f"错误句子: {result.corrupted}")
    print(f"是否造错: {result.is_corrupted}")
    print(f"错误数量: {len(result.edits)}")
    
    if result.edits:
        print("\n错误详情:")
        for edit in result.edits:
            print(f"  - {edit}")
    
    # 5. 转换为训练数据对格式
    error_sent, correct_sent = result.to_training_pair()
    print(f"\n训练数据对:")
    print(f"  输入(错误句): {error_sent}")
    print(f"  目标(正确句): {correct_sent}")


def example_2_batch_generation():
    """
    示例2: 批量生成训练数据
    """
    print("\n\n" + "=" * 60)
    print("示例2: 批量生成训练数据")
    print("=" * 60)
    
    # 准备干净句子列表
    clean_sentences = [
        "中华人民共和国国务院办公厅关于加强基层治理体系和治理能力现代化建设的意见。",
        "根据中华人民共和国宪法第六十七条第二款的规定，现予以公布。",
        "为进一步推动经济社会高质量发展，特制定本实施方案。",
        "各级人民政府应当加强组织领导，明确责任分工，确保各项措施落到实处。",
        "本通知自发布之日起施行，有效期为五年。",
    ]
    
    # 创建数据增强器
    config = AugmentationConfig(p_corrupt=0.8, lambda_=2.0, seed=42)
    augmentor = DataAugmentor(config)
    
    # 批量生成
    results = augmentor.augment_batch(clean_sentences, show_progress=True)
    
    # 查看结果
    print(f"\n生成了 {len(results)} 个样本:")
    for i, result in enumerate(results):
        if result.is_corrupted:
            print(f"\n样本 {i+1}:")
            print(f"  原始: {result.original[:40]}...")
            print(f"  错误: {result.corrupted[:40]}...")
            print(f"  错误数: {len(result.edits)}")
    
    # 统计信息
    stats = augmentor.get_stats(results)
    print(f"\n统计信息:")
    print(f"  造错率: {stats['corruption_rate']:.2%}")
    print(f"  平均错误数: {stats['avg_edits_per_sent']:.2f}")


def example_3_generate_full_dataset():
    """
    示例3: 生成完整的训练数据集
    这是最实用的方法！
    """
    print("\n\n" + "=" * 60)
    print("示例3: 生成完整的train/dev/test数据集")
    print("=" * 60)
    
    # 1. 准备干净句子（实际使用时从文件加载）
    # 这里只是示例，实际应该有几千到几万句
    clean_sentences = [
        f"这是第{i}个示例句子，用于演示如何生成训练数据。" 
        for i in range(1, 101)  # 生成100个示例句子
    ]
    
    # 2. 配置参数
    config = AugmentationConfig(
        p_corrupt=0.7,
        lambda_=1.5,
        pi_skip=0.2,
        pi_multiply=0.3,
        pi_replace=0.5,
        max_edits_per_sent=4,
        seed=42
    )
    
    # 3. 创建生成器
    generator = TrainingDataGenerator(config=config)
    
    # 4. 生成并保存数据集
    output_dir = "./generated_data"
    output_files = generator.generate_and_save(
        clean_sentences=clean_sentences,
        output_dir=output_dir,
        train_ratio=0.8,      # 80% 训练集
        dev_ratio=0.1,        # 10% 验证集
        test_ratio=0.1,       # 10% 测试集
        output_format="jsonl", # 输出格式：jsonl
        shuffle=True,
        seed=42,
        show_progress=True
    )
    
    print(f"\n数据已保存到: {output_dir}")
    print(f"  训练集: {output_files['train']}")
    print(f"  验证集: {output_files['dev']}")
    print(f"  测试集: {output_files['test']}")
    print(f"  统计信息: {output_files['stats']}")


def example_4_from_file():
    """
    示例4: 从文件读取clean句子并生成数据
    这是最常用的生产场景！
    """
    print("\n\n" + "=" * 60)
    print("示例4: 从文件读取并生成数据（实际使用）")
    print("=" * 60)
    
    # 首先创建一个示例clean文件
    clean_file = "./data/clean_sentences.txt"
    os.makedirs("./data", exist_ok=True)
    
    with open(clean_file, 'w', encoding='utf-8') as f:
        for i in range(1, 21):
            f.write(f"示例句子{i}：中华人民共和国国务院办公厅发布关于规范性文件的通知。\n")
    
    print(f"已创建示例文件: {clean_file}")
    
    # 创建生成器
    config = AugmentationConfig(p_corrupt=0.7, lambda_=1.5, seed=42)
    generator = TrainingDataGenerator(config=config)
    
    # 从文件加载clean句子
    sentences = generator.load_clean_sentences(
        file_path=clean_file,
        file_format="txt"  # 支持: "txt", "json", "jsonl"
    )
    
    print(f"从文件加载了 {len(sentences)} 个句子")
    
    # 生成数据集
    output_files = generator.generate_and_save(
        clean_sentences=sentences,
        output_dir="./generated_data_from_file",
        output_format="jsonl",
        show_progress=True
    )
    
    print(f"\n数据生成完成！可以直接用于训练:")
    print(f"  python scripts/train.py \\")
    print(f"    --train_file {output_files['train']} \\")
    print(f"    --dev_file {output_files['dev']} \\")
    print(f"    --data_format mucgec")  # 生成的jsonl格式可以用mucgec加载器读取


def example_5_custom_config():
    """
    示例5: 自定义配置 - 调整造错参数
    """
    print("\n\n" + "=" * 60)
    print("示例5: 自定义配置 - 针对特定错误类型")
    print("=" * 60)
    
    # 配置1: 只生成删字错误
    config_delete_only = AugmentationConfig(
        p_corrupt=0.8,
        lambda_=1.0,
        pi_skip=1.0,       # 100%删字
        pi_multiply=0.0,   # 0%重复字
        pi_replace=0.0,    # 0%错字
        seed=42
    )
    
    augmentor_delete = DataAugmentor(config_delete_only)
    
    sentence = "这是一个测试句子，用于演示删字错误。"
    result = augmentor_delete.augment(sentence)
    
    print(f"\n只造删字错误:")
    print(f"  原始: {result.original}")
    print(f"  错误: {result.corrupted}")
    
    # 配置2: 高错误率、多错误
    config_high_error = AugmentationConfig(
        p_corrupt=1.0,         # 100%造错
        lambda_=3.0,           # 平均3个错误
        max_edits_per_sent=6,  # 最多6个错误
        seed=42
    )
    
    augmentor_high = DataAugmentor(config_high_error)
    result = augmentor_high.augment(sentence)
    
    print(f"\n高错误率配置:")
    print(f"  原始: {result.original}")
    print(f"  错误: {result.corrupted}")
    print(f"  错误数: {len(result.edits)}")


def main():
    """
    主函数 - 运行所有示例
    """
    print("\n" + "="*60)
    print("Gap-ReLM 数据生成完整教程")
    print("从干净句子生成训练数据")
    print("="*60)
    
    # 运行示例
    example_1_basic_usage()
    example_2_batch_generation()
    example_3_generate_full_dataset()
    example_4_from_file()
    example_5_custom_config()
    
    print("\n\n" + "="*60)
    print("教程完成！")
    print("="*60)
    print("\n【实际使用建议】")
    print("1. 准备大量干净的公文语料（txt文件，每行一句）")
    print("2. 使用 example_4 的方式从文件读取并生成数据")
    print("3. 调整 AugmentationConfig 参数来控制错误类型和数量")
    print("4. 生成的 jsonl 文件可以直接用于训练")
    print("5. 配合标注数据一起使用效果更好！")
    
    print("\n【下一步】")
    print("运行训练:")
    print("  python scripts/train.py \\")
    print("    --train_file ./generated_data/train.jsonl \\")
    print("    --dev_file ./generated_data/dev.jsonl \\")
    print("    --data_format mucgec \\")
    print("    --output_dir ./outputs")


if __name__ == "__main__":
    main()
