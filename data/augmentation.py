"""
数据增强管道
整合混淆集、造错器、保护约束，从干净句子生成训练数据
"""

import os
import json
import random
from typing import List, Tuple, Optional, Dict, Any, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .confusion_set import ConfusionSet, create_default_confusion_set
from .protected_span import ProtectedSpanDetector, create_default_detector
from .error_generator import ErrorGenerator, CorruptionResult


@dataclass
class AugmentationConfig:
    """数据增强配置
    
    四个主旋钮：
    - p_corrupt: 造错概率
    - lambda_: 编辑数量泊松参数
    - pi: 各类型概率分布
    - max_edits: 单句最大编辑数
    """
    # 造错概率
    p_corrupt: float = 0.7
    
    # 泊松参数
    lambda_: float = 1.5
    
    # 错误类型概率 (S: 删字, M: 重复字, R: 错字)
    pi_skip: float = 0.2
    pi_multiply: float = 0.3
    pi_replace: float = 0.5
    
    # 最大编辑数
    max_edits_per_sent: int = 4
    max_insert_k: int = 3
    
    # 混淆集配置
    use_default_shape_confusion: bool = True
    use_default_pinyin_confusion: bool = True
    custom_confusion_files: List[str] = field(default_factory=list)
    
    # 保护约束配置
    enable_protection: bool = True
    enable_doc_number_protection: bool = True
    enable_date_protection: bool = True
    enable_amount_protection: bool = True
    enable_clause_protection: bool = True
    enable_org_protection: bool = True
    enable_law_protection: bool = True
    enable_phrase_protection: bool = True
    custom_protected_words: List[str] = field(default_factory=list)
    
    # 其他选项
    min_sentence_length: int = 5
    skip_punct: bool = True
    seed: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AugmentationConfig":
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'p_corrupt': self.p_corrupt,
            'lambda_': self.lambda_,
            'pi_skip': self.pi_skip,
            'pi_multiply': self.pi_multiply,
            'pi_replace': self.pi_replace,
            'max_edits_per_sent': self.max_edits_per_sent,
            'max_insert_k': self.max_insert_k,
            'enable_protection': self.enable_protection,
            'min_sentence_length': self.min_sentence_length,
            'seed': self.seed,
        }


class DataAugmentor:
    """
    数据增强器
    
    从干净句子自动生成训练数据
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        初始化数据增强器
        
        Args:
            config: 增强配置，如果为 None 则使用默认配置
        """
        self.config = config or AugmentationConfig()
        
        # 设置随机种子
        if self.config.seed is not None:
            random.seed(self.config.seed)
        
        # 创建混淆集
        self.confusion_set = ConfusionSet(
            use_default_shape=self.config.use_default_shape_confusion,
            use_default_pinyin=self.config.use_default_pinyin_confusion,
            custom_confusion_files=self.config.custom_confusion_files or None,
        )
        
        # 创建保护检测器
        self.protected_detector = ProtectedSpanDetector(
            enable_doc_number=self.config.enable_doc_number_protection,
            enable_date=self.config.enable_date_protection,
            enable_amount=self.config.enable_amount_protection,
            enable_clause_number=self.config.enable_clause_protection,
            enable_organization=self.config.enable_org_protection,
            enable_law_name=self.config.enable_law_protection,
            enable_fixed_phrase=self.config.enable_phrase_protection,
            custom_protected_words=set(self.config.custom_protected_words) if self.config.custom_protected_words else None,
        )
        
        # 创建造错器
        self.error_generator = ErrorGenerator(
            p_corrupt=self.config.p_corrupt,
            lambda_=self.config.lambda_,
            pi_skip=self.config.pi_skip,
            pi_multiply=self.config.pi_multiply,
            pi_replace=self.config.pi_replace,
            max_edits_per_sent=self.config.max_edits_per_sent,
            max_insert_k=self.config.max_insert_k,
            confusion_set=self.confusion_set,
            protected_detector=self.protected_detector,
            enable_protection=self.config.enable_protection,
            min_sentence_length=self.config.min_sentence_length,
            skip_punct=self.config.skip_punct,
            seed=self.config.seed,
        )
    
    def augment(self, sentence: str) -> CorruptionResult:
        """
        对单个句子进行增强
        
        Args:
            sentence: 干净的正确句子
            
        Returns:
            CorruptionResult
        """
        return self.error_generator.corrupt(sentence)
    
    def augment_batch(
        self,
        sentences: List[str],
        show_progress: bool = False
    ) -> List[CorruptionResult]:
        """
        批量增强
        
        Args:
            sentences: 句子列表
            show_progress: 是否显示进度
            
        Returns:
            增强结果列表
        """
        return self.error_generator.corrupt_batch(sentences, show_progress)
    
    def generate_training_pairs(
        self,
        sentences: List[str],
        show_progress: bool = False
    ) -> List[Tuple[str, str]]:
        """
        从干净句子生成训练数据对
        
        Args:
            sentences: 干净句子列表
            show_progress: 是否显示进度
            
        Returns:
            训练数据对列表 [(错误句, 正确句), ...]
        """
        results = self.augment_batch(sentences, show_progress)
        return [r.to_training_pair() for r in results]
    
    def update_params(self, **kwargs):
        """
        更新造错参数（用于调参）
        
        Args:
            **kwargs: 参数名=值
        """
        self.error_generator.set_params(**kwargs)
    
    def get_stats(self, results: List[CorruptionResult]) -> Dict[str, Any]:
        """获取统计信息"""
        return self.error_generator.stats(results)


class TrainingDataGenerator:
    """
    训练数据生成器
    
    从干净语料生成 train/dev/test 数据集并保存
    """
    
    def __init__(
        self,
        augmentor: Optional[DataAugmentor] = None,
        config: Optional[AugmentationConfig] = None,
    ):
        """
        初始化生成器
        
        Args:
            augmentor: 数据增强器
            config: 增强配置（如果 augmentor 为 None）
        """
        self.augmentor = augmentor or DataAugmentor(config)
    
    def load_clean_sentences(
        self,
        file_path: str,
        file_format: str = "txt",
        text_field: str = "text",
    ) -> List[str]:
        """
        加载干净句子
        
        Args:
            file_path: 文件路径
            file_format: 文件格式 ("txt", "json", "jsonl")
            text_field: JSON中的文本字段名
            
        Returns:
            句子列表
        """
        sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_format == "txt":
                for line in f:
                    line = line.strip()
                    if line:
                        sentences.append(line)
            
            elif file_format == "json":
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            sentences.append(item)
                        elif isinstance(item, dict) and text_field in item:
                            sentences.append(item[text_field])
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            sentences.append(value)
                        elif isinstance(value, dict) and text_field in value:
                            sentences.append(value[text_field])
            
            elif file_format == "jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if isinstance(data, str):
                                sentences.append(data)
                            elif isinstance(data, dict) and text_field in data:
                                sentences.append(data[text_field])
                        except json.JSONDecodeError:
                            continue
        
        return sentences
    
    def generate_and_save(
        self,
        clean_sentences: List[str],
        output_dir: str,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        output_format: str = "jsonl",
        shuffle: bool = True,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, str]:
        """
        生成训练数据并保存
        
        Args:
            clean_sentences: 干净句子列表
            output_dir: 输出目录
            train_ratio: 训练集比例
            dev_ratio: 验证集比例
            test_ratio: 测试集比例
            output_format: 输出格式 ("jsonl", "tsv", "json")
            shuffle: 是否打乱数据
            seed: 随机种子
            show_progress: 是否显示进度
            
        Returns:
            输出文件路径字典
        """
        # 验证比例
        total_ratio = train_ratio + dev_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            train_ratio /= total_ratio
            dev_ratio /= total_ratio
            test_ratio /= total_ratio
        
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
        
        # 打乱数据
        if shuffle:
            sentences = clean_sentences.copy()
            random.shuffle(sentences)
        else:
            sentences = clean_sentences
        
        # 划分数据集
        n = len(sentences)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        
        train_sentences = sentences[:n_train]
        dev_sentences = sentences[n_train:n_train + n_dev]
        test_sentences = sentences[n_train + n_dev:]
        
        print(f"Data split: train={len(train_sentences)}, dev={len(dev_sentences)}, test={len(test_sentences)}")
        
        # 生成各数据集
        print("Generating training data...")
        train_results = self.augmentor.augment_batch(train_sentences, show_progress)
        
        print("Generating dev data...")
        dev_results = self.augmentor.augment_batch(dev_sentences, show_progress)
        
        print("Generating test data...")
        test_results = self.augmentor.augment_batch(test_sentences, show_progress)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        output_files = {}
        
        output_files['train'] = self._save_results(
            train_results, os.path.join(output_dir, f"train.{output_format}"), output_format
        )
        output_files['dev'] = self._save_results(
            dev_results, os.path.join(output_dir, f"dev.{output_format}"), output_format
        )
        output_files['test'] = self._save_results(
            test_results, os.path.join(output_dir, f"test.{output_format}"), output_format
        )
        
        # 保存统计信息
        stats = {
            'train': self.augmentor.get_stats(train_results),
            'dev': self.augmentor.get_stats(dev_results),
            'test': self.augmentor.get_stats(test_results),
            'config': self.augmentor.config.to_dict(),
        }
        
        stats_file = os.path.join(output_dir, "stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        output_files['stats'] = stats_file
        
        print(f"\nGeneration complete!")
        print(f"  Train corruption rate: {stats['train']['corruption_rate']:.2%}")
        print(f"  Dev corruption rate: {stats['dev']['corruption_rate']:.2%}")
        print(f"  Test corruption rate: {stats['test']['corruption_rate']:.2%}")
        
        return output_files
    
    def _save_results(
        self,
        results: List[CorruptionResult],
        file_path: str,
        output_format: str
    ) -> str:
        """保存结果到文件"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if output_format == "jsonl":
                for r in results:
                    line = json.dumps({
                        'source': r.corrupted,      # 错误句（模型输入）
                        'target': r.original,       # 正确句（模型目标）
                        'is_corrupted': r.is_corrupted,
                        'num_edits': len(r.edits),
                    }, ensure_ascii=False)
                    f.write(line + '\n')
            
            elif output_format == "tsv":
                for r in results:
                    f.write(f"{r.corrupted}\t{r.original}\n")
            
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
        
        print(f"Saved {len(results)} samples to {file_path}")
        return file_path


# 便捷函数
def create_augmentor(
    p_corrupt: float = 0.7,
    lambda_: float = 1.5,
    pi_skip: float = 0.2,
    pi_multiply: float = 0.3,
    pi_replace: float = 0.5,
    **kwargs
) -> DataAugmentor:
    """创建数据增强器"""
    config = AugmentationConfig(
        p_corrupt=p_corrupt,
        lambda_=lambda_,
        pi_skip=pi_skip,
        pi_multiply=pi_multiply,
        pi_replace=pi_replace,
        **kwargs
    )
    return DataAugmentor(config)


def generate_training_data(
    clean_file: str,
    output_dir: str,
    p_corrupt: float = 0.7,
    lambda_: float = 1.5,
    pi_skip: float = 0.2,
    pi_multiply: float = 0.3,
    pi_replace: float = 0.5,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    file_format: str = "txt",
    output_format: str = "jsonl",
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, str]:
    """
    一键生成训练数据
    
    Args:
        clean_file: 干净语料文件路径
        output_dir: 输出目录
        p_corrupt: 造错概率
        lambda_: 泊松参数
        pi_skip: 删字概率
        pi_multiply: 重复字概率
        pi_replace: 错字概率
        train_ratio: 训练集比例
        dev_ratio: 验证集比例
        test_ratio: 测试集比例
        file_format: 输入文件格式
        output_format: 输出文件格式
        seed: 随机种子
        **kwargs: 其他配置参数
        
    Returns:
        输出文件路径字典
    """
    config = AugmentationConfig(
        p_corrupt=p_corrupt,
        lambda_=lambda_,
        pi_skip=pi_skip,
        pi_multiply=pi_multiply,
        pi_replace=pi_replace,
        seed=seed,
        **kwargs
    )
    
    augmentor = DataAugmentor(config)
    generator = TrainingDataGenerator(augmentor)
    
    # 加载干净句子
    sentences = generator.load_clean_sentences(clean_file, file_format)
    print(f"Loaded {len(sentences)} clean sentences from {clean_file}")
    
    # 生成并保存
    return generator.generate_and_save(
        clean_sentences=sentences,
        output_dir=output_dir,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        output_format=output_format,
        seed=seed,
    )


def grid_search_params(
    clean_sentences: List[str],
    eval_sentences: List[str],
    eval_fn,
    p_corrupt_range: List[float] = [0.3, 0.5, 0.7],
    lambda_range: List[float] = [1.0, 1.5, 2.0],
    pi_multiply_range: List[float] = [0.3, 0.5, 0.7],
    max_combinations: int = 27,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """
    网格搜索最佳参数
    
    Args:
        clean_sentences: 用于训练的干净句子
        eval_sentences: 用于评估的句子
        eval_fn: 评估函数，接受 (train_data, eval_data) 返回 score
        p_corrupt_range: 造错概率搜索范围
        lambda_range: 泊松参数搜索范围
        pi_multiply_range: 重复字概率搜索范围
        max_combinations: 最大组合数
        seed: 随机种子
        
    Returns:
        排序后的参数和得分列表
    """
    from itertools import product
    
    results = []
    combinations = list(product(p_corrupt_range, lambda_range, pi_multiply_range))
    
    if len(combinations) > max_combinations:
        combinations = random.sample(combinations, max_combinations)
    
    for p_corrupt, lambda_, pi_multiply in combinations:
        pi_skip = (1 - pi_multiply) / 2
        pi_replace = (1 - pi_multiply) / 2
        
        config = AugmentationConfig(
            p_corrupt=p_corrupt,
            lambda_=lambda_,
            pi_skip=pi_skip,
            pi_multiply=pi_multiply,
            pi_replace=pi_replace,
            seed=seed,
        )
        
        augmentor = DataAugmentor(config)
        train_data = augmentor.generate_training_pairs(clean_sentences)
        
        score = eval_fn(train_data, eval_sentences)
        
        results.append({
            'params': {
                'p_corrupt': p_corrupt,
                'lambda_': lambda_,
                'pi_skip': pi_skip,
                'pi_multiply': pi_multiply,
                'pi_replace': pi_replace,
            },
            'score': score,
        })
    
    # 按得分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results
