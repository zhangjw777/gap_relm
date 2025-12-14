"""
数据加载器工厂函数

支持两种数据模式：
1. 静态模式：使用预生成的训练数据
2. 在线动态增强模式：从干净句子实时生成错误
"""

import os
from typing import Optional, Tuple, List, Dict, Any
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from .dataset import GapReLMDataset, GapReLMCollator, OnlineAugmentedDataset, load_clean_sentences


def create_data_loaders(
    train_file: str,
    dev_file: Optional[str] = None,
    test_file: Optional[str] = None,
    tokenizer_name: str = "hfl/chinese-macbert-base",
    max_seq_length: int = 128,
    max_insert_num: int = 3,
    enable_insert: bool = True,
    enable_delete: bool = True,
    alignment_algorithm: str = "levenshtein",
    data_format: str = "mucgec",
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    normalize_text: bool = True,
    enable_aux_mlm: bool = True,
    aux_mlm_prob: float = 0.15,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], AutoTokenizer]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        train_file: 训练数据文件路径
        dev_file: 验证数据文件路径（可选）
        test_file: 测试数据文件路径（可选）
        tokenizer_name: tokenizer 名称或路径
        max_seq_length: 最大序列长度
        max_insert_num: 最大插入数量
        enable_insert: 是否启用插入
        enable_delete: 是否启用删除
        alignment_algorithm: 对齐算法
        data_format: 数据格式
        batch_size: 批大小
        num_workers: 数据加载工作进程数
        cache_dir: 缓存目录
        use_cache: 是否使用缓存
        normalize_text: 是否规范化文本
        enable_aux_mlm: 是否启用辅助MLM
        aux_mlm_prob: 辅助MLM mask比例
        distributed: 是否分布式训练
        world_size: 分布式世界大小
        rank: 当前进程排名
        
    Returns:
        (train_loader, dev_loader, test_loader, tokenizer)
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 创建 collator
    collator = GapReLMCollator(tokenizer, include_aux_mlm=enable_aux_mlm)
    
    # 创建数据集的通用参数
    dataset_kwargs = {
        'tokenizer': tokenizer,
        'max_seq_length': max_seq_length,
        'max_insert_num': max_insert_num,
        'enable_insert': enable_insert,
        'enable_delete': enable_delete,
        'alignment_algorithm': alignment_algorithm,
        'data_format': data_format,
        'cache_dir': cache_dir,
        'use_cache': use_cache,
        'normalize_text': normalize_text,
        'enable_aux_mlm': enable_aux_mlm,
        'aux_mlm_prob': aux_mlm_prob,
    }
    
    # 训练数据加载器
    train_dataset = GapReLMDataset(data_file=train_file, **dataset_kwargs)
    
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True
        )
    
    # 验证数据加载器
    dev_loader = None
    if dev_file and os.path.exists(dev_file):
        dev_dataset = GapReLMDataset(data_file=dev_file, **dataset_kwargs)
        
        if distributed:
            dev_sampler = DistributedSampler(
                dev_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                sampler=dev_sampler,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True
            )
        else:
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True
            )
    
    # 测试数据加载器
    test_loader = None
    if test_file and os.path.exists(test_file):
        test_dataset = GapReLMDataset(data_file=test_file, **dataset_kwargs)
        
        if distributed:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True
            )
    
    return train_loader, dev_loader, test_loader, tokenizer


def create_online_data_loaders(
    clean_train_file: str,
    frozen_dev_file: Optional[str] = None,
    test_file: Optional[str] = None,
    tokenizer_name: str = "hfl/chinese-macbert-base",
    max_seq_length: int = 128,
    max_insert_num: int = 3,
    enable_insert: bool = True,
    enable_delete: bool = True,
    alignment_algorithm: str = "levenshtein",
    data_format: str = "mucgec",
    batch_size: int = 32,
    num_workers: int = 4,
    normalize_text: bool = True,
    enable_aux_mlm: bool = True,
    aux_mlm_prob: float = 0.15,
    # 在线增强参数
    p_corrupt: float = 0.7,
    base_lambda: float = 1.5,
    pi_skip: float = 0.2,
    pi_multiply: float = 0.3,
    pi_replace: float = 0.5,
    max_edits_per_sent: int = 4,
    max_insert_k: int = 3,
    # 长度自适应配置
    enable_length_adaptive: bool = True,
    min_length_for_lambda: int = 20,
    max_length_for_lambda: int = 80,
    min_lambda: float = 1.0,
    max_lambda: float = 3.0,
    use_ratio_mode: bool = False,
    error_ratio: float = 0.05,
    # 混淆集配置
    use_default_shape_confusion: bool = True,
    use_default_pinyin_confusion: bool = True,
    custom_confusion_files: Optional[List[str]] = None,
    # 保护约束配置
    enable_protection: bool = True,
    # 分布式配置
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    # 干净文件格式
    clean_file_format: str = "txt",
    clean_text_field: str = "text",
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], AutoTokenizer]:
    """
    创建在线动态增强的数据加载器
    
    训练集使用在线增强（每个 epoch 数据不同），
    验证集使用固定的 Frozen-Dev-Synth（确保评估可比）。
    
    Args:
        clean_train_file: 干净训练句子文件路径
        frozen_dev_file: 固定验证集文件路径（已预生成错误的数据）
        test_file: 测试集文件路径
        tokenizer_name: tokenizer 名称或路径
        max_seq_length: 最大序列长度
        max_insert_num: 最大插入数量
        enable_insert: 是否启用插入
        enable_delete: 是否启用删除
        alignment_algorithm: 对齐算法
        data_format: 验证/测试集数据格式
        batch_size: 批大小
        num_workers: 数据加载工作进程数
        normalize_text: 是否规范化文本
        enable_aux_mlm: 是否启用辅助 MLM
        aux_mlm_prob: 辅助 MLM mask 比例
        p_corrupt: 造错概率
        base_lambda: 基础泊松参数
        pi_skip: 删字概率
        pi_multiply: 重复字概率
        pi_replace: 错字概率
        max_edits_per_sent: 单句最大编辑数
        max_insert_k: 单次最大重复字符数
        enable_length_adaptive: 是否启用长度自适应 λ
        min_length_for_lambda: λ 最小值对应的句子长度
        max_length_for_lambda: λ 最大值对应的句子长度
        min_lambda: 最小 λ
        max_lambda: 最大 λ
        use_ratio_mode: 是否使用错误比例模式
        error_ratio: 错误比例
        use_default_shape_confusion: 使用默认形近字混淆集
        use_default_pinyin_confusion: 使用默认音近字混淆集
        custom_confusion_files: 自定义混淆集文件
        enable_protection: 是否启用保护约束
        distributed: 是否分布式训练
        world_size: 分布式世界大小
        rank: 当前进程排名
        clean_file_format: 干净文件格式
        clean_text_field: JSON 中的文本字段名
        
    Returns:
        (train_loader, dev_loader, test_loader, tokenizer)
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 创建 collator
    collator = GapReLMCollator(tokenizer, include_aux_mlm=enable_aux_mlm)
    
    # 加载干净句子
    print(f"Loading clean sentences from {clean_train_file}")
    clean_sentences = load_clean_sentences(
        clean_train_file, 
        file_format=clean_file_format,
        text_field=clean_text_field,
    )
    print(f"Loaded {len(clean_sentences)} clean sentences for online augmentation")
    
    # 创建在线增强数据集
    train_dataset = OnlineAugmentedDataset(
        clean_sentences=clean_sentences,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_insert_num=max_insert_num,
        enable_insert=enable_insert,
        enable_delete=enable_delete,
        alignment_algorithm=alignment_algorithm,
        normalize_text=normalize_text,
        enable_aux_mlm=enable_aux_mlm,
        aux_mlm_prob=aux_mlm_prob,
        p_corrupt=p_corrupt,
        base_lambda=base_lambda,
        pi_skip=pi_skip,
        pi_multiply=pi_multiply,
        pi_replace=pi_replace,
        max_edits_per_sent=max_edits_per_sent,
        max_insert_k=max_insert_k,
        enable_length_adaptive=enable_length_adaptive,
        min_length_for_lambda=min_length_for_lambda,
        max_length_for_lambda=max_length_for_lambda,
        min_lambda=min_lambda,
        max_lambda=max_lambda,
        use_ratio_mode=use_ratio_mode,
        error_ratio=error_ratio,
        use_default_shape_confusion=use_default_shape_confusion,
        use_default_pinyin_confusion=use_default_pinyin_confusion,
        custom_confusion_files=custom_confusion_files,
        enable_protection=enable_protection,
    )
    
    # 训练数据加载器
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True
        )
    
    # 验证数据加载器（使用固定的 Frozen-Dev-Synth）
    dev_loader = None
    if frozen_dev_file and os.path.exists(frozen_dev_file):
        print(f"Loading frozen dev set from {frozen_dev_file}")
        dataset_kwargs = {
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'max_insert_num': max_insert_num,
            'enable_insert': enable_insert,
            'enable_delete': enable_delete,
            'alignment_algorithm': alignment_algorithm,
            'data_format': data_format,
            'cache_dir': None,
            'use_cache': False,
            'normalize_text': normalize_text,
            'enable_aux_mlm': enable_aux_mlm,
            'aux_mlm_prob': aux_mlm_prob,
        }
        dev_dataset = GapReLMDataset(data_file=frozen_dev_file, **dataset_kwargs)
        
        if distributed:
            dev_sampler = DistributedSampler(
                dev_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                sampler=dev_sampler,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True
            )
        else:
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True
            )
    
    # 测试数据加载器
    test_loader = None
    if test_file and os.path.exists(test_file):
        dataset_kwargs = {
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'max_insert_num': max_insert_num,
            'enable_insert': enable_insert,
            'enable_delete': enable_delete,
            'alignment_algorithm': alignment_algorithm,
            'data_format': data_format,
            'cache_dir': None,
            'use_cache': False,
            'normalize_text': normalize_text,
            'enable_aux_mlm': enable_aux_mlm,
            'aux_mlm_prob': aux_mlm_prob,
        }
        test_dataset = GapReLMDataset(data_file=test_file, **dataset_kwargs)
        
        if distributed:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True
            )
    
    return train_loader, dev_loader, test_loader, tokenizer


def create_inference_loader(
    data_file: str,
    tokenizer,
    max_seq_length: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    data_format: str = "parallel",
) -> DataLoader:
    """
    创建推理用数据加载器 (简化版，不需要对齐)
    
    Args:
        data_file: 数据文件路径
        tokenizer: tokenizer
        max_seq_length: 最大序列长度
        batch_size: 批大小
        num_workers: 工作进程数
        data_format: 数据格式
        
    Returns:
        DataLoader
    """
    # 推理时不需要 target，只需要 source
    # 这里使用简化的加载方式
    
    from torch.utils.data import Dataset
    
    class InferenceDataset(Dataset):
        def __init__(self, data_file, tokenizer, max_seq_length, data_format):
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length
            self.samples = self._load_data(data_file, data_format)
        
        def _load_data(self, data_file, data_format):
            samples = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if data_format == "parallel":
                        parts = line.split('\t')
                        samples.append(parts[0])
                    else:
                        samples.append(line)
            
            return samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            source = self.samples[idx]
            encoding = self.tokenizer(
                source,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'source_text': source,
            }
    
    dataset = InferenceDataset(data_file, tokenizer, max_seq_length, data_format)
    
    def collate_fn(batch):
        import torch
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'source_texts': [item['source_text'] for item in batch],
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
