"""
PyTorch Dataset 和 Collator
将处理后的样本转换为模型输入格式

支持三种模式：
1. 静态模式 (GapReLMDataset): 加载预处理好的训练数据（全部加载到内存，适合小数据集）
2. 惰性加载模式 (LazyGapReLMDataset): 按需读取数据（适合大规模数据集，节省内存）
3. 在线动态模式 (OnlineAugmentedDataset): 在 __getitem__ 中实时生成错误
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import os
import pickle
import random
import math
import mmap
import struct
from tqdm import tqdm

from .label_generator import ProcessedSample, create_sample_processor
from .preprocessor import TextPreprocessor, SentenceSplitter


@dataclass
class ModelInput:
    """模型输入数据结构"""
    # 编码器输入 (源序列)
    input_ids: torch.Tensor           # [seq_len]
    attention_mask: torch.Tensor      # [seq_len]
    
    # Planner 标签
    op_labels: torch.Tensor           # [src_len]，值域 {0, 1, 2}
    insert_labels: torch.Tensor       # [src_len]，值域 {0, ..., K}
    
    # Infiller 输入 (模板序列)
    template_input_ids: torch.Tensor  # [template_len]
    template_attention_mask: torch.Tensor  # [template_len]
    
    # Infiller 标签
    infill_labels: torch.Tensor       # [template_len]，非mask位置为-100
    
    # 辅助 MLM 标签 (可选)
    aux_mlm_labels: Optional[torch.Tensor] = None  # [src_len]
    
    # 元信息
    sample_id: Optional[str] = None
    source_text: Optional[str] = None
    target_text: Optional[str] = None


class GapReLMDataset(Dataset):
    """Gap-ReLM 数据集
    
    支持两种 MASK 模式:
    - full_mask_mode=False (稀疏 MASK): 模板中只在需要修改的位置有 [MASK]
    - full_mask_mode=True (全 MASK): 模板格式为 [CLS] source [SEP] [MASK]*N [SEP]
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_seq_length: int = 128,
        max_insert_num: int = 3,
        enable_insert: bool = True,
        enable_delete: bool = True,
        alignment_algorithm: str = "levenshtein",
        data_format: str = "mucgec",
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        normalize_text: bool = True,
        enable_aux_mlm: bool = True,
        aux_mlm_prob: float = 0.15,
        full_mask_mode: bool = False,
    ):
        """
        Args:
            data_file: 数据文件路径
            tokenizer: HuggingFace tokenizer
            max_seq_length: 最大序列长度
            max_insert_num: 最大插入数量
            enable_insert: 是否启用插入
            enable_delete: 是否启用删除
            alignment_algorithm: 对齐算法
            data_format: 数据格式
            cache_dir: 缓存目录
            use_cache: 是否使用缓存
            normalize_text: 是否规范化文本
            enable_aux_mlm: 是否启用辅助MLM
            aux_mlm_prob: 辅助MLM的mask比例
            full_mask_mode: 是否使用全 MASK 模式
        """
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_insert_num = max_insert_num
        self.enable_aux_mlm = enable_aux_mlm
        self.aux_mlm_prob = aux_mlm_prob
        self.full_mask_mode = full_mask_mode
        
        # 创建预处理器
        self.preprocessor = TextPreprocessor() if normalize_text else None
        
        # 创建样本处理器
        self.sample_processor = create_sample_processor(
            max_insert_num=max_insert_num,
            enable_insert=enable_insert,
            enable_delete=enable_delete,
            alignment_algorithm=alignment_algorithm
        )
        
        # 数据加载器
        self.data_loader_map = {
            "mucgec": self._load_mucgec,
            "sighan": self._load_sighan,
            "ecspell": self._load_ecspell,
            "custom": self._load_custom,
            "parallel": self._load_parallel,
        }
        
        # 加载数据
        self.samples = self._load_and_process(
            data_file, 
            data_format, 
            cache_dir, 
            use_cache
        )
    
    def _load_and_process(
        self,
        data_file: str,
        data_format: str,
        cache_dir: Optional[str],
        use_cache: bool
    ) -> List[ProcessedSample]:
        """加载并处理数据"""
        
        # 检查缓存
        if use_cache and cache_dir:
            cache_file = self._get_cache_path(data_file, cache_dir)
            if os.path.exists(cache_file):
                print(f"Loading from cache: {cache_file}")
                return self._load_cache(cache_file)
        
        # 加载原始数据
        if data_format not in self.data_loader_map:
            raise ValueError(f"Unknown data format: {data_format}")
        
        raw_pairs = self.data_loader_map[data_format](data_file)
        
        # 检查是否有预计算样本（由_load_mucgec设置）
        if hasattr(self, '_has_precomputed') and self._has_precomputed:
            samples = self._process_precomputed_samples()
        else:
            # 传统流程：预处理和对齐
            samples = []
            print(f"Processing {len(raw_pairs)} samples...")
            for i, (source, target) in enumerate(tqdm(raw_pairs)):
                # 预处理
                if self.preprocessor:
                    source, target = self.preprocessor.preprocess_pair(source, target)
                
                # 跳过空样本
                if not source or not target:
                    continue
                
                # 跳过过长样本
                if len(source) > self.max_seq_length - 2 or len(target) > self.max_seq_length - 2:
                    continue
                
                try:
                    sample = self.sample_processor.process(
                        source, target, sample_id=f"sample_{i}"
                    )
                    samples.append(sample)
                except Exception as e:
                    print(f"Warning: Failed to process sample {i}: {e}")
                    continue
        
        # 保存缓存
        if use_cache and cache_dir:
            self._save_cache(samples, cache_file)
        
        print(f"Loaded {len(samples)} samples")
        return samples
    
    def _process_precomputed_samples(self) -> List[ProcessedSample]:
        """处理预计算标签的样本（无需对齐）"""
        from .label_generator import PlannerLabels, GoldTemplate, ProcessedSample
        from .alignment import AlignmentResult, EditOperation, EditType
        
        samples = []
        print(f"加载 {len(self._precomputed_samples)} 个预计算样本...")
        
        for i, data in enumerate(tqdm(self._precomputed_samples)):
            source = data['source']
            target = data['target']
            
            # 跳过过长样本
            if len(source) > self.max_seq_length - 2:
                continue
            
            # 构建 PlannerLabels
            planner_labels = PlannerLabels(
                op_labels=data['op_labels'],
                insert_labels=data['insert_labels'],
                source=source,
                target=target,
            )
            
            # 构建 GoldTemplate
            template_tokens = data.get('template_tokens', list(source))
            gold_tokens = data.get('gold_tokens', [])
            mask_positions = data.get('mask_positions', [])
            
            gold_template = GoldTemplate(
                template_tokens=template_tokens,
                gold_tokens=gold_tokens,
                mask_positions=mask_positions,
                source=source,
                target=target,
            )
            
            # 构建空的 AlignmentResult（预计算不需要）
            alignment_result = AlignmentResult(
                source=source,
                target=target,
                operations=[],
                edit_distance=len([op for op in data['op_labels'] if op != 0]),
            )
            
            sample = ProcessedSample(
                source=source,
                target=target,
                planner_labels=planner_labels,
                gold_template=gold_template,
                alignment_result=alignment_result,
                sample_id=f"sample_{i}",
            )
            samples.append(sample)
        
        return samples
    
    def _get_cache_path(self, data_file: str, cache_dir: str) -> str:
        """获取缓存文件路径"""
        os.makedirs(cache_dir, exist_ok=True)
        basename = os.path.basename(data_file)
        return os.path.join(cache_dir, f"{basename}.cache.pkl")
    
    def _load_cache(self, cache_file: str) -> List[ProcessedSample]:
        """加载缓存"""
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def _save_cache(self, samples: List[ProcessedSample], cache_file: str):
        """保存缓存"""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Cache saved to: {cache_file}")
    
    # ========== 数据加载器 ==========
    
    def _load_mucgec(self, data_file: str) -> List[Tuple[str, str]]:
        """加载 MuCGEC 格式数据
        
        支持两种格式：
        1. 基础格式：{"source": "...", "target": "..."} - 需要后续对齐
        2. 预计算格式：{"source": "...", "target": "...", "op_labels": [...], ...} - 直接使用
        """
        pairs = []
        self._precomputed_samples = []  # 存储预计算样本
        self._has_precomputed = False
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    source = data.get('source', data.get('src', ''))
                    target = data.get('target', data.get('tgt', ''))
                    
                    # MuCGEC 可能有多个参考答案，这里取第一个
                    if isinstance(target, list):
                        target = target[0] if target else source
                    
                    if source and target:
                        # 检查是否有预计算标签
                        if 'op_labels' in data and 'insert_labels' in data:
                            self._has_precomputed = True
                            self._precomputed_samples.append(data)
                        else:
                            pairs.append((source, target))
                            
                except json.JSONDecodeError:
                    # 可能是制表符分隔格式
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pairs.append((parts[0], parts[1]))
        
        if self._has_precomputed:
            print(f"检测到预计算标签格式，将跳过对齐步骤")
        
        return pairs
    
    def _load_sighan(self, data_file: str) -> List[Tuple[str, str]]:
        """加载 SIGHAN 格式数据"""
        pairs = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        
        return pairs
    
    def _load_ecspell(self, data_file: str) -> List[Tuple[str, str]]:
        """加载 ECSpell 格式数据"""
        pairs = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # ECSpell 格式: source\ttarget 或 JSON
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pairs.append((parts[0], parts[1]))
                else:
                    try:
                        data = json.loads(line)
                        source = data.get('original', data.get('src', ''))
                        target = data.get('correct', data.get('tgt', ''))
                        if source and target:
                            pairs.append((source, target))
                    except:
                        continue
        
        return pairs
    
    def _load_custom(self, data_file: str) -> List[Tuple[str, str]]:
        """加载自定义 JSON 格式数据"""
        pairs = []
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    source = item.get('source', item.get('src', item.get('input', '')))
                    target = item.get('target', item.get('tgt', item.get('output', '')))
                    if source and target:
                        pairs.append((source, target))
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        source = value.get('source', value.get('src', ''))
                        target = value.get('target', value.get('tgt', ''))
                        if source and target:
                            pairs.append((source, target))
        
        return pairs
    
    def _load_parallel(self, data_file: str) -> List[Tuple[str, str]]:
        """加载平行语料格式 (制表符分隔)"""
        pairs = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        sample = self.samples[idx]
        return self._convert_to_features(sample)
    
    def _convert_to_features(self, sample: ProcessedSample) -> Dict[str, Any]:
        """将 ProcessedSample 转换为模型输入特征
        
        根据 full_mask_mode 选择不同的模板构建方式:
        - 稀疏 MASK: [CLS] template_with_sparse_masks [SEP]
        - 全 MASK: [CLS] source [SEP] [MASK]*N [SEP]
        """
        source = sample.source
        target = sample.target
        
        # 1. 编码源序列
        source_encoding = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. 处理 Planner 标签
        # 注意：需要对齐到 tokenizer 的输出
        # 这里假设是字符级 tokenizer (MacBERT)
        src_len = len(source)
        
        # Pad Planner 标签到 max_seq_length
        # [CLS] + tokens + [SEP] + padding
        op_labels = [-100] + sample.planner_labels.op_labels[:self.max_seq_length-2] + [-100]
        insert_labels = [0] + sample.planner_labels.insert_labels[:self.max_seq_length-2] + [0]
        
        # Padding
        op_labels = op_labels + [-100] * (self.max_seq_length - len(op_labels))
        insert_labels = insert_labels + [0] * (self.max_seq_length - len(insert_labels))
        
        if self.full_mask_mode:
            # 全 MASK 模式: 构建 [CLS] source [SEP] [MASK]*N [SEP]
            return self._convert_to_full_mask_features(
                sample, source_encoding, op_labels, insert_labels
            )
        else:
            # 稀疏 MASK 模式（原有逻辑）
            return self._convert_to_sparse_mask_features(
                sample, source_encoding, op_labels, insert_labels
            )
    
    def _convert_to_sparse_mask_features(
        self,
        sample: ProcessedSample,
        source_encoding: Dict,
        op_labels: List[int],
        insert_labels: List[int],
    ) -> Dict[str, Any]:
        """稀疏 MASK 模式的特征转换（原有逻辑）"""
        # 3. 构建模板输入
        template_tokens = sample.gold_template.template_tokens
        template_str = ''.join(template_tokens).replace('[MASK]', self.tokenizer.mask_token)
        
        template_encoding = self.tokenizer(
            template_str,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 4. 构建 Infiller 标签
        # 找出 [MASK] 位置并填入正确答案
        mask_token_id = self.tokenizer.mask_token_id
        infill_labels = [-100] * self.max_seq_length
        
        template_input_ids = template_encoding['input_ids'].squeeze(0)
        mask_positions_in_ids = (template_input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        
        gold_tokens = sample.gold_template.gold_tokens
        for i, pos in enumerate(mask_positions_in_ids):
            if i < len(gold_tokens):
                token_id = self.tokenizer.convert_tokens_to_ids(gold_tokens[i])
                infill_labels[pos.item()] = token_id
        
        # 5. 辅助 MLM (可选)
        aux_mlm_labels = None
        if self.enable_aux_mlm:
            aux_mlm_labels = self._create_aux_mlm_labels(
                source_encoding['input_ids'].squeeze(0),
                sample.planner_labels.op_labels
            )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'op_labels': torch.tensor(op_labels, dtype=torch.long),
            'insert_labels': torch.tensor(insert_labels, dtype=torch.long),
            'template_input_ids': template_encoding['input_ids'].squeeze(0),
            'template_attention_mask': template_encoding['attention_mask'].squeeze(0),
            'infill_labels': torch.tensor(infill_labels, dtype=torch.long),
            'aux_mlm_labels': aux_mlm_labels,
            'sample_id': sample.sample_id,
            'source_text': sample.source,
            'target_text': sample.target,
        }
    
    def _convert_to_full_mask_features(
        self,
        sample: ProcessedSample,
        source_encoding: Dict,
        op_labels: List[int],
        insert_labels: List[int],
    ) -> Dict[str, Any]:
        """全 MASK 模式的特征转换
        
        模板格式: [CLS] source [SEP] [MASK]*N [SEP]
        Labels: source 部分为 -100，[MASK]*N 部分为 target token IDs
        """
        source = sample.source
        target = sample.target
        
        # 计算 target 长度 N
        # N 由 planner labels 推导: KEEP/REPLACE 各贡献 1，DELETE 贡献 0，INSERT(k) 贡献 k
        target_length = self._compute_target_length_from_labels(
            sample.planner_labels.op_labels,
            sample.planner_labels.insert_labels
        )
        
        # 构建模板: [CLS] source [SEP] [MASK]*N [SEP]
        # 计算各部分长度
        src_len = len(source)
        
        # 确保不超过 max_seq_length
        # [CLS] + src + [SEP] + N + [SEP] <= max_seq_length
        max_target_len = self.max_seq_length - src_len - 3  # -3 for [CLS], [SEP], [SEP]
        actual_target_len = min(target_length, max(0, max_target_len))
        
        # 构建 template_input_ids
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        
        source_input_ids = source_encoding['input_ids'].squeeze(0).tolist()
        # source_input_ids 格式: [CLS] source_tokens [SEP] [PAD]...
        
        # 提取 source tokens (不含 CLS、SEP、PAD)
        source_tokens = []
        for i, tid in enumerate(source_input_ids):
            if tid not in [cls_id, sep_id, pad_id]:
                source_tokens.append(tid)
        
        # 构建新序列: [CLS] source [SEP] [MASK]*N [SEP]
        template_ids = [cls_id] + source_tokens[:src_len] + [sep_id]
        target_start_pos = len(template_ids)  # [MASK] 开始位置
        template_ids += [mask_id] * actual_target_len
        template_ids += [sep_id]
        
        # Padding
        total_len = len(template_ids)
        pad_len = self.max_seq_length - total_len
        template_ids += [pad_id] * max(0, pad_len)
        template_ids = template_ids[:self.max_seq_length]  # 截断
        
        # 构建 attention_mask
        template_mask = [1] * min(total_len, self.max_seq_length)
        template_mask += [0] * max(0, self.max_seq_length - total_len)
        template_mask = template_mask[:self.max_seq_length]
        
        # 构建 infill_labels: source 部分 -100，[MASK] 部分为 target token IDs
        infill_labels = [-100] * self.max_seq_length
        
        # 对 target 进行 tokenize
        target_encoding = self.tokenizer(
            target,
            add_special_tokens=False,
            max_length=actual_target_len,
            truncation=True,
            return_tensors='pt'
        )
        target_token_ids = target_encoding['input_ids'].squeeze(0).tolist()
        
        # 填入 target labels
        for i, tid in enumerate(target_token_ids):
            pos = target_start_pos + i
            if pos < self.max_seq_length:
                infill_labels[pos] = tid
        
        # 5. 辅助 MLM (可选)
        aux_mlm_labels = None
        if self.enable_aux_mlm:
            aux_mlm_labels = self._create_aux_mlm_labels(
                source_encoding['input_ids'].squeeze(0),
                sample.planner_labels.op_labels
            )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'op_labels': torch.tensor(op_labels, dtype=torch.long),
            'insert_labels': torch.tensor(insert_labels, dtype=torch.long),
            'template_input_ids': torch.tensor(template_ids, dtype=torch.long),
            'template_attention_mask': torch.tensor(template_mask, dtype=torch.long),
            'infill_labels': torch.tensor(infill_labels, dtype=torch.long),
            'aux_mlm_labels': aux_mlm_labels,
            'sample_id': sample.sample_id,
            'source_text': sample.source,
            'target_text': sample.target,
            # 全 MASK 模式额外信息
            'target_start_pos': target_start_pos,
            'target_length': actual_target_len,
        }
    
    def _compute_target_length_from_labels(
        self,
        op_labels: List[int],
        insert_labels: List[int],
    ) -> int:
        """根据 planner labels 计算 target 长度
        
        计算规则:
        - KEEP (0): 贡献 1
        - DELETE (1): 贡献 0
        - REPLACE (2): 贡献 1
        - INSERT: 贡献 insert_labels[i]
        """
        length = 0
        for i, op in enumerate(op_labels):
            if op == 0:  # KEEP
                length += 1
            elif op == 1:  # DELETE
                pass  # 贡献 0
            elif op == 2:  # REPLACE
                length += 1
            
            # INSERT 贡献（仅对非删除位置）
            if op != 1 and i < len(insert_labels):
                length += insert_labels[i]
        
        return length
    
    def _create_aux_mlm_labels(
        self,
        input_ids: torch.Tensor,
        op_labels: List[int]
    ) -> torch.Tensor:
        """
        创建辅助 MLM 标签
        随机 mask 非错误位置的字符
        """
        import random
        
        labels = torch.full_like(input_ids, -100)
        
        # 找出 KEEP 位置（非错误字符）
        keep_positions = []
        for i, op in enumerate(op_labels):
            if op == 0:  # KEEP
                # 对应到 input_ids 的位置 (加1因为[CLS])
                pos = i + 1
                if pos < len(input_ids) - 1:  # 避免[SEP]
                    keep_positions.append(pos)
        
        # 随机选择要 mask 的位置
        num_to_mask = int(len(keep_positions) * self.aux_mlm_prob)
        if num_to_mask > 0:
            positions_to_mask = random.sample(keep_positions, num_to_mask)
            for pos in positions_to_mask:
                labels[pos] = input_ids[pos].clone()
        
        return labels


class LazyGapReLMDataset(Dataset):
    """惰性加载 Gap-ReLM 数据集（内存友好版本）
    
    与 GapReLMDataset 不同，此类不会将所有数据加载到内存中。
    它预先扫描文件建立索引（记录每行的字节偏移），
    在 __getitem__ 时按需读取单行数据。
    
    适用场景：
    - 大规模训练数据（百万级以上样本）
    - 内存受限的环境
    - 预计算标签格式的 JSONL 文件
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_seq_length: int = 128,
        max_insert_num: int = 3,
        enable_insert: bool = True,
        enable_delete: bool = True,
        normalize_text: bool = True,
        enable_aux_mlm: bool = True,
        aux_mlm_prob: float = 0.15,
        index_cache_dir: Optional[str] = None,
        use_index_cache: bool = True,
        full_mask_mode: bool = False,
    ):
        """
        Args:
            data_file: JSONL 格式数据文件路径（预计算标签格式）
            tokenizer: HuggingFace tokenizer
            max_seq_length: 最大序列长度
            max_insert_num: 最大插入数量
            enable_insert: 是否启用插入
            enable_delete: 是否启用删除
            normalize_text: 是否规范化文本
            enable_aux_mlm: 是否启用辅助MLM
            aux_mlm_prob: 辅助MLM的mask比例
            index_cache_dir: 索引缓存目录
            use_index_cache: 是否使用/保存索引缓存
            full_mask_mode: 是否使用全 MASK 模式
        """
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_insert_num = max_insert_num
        self.enable_insert = enable_insert
        self.enable_delete = enable_delete
        self.enable_aux_mlm = enable_aux_mlm
        self.aux_mlm_prob = aux_mlm_prob
        self.full_mask_mode = full_mask_mode
        
        # 预处理器
        self.preprocessor = TextPreprocessor() if normalize_text else None
        
        # 构建或加载行偏移索引
        self.line_offsets: List[int] = []
        self._build_index(index_cache_dir, use_index_cache)
        
        # 打开文件句柄（用于后续读取）
        self._file_handle = None
        
        print(f"LazyGapReLMDataset initialized with {len(self.line_offsets)} samples (memory-efficient mode)")
    
    def _build_index(self, index_cache_dir: Optional[str], use_index_cache: bool):
        """构建行偏移索引"""
        index_file = None
        if use_index_cache and index_cache_dir:
            os.makedirs(index_cache_dir, exist_ok=True)
            basename = os.path.basename(self.data_file)
            index_file = os.path.join(index_cache_dir, f"{basename}.lazy_index.pkl")
            
            # 尝试加载缓存
            if os.path.exists(index_file):
                # 检查索引是否比数据文件新
                if os.path.getmtime(index_file) > os.path.getmtime(self.data_file):
                    print(f"Loading line index from cache: {index_file}")
                    with open(index_file, 'rb') as f:
                        self.line_offsets = pickle.load(f)
                    return
        
        # 扫描文件构建索引
        print(f"Building line index for {self.data_file}...")
        self.line_offsets = []
        
        with open(self.data_file, 'rb') as f:
            offset = 0
            for line in tqdm(f, desc="Indexing"):
                line_str = line.decode('utf-8').strip()
                if line_str:
                    # 验证是否为有效 JSON 且为预计算格式
                    try:
                        data = json.loads(line_str)
                        source = data.get('source', data.get('src', ''))
                        # 检查是否为预计算格式
                        if 'op_labels' in data and 'insert_labels' in data and source:
                            # 检查长度限制
                            if len(source) <= self.max_seq_length - 2:
                                self.line_offsets.append(offset)
                    except json.JSONDecodeError:
                        pass
                offset += len(line)
        
        print(f"Indexed {len(self.line_offsets)} valid samples")
        
        # 保存索引缓存
        if use_index_cache and index_file:
            with open(index_file, 'wb') as f:
                pickle.dump(self.line_offsets, f)
            print(f"Index cache saved to: {index_file}")
    
    def _get_file_handle(self):
        """获取文件句柄（懒加载）"""
        if self._file_handle is None:
            self._file_handle = open(self.data_file, 'r', encoding='utf-8')
        return self._file_handle
    
    def _read_line_at_offset(self, offset: int) -> str:
        """读取指定偏移位置的行"""
        f = self._get_file_handle()
        f.seek(offset)
        return f.readline().strip()
    
    def __len__(self) -> int:
        return len(self.line_offsets)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """按需读取单个样本"""
        offset = self.line_offsets[idx]
        line = self._read_line_at_offset(offset)
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # 如果解析失败，返回空样本（会在后续被过滤）
            raise ValueError(f"Failed to parse JSON at index {idx}")
        
        return self._convert_precomputed_to_features(data, idx)
    
    def _convert_precomputed_to_features(self, data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """将预计算数据转换为模型输入特征"""
        source = data['source']
        target = data['target']
        op_labels = data['op_labels']
        insert_labels = data['insert_labels']
        
        # 预处理
        if self.preprocessor:
            source = self.preprocessor.preprocess(source)
            target = self.preprocessor.preprocess(target)
        
        # 编码源序列
        source_encoding = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理 Planner 标签
        # [CLS] + tokens + [SEP] + padding
        op_labels_padded = [-100] + op_labels[:self.max_seq_length-2] + [-100]
        insert_labels_padded = [0] + insert_labels[:self.max_seq_length-2] + [0]
        
        # Padding
        op_labels_padded = op_labels_padded + [-100] * (self.max_seq_length - len(op_labels_padded))
        insert_labels_padded = insert_labels_padded + [0] * (self.max_seq_length - len(insert_labels_padded))
        
        if self.full_mask_mode:
            # 全 MASK 模式
            return self._build_full_mask_features(
                source, target, op_labels, insert_labels,
                source_encoding, op_labels_padded, insert_labels_padded, idx
            )
        else:
            # 稀疏 MASK 模式（原有逻辑）
            return self._build_sparse_mask_features(
                source, target, data, source_encoding, 
                op_labels_padded, insert_labels_padded, op_labels, idx
            )
    
    def _build_sparse_mask_features(
        self, source, target, data, source_encoding,
        op_labels_padded, insert_labels_padded, op_labels, idx
    ) -> Dict[str, Any]:
        """稀疏 MASK 模式的特征构建（原有逻辑）"""
        # 构建模板输入
        template_tokens = data.get('template_tokens', list(source))
        template_str = ''.join(template_tokens).replace('[MASK]', self.tokenizer.mask_token)
        
        template_encoding = self.tokenizer(
            template_str,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 构建 Infiller 标签
        mask_token_id = self.tokenizer.mask_token_id
        infill_labels = [-100] * self.max_seq_length
        
        template_input_ids = template_encoding['input_ids'].squeeze(0)
        mask_positions_in_ids = (template_input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        
        gold_tokens = data.get('gold_tokens', [])
        for i, pos in enumerate(mask_positions_in_ids):
            if i < len(gold_tokens):
                token_id = self.tokenizer.convert_tokens_to_ids(gold_tokens[i])
                infill_labels[pos.item()] = token_id
        
        # 辅助 MLM (可选)
        aux_mlm_labels = None
        if self.enable_aux_mlm:
            aux_mlm_labels = self._create_aux_mlm_labels(
                source_encoding['input_ids'].squeeze(0),
                op_labels
            )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'op_labels': torch.tensor(op_labels_padded, dtype=torch.long),
            'insert_labels': torch.tensor(insert_labels_padded, dtype=torch.long),
            'template_input_ids': template_encoding['input_ids'].squeeze(0),
            'template_attention_mask': template_encoding['attention_mask'].squeeze(0),
            'infill_labels': torch.tensor(infill_labels, dtype=torch.long),
            'aux_mlm_labels': aux_mlm_labels,
            'sample_id': f"lazy_{idx}",
            'source_text': source,
            'target_text': target,
        }
    
    def _build_full_mask_features(
        self, source, target, op_labels, insert_labels,
        source_encoding, op_labels_padded, insert_labels_padded, idx
    ) -> Dict[str, Any]:
        """全 MASK 模式的特征构建"""
        # 计算 target 长度
        target_length = self._compute_target_length(op_labels, insert_labels)
        
        # 构建模板: [CLS] source [SEP] [MASK]*N [SEP]
        src_len = len(source)
        max_target_len = self.max_seq_length - src_len - 3
        actual_target_len = min(target_length, max(0, max_target_len))
        
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        
        source_input_ids = source_encoding['input_ids'].squeeze(0).tolist()
        source_tokens = [tid for tid in source_input_ids if tid not in [cls_id, sep_id, pad_id]]
        
        template_ids = [cls_id] + source_tokens[:src_len] + [sep_id]
        target_start_pos = len(template_ids)
        template_ids += [mask_id] * actual_target_len
        template_ids += [sep_id]
        
        total_len = len(template_ids)
        pad_len = self.max_seq_length - total_len
        template_ids += [pad_id] * max(0, pad_len)
        template_ids = template_ids[:self.max_seq_length]
        
        template_mask = [1] * min(total_len, self.max_seq_length)
        template_mask += [0] * max(0, self.max_seq_length - total_len)
        template_mask = template_mask[:self.max_seq_length]
        
        # 构建 infill_labels
        infill_labels = [-100] * self.max_seq_length
        target_encoding = self.tokenizer(
            target,
            add_special_tokens=False,
            max_length=actual_target_len,
            truncation=True,
            return_tensors='pt'
        )
        target_token_ids = target_encoding['input_ids'].squeeze(0).tolist()
        
        for i, tid in enumerate(target_token_ids):
            pos = target_start_pos + i
            if pos < self.max_seq_length:
                infill_labels[pos] = tid
        
        # 辅助 MLM
        aux_mlm_labels = None
        if self.enable_aux_mlm:
            aux_mlm_labels = self._create_aux_mlm_labels(
                source_encoding['input_ids'].squeeze(0),
                op_labels
            )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'op_labels': torch.tensor(op_labels_padded, dtype=torch.long),
            'insert_labels': torch.tensor(insert_labels_padded, dtype=torch.long),
            'template_input_ids': torch.tensor(template_ids, dtype=torch.long),
            'template_attention_mask': torch.tensor(template_mask, dtype=torch.long),
            'infill_labels': torch.tensor(infill_labels, dtype=torch.long),
            'aux_mlm_labels': aux_mlm_labels,
            'sample_id': f"lazy_{idx}",
            'source_text': source,
            'target_text': target,
            'target_start_pos': target_start_pos,
            'target_length': actual_target_len,
        }
    
    def _compute_target_length(self, op_labels: List[int], insert_labels: List[int]) -> int:
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
    
    def _create_aux_mlm_labels(
        self,
        input_ids: torch.Tensor,
        op_labels: List[int]
    ) -> torch.Tensor:
        """创建辅助 MLM 标签"""
        labels = torch.full_like(input_ids, -100)
        
        keep_positions = []
        for i, op in enumerate(op_labels):
            if op == 0:  # KEEP
                pos = i + 1
                if pos < len(input_ids) - 1:
                    keep_positions.append(pos)
        
        num_to_mask = int(len(keep_positions) * self.aux_mlm_prob)
        if num_to_mask > 0 and keep_positions:
            positions_to_mask = random.sample(keep_positions, min(num_to_mask, len(keep_positions)))
            for pos in positions_to_mask:
                labels[pos] = input_ids[pos].clone()
        
        return labels
    
    def __del__(self):
        """清理文件句柄"""
        if self._file_handle is not None:
            self._file_handle.close()
    
    def __getstate__(self):
        """pickle 序列化时不保存文件句柄"""
        state = self.__dict__.copy()
        state['_file_handle'] = None
        return state
    
    def __setstate__(self, state):
        """pickle 反序列化时重置文件句柄"""
        self.__dict__.update(state)
        self._file_handle = None


class GapReLMCollator:
    """数据批处理 Collator"""
    
    def __init__(self, tokenizer, include_aux_mlm: bool = True):
        self.tokenizer = tokenizer
        self.include_aux_mlm = include_aux_mlm
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将一批样本整理为模型输入"""
        
        # 堆叠张量
        result = {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'op_labels': torch.stack([item['op_labels'] for item in batch]),
            'insert_labels': torch.stack([item['insert_labels'] for item in batch]),
            'template_input_ids': torch.stack([item['template_input_ids'] for item in batch]),
            'template_attention_mask': torch.stack([item['template_attention_mask'] for item in batch]),
            'infill_labels': torch.stack([item['infill_labels'] for item in batch]),
        }
        
        # 辅助 MLM
        if self.include_aux_mlm and batch[0]['aux_mlm_labels'] is not None:
            result['aux_mlm_labels'] = torch.stack([item['aux_mlm_labels'] for item in batch])
        
        # 元信息 (不转为tensor)
        result['sample_ids'] = [item['sample_id'] for item in batch]
        result['source_texts'] = [item['source_text'] for item in batch]
        result['target_texts'] = [item['target_text'] for item in batch]
        
        return result


class LengthAdaptiveLambda:
    """长度自适应的 λ 参数
    
    根据句子长度动态调整泊松分布的 λ 参数：
    - 短句 (< min_length)：使用较小的 λ
    - 长句 (> max_length)：使用较大的 λ
    - 中间：线性插值
    
    也可以使用比例模式：错误数占句子长度的一定比例
    """
    
    def __init__(
        self,
        base_lambda: float = 1.5,
        min_length: int = 20,
        max_length: int = 80,
        min_lambda: float = 1.0,
        max_lambda: float = 3.0,
        use_ratio_mode: bool = False,
        error_ratio: float = 0.05,  # 错误占比 5%
    ):
        self.base_lambda = base_lambda
        self.min_length = min_length
        self.max_length = max_length
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.use_ratio_mode = use_ratio_mode
        self.error_ratio = error_ratio
    
    def get_lambda(self, sentence_length: int) -> float:
        """根据句子长度获取 λ"""
        if self.use_ratio_mode:
            # 比例模式：错误数与句子长度成正比
            return max(1.0, sentence_length * self.error_ratio)
        
        # 线性插值模式
        if sentence_length <= self.min_length:
            return self.min_lambda
        elif sentence_length >= self.max_length:
            return self.max_lambda
        else:
            # 线性插值
            ratio = (sentence_length - self.min_length) / (self.max_length - self.min_length)
            return self.min_lambda + ratio * (self.max_lambda - self.min_lambda)


class OnlineAugmentedDataset(Dataset):
    """在线动态数据增强数据集
    
    每次 __getitem__ 调用时实时生成错误，每个 Epoch 模型看到的数据都不同。
    
    优点：
    1. 无限数据：只要训练不停止，模型永远在看新的错误组合
    2. 防止过拟合：模型无法死记硬背
    3. 充分利用 GPU 计算能力
    
    注意：
    - 需要配合 Frozen-Dev-Synth（固定验证集）使用，确保评估结果可比
    - λ 可以根据句子长度动态调整
    """
    
    def __init__(
        self,
        clean_sentences: List[str],
        tokenizer,
        max_seq_length: int = 128,
        max_insert_num: int = 3,
        enable_insert: bool = True,
        enable_delete: bool = True,
        alignment_algorithm: str = "levenshtein",
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
        # MASK 模式
        full_mask_mode: bool = True,
        # 其他
        seed: Optional[int] = None,
    ):
        """
        Args:
            clean_sentences: 干净的正确句子列表
            tokenizer: HuggingFace tokenizer
            max_seq_length: 最大序列长度
            max_insert_num: Planner 最大插入数量 K
            enable_insert: 是否启用插入操作
            enable_delete: 是否启用删除操作
            alignment_algorithm: 对齐算法
            normalize_text: 是否规范化文本
            enable_aux_mlm: 是否启用辅助 MLM
            aux_mlm_prob: 辅助 MLM 的 mask 比例
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
            error_ratio: 错误比例（用于比例模式）
            use_default_shape_confusion: 使用默认形近字混淆集
            use_default_pinyin_confusion: 使用默认音近字混淆集
            custom_confusion_files: 自定义混淆集文件
            enable_protection: 是否启用保护约束
            full_mask_mode: 是否使用 full MASK 模式（ReLM style）
            seed: 随机种子（用于初始化，实际训练时每个样本随机）
        """
        self.clean_sentences = clean_sentences
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_insert_num = max_insert_num
        self.enable_aux_mlm = enable_aux_mlm
        self.aux_mlm_prob = aux_mlm_prob
        self.full_mask_mode = full_mask_mode
        
        # 预处理器
        self.preprocessor = TextPreprocessor() if normalize_text else None
        
        # 样本处理器（用于生成标签）
        self.sample_processor = create_sample_processor(
            max_insert_num=max_insert_num,
            enable_insert=enable_insert,
            enable_delete=enable_delete,
            alignment_algorithm=alignment_algorithm
        )
        
        # 长度自适应 λ
        self.enable_length_adaptive = enable_length_adaptive
        self.lambda_adapter = LengthAdaptiveLambda(
            base_lambda=base_lambda,
            min_length=min_length_for_lambda,
            max_length=max_length_for_lambda,
            min_lambda=min_lambda,
            max_lambda=max_lambda,
            use_ratio_mode=use_ratio_mode,
            error_ratio=error_ratio,
        )
        self.base_lambda = base_lambda
        
        # 导入并创建错误生成器
        from .confusion_set import ConfusionSet
        from .protected_span import ProtectedSpanDetector
        from .error_generator import ErrorGenerator
        
        # 混淆集
        self.confusion_set = ConfusionSet(
            use_default_shape=use_default_shape_confusion,
            use_default_pinyin=use_default_pinyin_confusion,
            custom_confusion_files=custom_confusion_files,
        )
        
        # 保护检测器
        self.protected_detector = ProtectedSpanDetector() if enable_protection else None
        
        # 造错参数（保存以便在线使用）
        self.p_corrupt = p_corrupt
        self.pi_skip = pi_skip
        self.pi_multiply = pi_multiply
        self.pi_replace = pi_replace
        self.max_edits_per_sent = max_edits_per_sent
        self.max_insert_k = max_insert_k
        self.enable_protection = enable_protection
        
        # 错误生成器（用于在线造错）
        self.error_generator = ErrorGenerator(
            p_corrupt=p_corrupt,
            lambda_=base_lambda,
            pi_skip=pi_skip,
            pi_multiply=pi_multiply,
            pi_replace=pi_replace,
            max_edits_per_sent=max_edits_per_sent,
            max_insert_k=max_insert_k,
            confusion_set=self.confusion_set,
            protected_detector=self.protected_detector,
            enable_protection=enable_protection,
            seed=None,  # 不固定种子，保证随机性
        )
        
        # 预处理：过滤过长句子并规范化
        self._preprocess_sentences()
        
        print(f"OnlineAugmentedDataset initialized with {len(self.clean_sentences)} sentences")
        print(f"  p_corrupt={p_corrupt}, base_lambda={base_lambda}")
        print(f"  pi_skip={pi_skip}, pi_multiply={pi_multiply}, pi_replace={pi_replace}")
        if enable_length_adaptive:
            print(f"  Length adaptive λ: [{min_lambda}, {max_lambda}] for lengths [{min_length_for_lambda}, {max_length_for_lambda}]")
    
    def _preprocess_sentences(self):
        """预处理句子：规范化、过滤过长"""
        processed = []
        for sent in self.clean_sentences:
            # 规范化
            if self.preprocessor:
                sent = self.preprocessor.preprocess(sent)
            
            # 跳过空句子
            if not sent or len(sent) < 5:
                continue
            
            # 跳过过长句子（留出特殊 token 的空间）
            if len(sent) > self.max_seq_length - 10:
                continue
            
            processed.append(sent)
        
        self.clean_sentences = processed
    
    def __len__(self) -> int:
        return len(self.clean_sentences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本（实时造错）"""
        clean_sentence = self.clean_sentences[idx]
        
        # 使用可控 RNG：结合 idx + worker_id + epoch 生成确定性随机种子
        # 这样保证分布式训练时各 worker 数据不同但可复现
        import torch
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # 使用 hash 生成种子，结合 idx 和 worker_id
        sample_seed = hash((idx, worker_id, torch.initial_seed())) & 0x7FFFFFFF
        
        # 为当前样本创建独立的 RNG
        import random
        sample_rng = random.Random(sample_seed)
        
        # 动态调整 λ
        if self.enable_length_adaptive:
            current_lambda = self.lambda_adapter.get_lambda(len(clean_sentence))
            self.error_generator.set_params(lambda_=current_lambda)
        
        # 在线造错（使用独立 RNG）
        corruption_result = self.error_generator.corrupt(clean_sentence, rng=sample_rng)
        
        # 获取 source（可能有错）和 target（正确）
        source = corruption_result.corrupted  # 错误句子（模型输入）
        target = corruption_result.original   # 正确句子（模型目标）
        
        # 处理对齐和生成标签
        try:
            sample = self.sample_processor.process(
                source, target, sample_id=f"online_{idx}"
            )
            return self._convert_to_features(sample)
        except Exception as e:
            # 如果处理失败，使用原句子作为正例（source == target, 全KEEP标签）
            # 这样可以让模型学到：对于正确的句子，不做任何修改
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to process sample {idx}, using as positive example: {e}")
            
            fallback_sample = self.sample_processor.process(
                clean_sentence, clean_sentence, sample_id=f"online_{idx}_positive"
            )
            return self._convert_to_features(fallback_sample)
    
    def _convert_to_features(self, sample: ProcessedSample) -> Dict[str, Any]:
        """将 ProcessedSample 转换为模型输入特征"""
        # 根据 MASK 模式选择不同的转换方法
        if self.full_mask_mode:
            return self._convert_to_full_mask_features(sample)
        else:
            return self._convert_to_sparse_mask_features(sample)
    
    def _convert_to_sparse_mask_features(self, sample: ProcessedSample) -> Dict[str, Any]:
        """将 ProcessedSample 转换为稀疏 MASK 模式的模型输入特征"""
        source = sample.source
        
        # 1. 编码源序列
        source_encoding = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. 处理 Planner 标签
        src_len = len(source)
        
        # Pad Planner 标签到 max_seq_length
        # [CLS] + tokens + [SEP] + padding
        op_labels = [-100] + sample.planner_labels.op_labels[:self.max_seq_length-2] + [-100]
        insert_labels = [0] + sample.planner_labels.insert_labels[:self.max_seq_length-2] + [0]
        
        # Padding
        op_labels = op_labels + [-100] * (self.max_seq_length - len(op_labels))
        insert_labels = insert_labels + [0] * (self.max_seq_length - len(insert_labels))
        
        # 3. 构建模板输入
        template_tokens = sample.gold_template.template_tokens
        template_str = ''.join(template_tokens).replace('[MASK]', self.tokenizer.mask_token)
        
        template_encoding = self.tokenizer(
            template_str,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 4. 构建 Infiller 标签
        mask_token_id = self.tokenizer.mask_token_id
        infill_labels = [-100] * self.max_seq_length
        
        template_input_ids = template_encoding['input_ids'].squeeze(0)
        mask_positions_in_ids = (template_input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        
        gold_tokens = sample.gold_template.gold_tokens
        for i, pos in enumerate(mask_positions_in_ids):
            if i < len(gold_tokens):
                token_id = self.tokenizer.convert_tokens_to_ids(gold_tokens[i])
                infill_labels[pos.item()] = token_id
        
        # 5. 辅助 MLM (可选)
        aux_mlm_labels = None
        if self.enable_aux_mlm:
            aux_mlm_labels = self._create_aux_mlm_labels(
                source_encoding['input_ids'].squeeze(0),
                sample.planner_labels.op_labels
            )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'op_labels': torch.tensor(op_labels, dtype=torch.long),
            'insert_labels': torch.tensor(insert_labels, dtype=torch.long),
            'template_input_ids': template_encoding['input_ids'].squeeze(0),
            'template_attention_mask': template_encoding['attention_mask'].squeeze(0),
            'infill_labels': torch.tensor(infill_labels, dtype=torch.long),
            'aux_mlm_labels': aux_mlm_labels,
            'sample_id': sample.sample_id,
            'source_text': source,
            'target_text': sample.target,
        }
    
    def _convert_to_full_mask_features(self, sample: ProcessedSample) -> Dict[str, Any]:
        """将 ProcessedSample 转换为 full MASK 模式的模型输入特征（ReLM style）
        
        模板格式: [CLS] source [SEP] [MASK]*N [SEP]
        其中 N 由 Planner 标签推断。
        """
        source = sample.source
        target = sample.target
        
        # 1. 编码源序列
        source_encoding = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. 处理 Planner 标签
        op_labels_raw = sample.planner_labels.op_labels
        insert_labels_raw = sample.planner_labels.insert_labels
        
        # Pad Planner 标签
        op_labels = [-100] + op_labels_raw[:self.max_seq_length-2] + [-100]
        insert_labels = [0] + insert_labels_raw[:self.max_seq_length-2] + [0]
        op_labels = op_labels + [-100] * (self.max_seq_length - len(op_labels))
        insert_labels = insert_labels + [0] * (self.max_seq_length - len(insert_labels))
        
        # 3. 计算 target 长度（从 Planner 标签）
        target_length = self._compute_target_length_from_labels(op_labels_raw, insert_labels_raw)
        
        # 4. 编码 target（用于获取 token ids 作为标签）
        target_encoding = self.tokenizer(
            target,
            add_special_tokens=False,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors='pt'
        )
        target_token_ids = target_encoding['input_ids'].squeeze(0).tolist()
        
        # 5. 构建 full MASK 模板: [CLS] source [SEP] [MASK]*N [SEP]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        
        # 获取 source token ids（不含特殊 token）
        source_input_ids = source_encoding['input_ids'].squeeze(0).tolist()
        source_tokens = [tid for tid in source_input_ids if tid not in [cls_id, sep_id, pad_id]]
        src_len = len(source)
        
        # 计算可用空间：max_seq_length - [CLS] - src_len - [SEP] - [SEP] = max_seq_length - src_len - 3
        max_target_len = self.max_seq_length - src_len - 3
        actual_target_len = min(target_length, max(0, max_target_len))
        
        # 构建模板
        template_ids = [cls_id] + source_tokens[:src_len] + [sep_id]
        target_start_pos = len(template_ids)  # [MASK] 区域的起始位置
        template_ids += [mask_id] * actual_target_len
        template_ids += [sep_id]
        
        # 填充到 max_seq_length
        total_len = len(template_ids)
        template_ids += [pad_id] * (self.max_seq_length - total_len)
        template_ids = template_ids[:self.max_seq_length]
        
        # attention mask
        template_mask = [1] * min(total_len, self.max_seq_length)
        template_mask += [0] * (self.max_seq_length - len(template_mask))
        template_mask = template_mask[:self.max_seq_length]
        
        # 6. 构建 Infiller 标签（target token ids 放在 MASK 位置）
        infill_labels = [-100] * self.max_seq_length
        for i, tid in enumerate(target_token_ids[:actual_target_len]):
            pos = target_start_pos + i
            if pos < self.max_seq_length:
                infill_labels[pos] = tid
        
        # 7. 辅助 MLM (可选)
        aux_mlm_labels = None
        if self.enable_aux_mlm:
            aux_mlm_labels = self._create_aux_mlm_labels(
                source_encoding['input_ids'].squeeze(0),
                op_labels_raw
            )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'op_labels': torch.tensor(op_labels, dtype=torch.long),
            'insert_labels': torch.tensor(insert_labels, dtype=torch.long),
            'template_input_ids': torch.tensor(template_ids, dtype=torch.long),
            'template_attention_mask': torch.tensor(template_mask, dtype=torch.long),
            'infill_labels': torch.tensor(infill_labels, dtype=torch.long),
            'aux_mlm_labels': aux_mlm_labels,
            'sample_id': sample.sample_id,
            'source_text': source,
            'target_text': target,
            # Full MASK 模式额外信息
            'target_length': target_length,
            'target_start_pos': target_start_pos,
        }
    
    def _compute_target_length_from_labels(
        self, 
        op_labels: List[int], 
        insert_labels: List[int]
    ) -> int:
        """从 Planner 标签计算 target 长度
        
        规则：
        - KEEP (0): +1
        - DELETE (1): +0
        - REPLACE (2): +1
        - INSERT: 每个位置 +insert_labels[i]（仅非 DELETE 位置）
        """
        length = 0
        for i, op in enumerate(op_labels):
            if op == 0:  # KEEP
                length += 1
            elif op == 2:  # REPLACE
                length += 1
            # DELETE (1) 贡献 0
            
            # INSERT: 非 DELETE 位置加上插入数量
            if op != 1 and i < len(insert_labels):
                length += insert_labels[i]
        
        return length
    
    def _create_aux_mlm_labels(
        self,
        input_ids: torch.Tensor,
        op_labels: List[int]
    ) -> torch.Tensor:
        """创建辅助 MLM 标签"""
        labels = torch.full_like(input_ids, -100)
        
        # 找出 KEEP 位置（非错误字符）
        keep_positions = []
        for i, op in enumerate(op_labels):
            if op == 0:  # KEEP
                pos = i + 1
                if pos < len(input_ids) - 1:
                    keep_positions.append(pos)
        
        # 随机选择要 mask 的位置
        num_to_mask = int(len(keep_positions) * self.aux_mlm_prob)
        if num_to_mask > 0 and keep_positions:
            positions_to_mask = random.sample(keep_positions, min(num_to_mask, len(keep_positions)))
            for pos in positions_to_mask:
                labels[pos] = input_ids[pos].clone()
        
        return labels


def load_clean_sentences(
    file_path: str,
    file_format: str = "txt",
    text_field: str = "text",
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    加载干净句子用于在线数据增强
    
    Args:
        file_path: 文件路径
        file_format: 文件格式 ("txt", "json", "jsonl")
        text_field: JSON 中的文本字段名
        max_samples: 最大样本数（None 表示全部加载）
        
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
                            text = data.get(text_field) or data.get('text') or data.get('sentence') or data.get('content')
                            if text:
                                sentences.append(text)
                    except json.JSONDecodeError:
                        continue
                if max_samples and len(sentences) >= max_samples:
                    break
    
    return sentences

class TokenizedBinaryDataset(Dataset):
    """
    预计算 tokenize 的二进制数据集（最高效的数据加载方式）
    
    特点：
    - 数据在生成阶段就已完成 tokenize，训练时零 CPU 计算
    - 使用内存映射 (mmap) 直接访问二进制文件，避免文件 I/O 开销
    - 每个样本固定大小，支持 O(1) 随机访问
    - 适合千万级数据集训练
    
    文件格式：
    - .bin: 连续存储的二进制 tensor 数据
    - .idx: 头部信息 + 样本偏移数组
    
    使用方法：
    1. 先用 generate_tokenized_data.py 将 JSONL 转换为二进制格式
    2. 训练时使用此数据集类加载
    """
    
    def __init__(
        self,
        data_prefix: str,
        enable_aux_mlm: bool = True,
        aux_mlm_prob: float = 0.15,
    ):
        """
        Args:
            data_prefix: 数据文件前缀（不含 .bin/.idx 后缀）
            enable_aux_mlm: 是否启用辅助 MLM（训练时随机 mask）
            aux_mlm_prob: 辅助 MLM 的 mask 比例
        """
        self.data_prefix = data_prefix
        self.enable_aux_mlm = enable_aux_mlm
        self.aux_mlm_prob = aux_mlm_prob
        
        # 读取索引文件
        idx_file = f"{data_prefix}.idx"
        if not os.path.exists(idx_file):
            raise FileNotFoundError(f"Index file not found: {idx_file}")
        
        with open(idx_file, 'rb') as f:
            # 读取头部
            self.max_seq_length = struct.unpack('<I', f.read(4))[0]
            self.num_samples = struct.unpack('<Q', f.read(8))[0]
            
            # 读取偏移数组
            self.offsets = []
            for _ in range(self.num_samples):
                offset = struct.unpack('<Q', f.read(8))[0]
                self.offsets.append(offset)
        
        # 计算每个样本的字节大小
        # input_ids(int16) + attention_mask(int8) + op_labels(int8) + insert_labels(int8)
        # + template_input_ids(int16) + template_attention_mask(int8) + infill_labels(int16)
        self.sample_size = self.max_seq_length * 10
        
        # 内存映射二进制文件
        bin_file = f"{data_prefix}.bin"
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Binary file not found: {bin_file}")
        
        self._mmap = None
        self._bin_file = None
        self._open_mmap(bin_file)
        
        print(f"TokenizedBinaryDataset initialized:")
        print(f"  Samples: {self.num_samples:,}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  Sample size: {self.sample_size} bytes")
        print(f"  Binary file: {bin_file}")
    
    def _open_mmap(self, bin_file: str):
        """打开内存映射文件"""
        self._bin_file = open(bin_file, 'rb')
        self._mmap = mmap.mmap(
            self._bin_file.fileno(),
            0,  # 整个文件
            access=mmap.ACCESS_READ
        )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本（直接从内存映射读取，几乎零开销）"""
        if self._mmap is None:
            raise RuntimeError("Memory map not initialized")
        
        offset = self.offsets[idx]
        seq_len = self.max_seq_length
        
        # 从内存映射读取二进制数据
        data = self._mmap[offset:offset + self.sample_size]
        
        # 解析各个字段
        ptr = 0
        
        # input_ids: int16
        input_ids = np.frombuffer(data[ptr:ptr + seq_len * 2], dtype=np.int16).copy()
        ptr += seq_len * 2
        
        # attention_mask: int8
        attention_mask = np.frombuffer(data[ptr:ptr + seq_len], dtype=np.int8).copy()
        ptr += seq_len
        
        # op_labels: int8
        op_labels = np.frombuffer(data[ptr:ptr + seq_len], dtype=np.int8).copy()
        ptr += seq_len
        
        # insert_labels: int8
        insert_labels = np.frombuffer(data[ptr:ptr + seq_len], dtype=np.int8).copy()
        ptr += seq_len
        
        # template_input_ids: int16
        template_input_ids = np.frombuffer(data[ptr:ptr + seq_len * 2], dtype=np.int16).copy()
        ptr += seq_len * 2
        
        # template_attention_mask: int8
        template_attention_mask = np.frombuffer(data[ptr:ptr + seq_len], dtype=np.int8).copy()
        ptr += seq_len
        
        # infill_labels: int16
        infill_labels = np.frombuffer(data[ptr:ptr + seq_len * 2], dtype=np.int16).copy()
        
        # 转换为 PyTorch tensor
        input_ids_tensor = torch.from_numpy(input_ids.astype(np.int64))
        attention_mask_tensor = torch.from_numpy(attention_mask.astype(np.int64))
        op_labels_tensor = torch.from_numpy(op_labels.astype(np.int64))
        insert_labels_tensor = torch.from_numpy(insert_labels.astype(np.int64))
        template_input_ids_tensor = torch.from_numpy(template_input_ids.astype(np.int64))
        template_attention_mask_tensor = torch.from_numpy(template_attention_mask.astype(np.int64))
        infill_labels_tensor = torch.from_numpy(infill_labels.astype(np.int64))
        
        # 辅助 MLM（可选，训练时随机 mask KEEP 位置）
        aux_mlm_labels = None
        if self.enable_aux_mlm:
            aux_mlm_labels = self._create_aux_mlm_labels(input_ids_tensor, op_labels_tensor)
        
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'op_labels': op_labels_tensor,
            'insert_labels': insert_labels_tensor,
            'template_input_ids': template_input_ids_tensor,
            'template_attention_mask': template_attention_mask_tensor,
            'infill_labels': infill_labels_tensor,
            'aux_mlm_labels': aux_mlm_labels,
            'sample_id': f"tokenized_{idx}",
            'source_text': None,  # 二进制格式不存储原文
            'target_text': None,
        }
    
    def _create_aux_mlm_labels(
        self,
        input_ids: torch.Tensor,
        op_labels: torch.Tensor
    ) -> torch.Tensor:
        """创建辅助 MLM 标签（训练时随机 mask）"""
        labels = torch.full_like(input_ids, -100)
        
        # 找出 KEEP 位置（op_labels == 0，且不是 padding）
        # op_labels 中 -100 是 padding/special token
        keep_mask = (op_labels == 0)
        keep_positions = torch.where(keep_mask)[0].tolist()
        
        # 随机选择要 mask 的位置
        num_to_mask = int(len(keep_positions) * self.aux_mlm_prob)
        if num_to_mask > 0 and keep_positions:
            positions_to_mask = random.sample(keep_positions, min(num_to_mask, len(keep_positions)))
            for pos in positions_to_mask:
                labels[pos] = input_ids[pos].clone()
        
        return labels
    
    def __del__(self):
        """清理资源"""
        if self._mmap is not None:
            self._mmap.close()
        if self._bin_file is not None:
            self._bin_file.close()
    
    def __getstate__(self):
        """pickle 序列化时不保存 mmap"""
        state = self.__dict__.copy()
        state['_mmap'] = None
        state['_bin_file'] = None
        return state
    
    def __setstate__(self, state):
        """pickle 反序列化时重新打开 mmap"""
        self.__dict__.update(state)
        bin_file = f"{self.data_prefix}.bin"
        self._open_mmap(bin_file)