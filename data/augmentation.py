"""
数据增强管道
整合混淆集、造错器、保护约束，从干净句子生成训练数据
"""

import os
import json
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from .confusion_set import ConfusionSet
from .protected_span import ProtectedSpanDetector
from .error_generator import ErrorGenerator, CorruptionResult, ErrorType, ErrorEdit


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    # 造错概率（在线动态增强用）
    p_corrupt: float = 0.7
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
    数据增强器（在线动态增强用）
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
        
        self.confusion_set = ConfusionSet(
            use_default_shape=self.config.use_default_shape_confusion,
            use_default_pinyin=self.config.use_default_pinyin_confusion,
            custom_confusion_files=self.config.custom_confusion_files or None,
        )
        
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
        """对单个句子进行增强"""
        return self.error_generator.corrupt(sentence)
    
    def augment_batch(self, sentences: List[str], show_progress: bool = False) -> List[CorruptionResult]:
        """批量增强"""
        return self.error_generator.corrupt_batch(sentences, show_progress)
    
    def update_params(self, **kwargs):
        """更新造错参数"""
        self.error_generator.set_params(**kwargs)
    
    def get_stats(self, results: List[CorruptionResult]) -> Dict[str, Any]:
        """获取统计信息"""
        return self.error_generator.stats(results)


@dataclass
class StaticSampleConfig:
    """静态数据生成配置
    
    每个干净句子生成：1个正例 + num_negative个不同类型的负例
    每个负例只有1个错误
    """
    # 每个句子生成的负例数量
    num_negative_per_sentence: int = 2
    
    # 错误类型概率（用于选择错误类型）
    pi_skip: float = 0.2
    pi_multiply: float = 0.3
    pi_replace: float = 0.5
    
    # 混淆集配置
    use_default_shape_confusion: bool = True
    use_default_pinyin_confusion: bool = True
    custom_confusion_files: List[str] = field(default_factory=list)
    
    # 保护约束配置
    enable_protection: bool = True
    
    # 其他选项
    min_sentence_length: int = 5
    skip_punct: bool = True
    max_insert_k: int = 3
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_negative_per_sentence': self.num_negative_per_sentence,
            'pi_skip': self.pi_skip,
            'pi_multiply': self.pi_multiply,
            'pi_replace': self.pi_replace,
            'enable_protection': self.enable_protection,
            'min_sentence_length': self.min_sentence_length,
            'max_insert_k': self.max_insert_k,
            'seed': self.seed,
        }


class StaticDataGenerator:
    """
    静态训练数据生成器
    
    特点：
    - 每个句子生成1个正例 + N个负例
    - 每个负例只有1个错误，且错误类型不同
    - 先划分句子再生成样本，避免同源句泄露
    """
    
    ERROR_TYPES = [ErrorType.SKIP, ErrorType.MULTIPLY, ErrorType.REPLACE]
    
    def __init__(self, config: Optional[StaticSampleConfig] = None):
        self.config = config or StaticSampleConfig()
        
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
            enable_doc_number=self.config.enable_protection,
            enable_date=self.config.enable_protection,
            enable_amount=self.config.enable_protection,
            enable_clause_number=self.config.enable_protection,
            enable_organization=self.config.enable_protection,
            enable_law_name=self.config.enable_protection,
            enable_fixed_phrase=self.config.enable_protection,
        )
        
        # 创建造错器
        self.error_generator = ErrorGenerator(
            p_corrupt=1.0,  # 总是造错
            lambda_=1.0,
            pi_skip=self.config.pi_skip,
            pi_multiply=self.config.pi_multiply,
            pi_replace=self.config.pi_replace,
            max_edits_per_sent=1,
            max_insert_k=self.config.max_insert_k,
            confusion_set=self.confusion_set,
            protected_detector=self.protected_detector,
            enable_protection=self.config.enable_protection,
            min_sentence_length=self.config.min_sentence_length,
            skip_punct=self.config.skip_punct,
            seed=self.config.seed,
        )
        
        # 归一化错误类型概率
        total = self.config.pi_skip + self.config.pi_multiply + self.config.pi_replace
        self.type_probs = {
            ErrorType.SKIP: self.config.pi_skip / total,
            ErrorType.MULTIPLY: self.config.pi_multiply / total,
            ErrorType.REPLACE: self.config.pi_replace / total,
        }
    
    def _sample_error_types(self, n: int, rng: random.Random) -> List[ErrorType]:
        """按概率采样n个不同的错误类型"""
        if n >= len(self.ERROR_TYPES):
            types = self.ERROR_TYPES.copy()
            rng.shuffle(types)
            return types
        
        # 按概率加权采样不重复的类型
        selected = []
        available = list(self.ERROR_TYPES)
        
        for _ in range(n):
            weights = [self.type_probs[t] for t in available]
            total = sum(weights)
            weights = [w / total for w in weights]
            
            r = rng.random()
            cumsum = 0.0
            for i, w in enumerate(weights):
                cumsum += w
                if r < cumsum:
                    selected.append(available.pop(i))
                    break
        
        return selected
    
    def _generate_labels_from_edit(
        self,
        source: str,          # 错误句子
        target: str,          # 正确句子
        edit: ErrorEdit,      # 造错时的编辑操作
    ) -> Dict[str, Any]:
        """
        从造错的编辑操作直接生成训练标签
        
        注意：造错是 target → source（正确→错误）
        纠错是 source → target（错误→正确）
        
        造错操作与纠错标签的映射：
        - SKIP（删字）: 正确句子中某字被删除 → 纠错需要在source中插入
        - MULTIPLY（重复字）: 正确句子中某字被重复 → 纠错需要删除source中多余的字
        - REPLACE（替换）: 正确句子中某字被替换 → 纠错需要替换source中的错字
        """
        n = len(source)
        op_labels = [0] * n       # 默认 KEEP=0
        insert_labels = [0] * n   # 默认不插入
        
        template_tokens = []
        gold_tokens = []
        mask_positions = []
        
        edit_pos = edit.position  # 在原始正确句子(target)中的位置
        
        if edit.error_type == ErrorType.SKIP:
            # SKIP：target中位置edit_pos的字被删除了
            # source比target短1个字
            # 纠错：需要在source的edit_pos位置**之前**插入被删除的字
            # 使用Left-Association规则：插入归属到左侧字符
            if edit_pos > 0:
                # 插入归属到source[edit_pos-1]（因为target[edit_pos]被删了，source中edit_pos-1后面需要插入）
                insert_labels[edit_pos - 1] = 1
                # Gold Template: 前edit_pos个字 + [MASK] + 后面的字
                template_tokens = list(source[:edit_pos]) + ['[MASK]'] + list(source[edit_pos:])
                gold_tokens = [edit.original_char]  # 被删除的字
                mask_positions = [edit_pos]
            else:
                # 句首插入：归属到第一个字符
                insert_labels[0] = 1
                template_tokens = ['[MASK]'] + list(source)
                gold_tokens = [edit.original_char]
                mask_positions = [0]
        
        elif edit.error_type == ErrorType.MULTIPLY:
            # MULTIPLY：target中位置edit_pos的字后面被重复了k次
            # source比target长k个字
            # 错误字符串 = edit.error_char（重复的字符串，长度可能>1）
            k = len(edit.error_char)  # 重复的次数
            # source中：edit_pos是原字符，edit_pos+1到edit_pos+k是重复的字符
            # 纠错：需要删除source中edit_pos+1到edit_pos+k的字符
            for i in range(k):
                pos_in_source = edit_pos + 1 + i
                if pos_in_source < n:
                    op_labels[pos_in_source] = 1  # DELETE
            # Gold Template: 删除的字不出现
            for i in range(n):
                if op_labels[i] != 1:  # 不是DELETE的字
                    template_tokens.append(source[i])
        
        elif edit.error_type == ErrorType.REPLACE:
            # REPLACE：target中位置edit_pos的字被替换成了错误字符
            # source和target长度相同
            # source[edit_pos] = edit.error_char (错误的字)
            # target[edit_pos] = edit.original_char (正确的字)
            # 纠错：需要替换source[edit_pos]
            op_labels[edit_pos] = 2  # REPLACE
            # Gold Template: 替换位置放[MASK]
            for i in range(n):
                if i == edit_pos:
                    template_tokens.append('[MASK]')
                    gold_tokens.append(edit.original_char)  # 正确的字
                    mask_positions.append(len(template_tokens) - 1)
                else:
                    template_tokens.append(source[i])
        
        # 如果template还是空（不应该发生），设为source
        if not template_tokens:
            template_tokens = list(source)
        
        return {
            'op_labels': op_labels,
            'insert_labels': insert_labels,
            'template_tokens': template_tokens,
            'gold_tokens': gold_tokens,
            'mask_positions': mask_positions,
        }
    
    def _generate_positive_labels(self, sentence: str) -> Dict[str, Any]:
        """为正例生成标签（所有位置都是KEEP，无MASK）"""
        n = len(sentence)
        return {
            'op_labels': [0] * n,
            'insert_labels': [0] * n,
            'template_tokens': list(sentence),
            'gold_tokens': [],
            'mask_positions': [],
        }

    def generate_samples_for_sentence(
        self, 
        clean_sentence: str, 
        rng: Optional[random.Random] = None
    ) -> List[Dict[str, Any]]:
        """为单个句子生成样本：1正例 + N负例，包含预计算的标签"""
        _random = rng if rng is not None else random
        samples = []
        
        # 正例（source == target）
        pos_labels = self._generate_positive_labels(clean_sentence)
        samples.append({
            'source': clean_sentence,
            'target': clean_sentence,
            **pos_labels,  # 包含op_labels, insert_labels等
        })
        
        # 采样不同的错误类型
        selected_types = self._sample_error_types(
            self.config.num_negative_per_sentence, _random
        )
        
        # 生成负例
        for error_type in selected_types:
            result = self.error_generator.corrupt_with_type(
                clean_sentence, error_type, rng=_random
            )
            
            if result.is_corrupted and result.edits:
                edit = result.edits[0]  # 只有一个错误
                labels = self._generate_labels_from_edit(
                    source=result.corrupted,
                    target=result.original,
                    edit=edit,
                )
                samples.append({
                    'source': result.corrupted,
                    'target': result.original,
                    **labels,
                })
            else:
                # 回退到其他类型
                for fallback in self.ERROR_TYPES:
                    if fallback != error_type:
                        result = self.error_generator.corrupt_with_type(
                            clean_sentence, fallback, rng=_random
                        )
                        if result.is_corrupted and result.edits:
                            edit = result.edits[0]
                            labels = self._generate_labels_from_edit(
                                source=result.corrupted,
                                target=result.original,
                                edit=edit,
                            )
                            samples.append({
                                'source': result.corrupted,
                                'target': result.original,
                                **labels,
                            })
                            break
        
        return samples
    
    def generate_and_save(
        self,
        clean_sentences: List[str],
        output_dir: str,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, str]:
        """生成并保存数据集"""
        if seed is not None:
            random.seed(seed)
        rng = random.Random(seed) if seed is not None else random
        
        # 归一化比例
        total = train_ratio + dev_ratio + test_ratio
        train_ratio, dev_ratio, test_ratio = train_ratio/total, dev_ratio/total, test_ratio/total
        
        # 句子层面划分
        sentences = clean_sentences.copy()
        if shuffle:
            rng.shuffle(sentences)
        
        n = len(sentences)
        n_train, n_dev = int(n * train_ratio), int(n * dev_ratio)
        
        train_sents = sentences[:n_train]
        dev_sents = sentences[n_train:n_train + n_dev]
        test_sents = sentences[n_train + n_dev:]
        
        print(f"句子划分: train={len(train_sents)}, dev={len(dev_sents)}, test={len(test_sents)}")
        print(f"每句生成: 1正例 + {self.config.num_negative_per_sentence}负例")
        
        # 生成样本
        train_samples = self._generate_split(train_sents, rng, show_progress, "train")
        dev_samples = self._generate_split(dev_sents, rng, show_progress, "dev")
        test_samples = self._generate_split(test_sents, rng, show_progress, "test")
        
        # 样本层面打乱
        if shuffle:
            rng.shuffle(train_samples)
            rng.shuffle(dev_samples)
            rng.shuffle(test_samples)
        
        # 保存
        os.makedirs(output_dir, exist_ok=True)
        output_files = {
            'train': self._save(train_samples, os.path.join(output_dir, "train.jsonl")),
            'dev': self._save(dev_samples, os.path.join(output_dir, "dev.jsonl")),
            'test': self._save(test_samples, os.path.join(output_dir, "test.jsonl")),
        }
        
        # 统计信息
        stats = {
            'config': self.config.to_dict(),
            'sentence_split': {'train': len(train_sents), 'dev': len(dev_sents), 'test': len(test_sents)},
            'sample_stats': {
                'train': self._stats(train_samples),
                'dev': self._stats(dev_samples),
                'test': self._stats(test_samples),
            },
        }
        stats_file = os.path.join(output_dir, "stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        output_files['stats'] = stats_file
        
        print(f"\n完成！train={len(train_samples)}, dev={len(dev_samples)}, test={len(test_samples)}")
        return output_files
    
    def _generate_split(self, sentences, rng, show_progress, name):
        samples = []
        iterator = sentences
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(sentences, desc=f"生成{name}")
            except ImportError:
                pass
        for sent in iterator:
            samples.extend(self.generate_samples_for_sentence(sent, rng))
        return samples
    
    def _save(self, samples, path):
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"保存 {len(samples)} 样本到 {path}")
        return path
    
    def _stats(self, samples):
        total = len(samples)
        positive = sum(1 for s in samples if not s.get('gold_tokens'))
        negative = total - positive
        
        # 统计错误类型（通过label推断）
        type_counts = {'S': 0, 'M': 0, 'R': 0}
        for s in samples:
            op_labels = s.get('op_labels', [])
            insert_labels = s.get('insert_labels', [])
            
            if any(ins > 0 for ins in insert_labels):
                type_counts['S'] += 1  # 有插入标签 → 原来是SKIP造错
            elif 1 in op_labels:
                type_counts['M'] += 1  # 有DELETE标签 → 原来是MULTIPLY造错
            elif 2 in op_labels:
                type_counts['R'] += 1  # 有REPLACE标签 → 原来是REPLACE造错
        
        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'error_types': type_counts,
        }
    
    @staticmethod
    def load_sentences(file_path: str, file_format: str = "txt") -> List[str]:
        """加载句子文件"""
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_format == "txt":
                sentences = [line.strip() for line in f if line.strip()]
            elif file_format == "jsonl":
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if isinstance(data, str):
                            sentences.append(data)
                        elif isinstance(data, dict):
                            sentences.append(data.get('text', data.get('source', '')))
        return sentences
