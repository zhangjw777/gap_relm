# Gap-ReLM 数据处理模块

# 文本预处理
from .preprocessor import TextPreprocessor, SentenceSplitter

# 字符对齐
from .alignment import CharacterAligner, EditOperation

# 标签生成
from .label_generator import LabelGenerator, GoldTemplateBuilder

# 数据集和加载器
from .dataset import (
    GapReLMDataset,
    GapReLMCollator,
    OnlineAugmentedDataset,
    LengthAdaptiveLambda,
    load_clean_sentences,
    LazyGapReLMDataset,  # 内存友好的惰性加载数据集
    TokenizedBinaryDataset,  # 最高效的预计算 tokenize 数据集
)
from .data_loader import (
    create_data_loaders, 
    create_online_data_loaders,
    create_tokenized_data_loaders,  # 预计算 tokenize 数据加载器
)

# 数据增强模块（独立功能）
from .confusion_set import (
    ConfusionSet,
    create_default_confusion_set,
    create_shape_only_confusion_set,
    create_pinyin_only_confusion_set,
    load_confusion_set,
)
from .protected_span import (
    ProtectedSpanDetector,
    ProtectedSpan,
    ProtectedType,
    create_default_detector,
    create_document_detector,
    create_minimal_detector,
)
from .error_generator import (
    ErrorGenerator,
    ErrorType,
    ErrorEdit,
    CorruptionResult,
    create_default_error_generator,
    create_balanced_error_generator,
    create_insert_focused_error_generator,
    create_conservative_error_generator,
    create_aggressive_error_generator,
)
from .augmentation import (
    AugmentationConfig,
    DataAugmentor,
    StaticDataGenerator,
    StaticSampleConfig,
)


__all__ = [
    # 文本预处理
    "TextPreprocessor",
    "SentenceSplitter",
    
    # 字符对齐
    "CharacterAligner",
    "EditOperation",
    
    # 标签生成
    "LabelGenerator",
    "GoldTemplateBuilder",
    
    # 数据集
    "GapReLMDataset",
    "GapReLMCollator",
    "OnlineAugmentedDataset",
    "LengthAdaptiveLambda",
    "load_clean_sentences",
    "create_data_loaders",
    "create_online_data_loaders",
    
    # 混淆集
    "ConfusionSet",
    "create_default_confusion_set",
    "create_shape_only_confusion_set",
    "create_pinyin_only_confusion_set",
    "load_confusion_set",
    
    # 保护约束
    "ProtectedSpanDetector",
    "ProtectedSpan",
    "ProtectedType",
    "create_default_detector",
    "create_document_detector",
    "create_minimal_detector",
    
    # 规则造错
    "ErrorGenerator",
    "ErrorType",
    "ErrorEdit",
    "CorruptionResult",
    "create_default_error_generator",
    "create_balanced_error_generator",
    "create_insert_focused_error_generator",
    "create_conservative_error_generator",
    "create_aggressive_error_generator",
    
    # 数据增强
    "AugmentationConfig",
    "DataAugmentor",
    "StaticDataGenerator",
    "StaticSampleConfig",
]
