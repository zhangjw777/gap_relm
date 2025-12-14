"""
Gap-ReLM 配置模块
包含所有可配置参数，支持消融实验开关控制
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class TrainingStage(Enum):
    """训练阶段枚举"""
    STAGE_A = "infiller_pretrain"      # Infiller预训练 (Gold Template Teacher Forcing)
    STAGE_B = "planner_train"          # Planner训练 (纯监督序列标注)
    STAGE_C = "joint_finetune"         # 联合微调 (解决训练/推理不一致)
    STAGE_D = "quality_enhance"        # 质量增强 (迭代精炼/Verifier)


@dataclass
class ModelConfig:
    """模型架构配置"""
    # 预训练模型
    pretrained_model_name: str = "hfl/chinese-macbert-base"
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    
    # 词表相关
    vocab_size: int = 21128
    max_seq_length: int = 512
    
    # Edit Planner 配置
    num_op_labels: int = 3              # KEEP=0, DELETE=1, REPLACE=2
    max_insert_num: int = 3             # 最大插入数量 K (可配置，默认<=3)
    
    # Dropout
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: float = 0.1
    
    # 是否共享编码器 (Planner和Infiller)
    share_encoder: bool = True


@dataclass
class DataConfig:
    """数据处理配置"""
    # 数据路径
    train_file: str = ""
    dev_file: str = ""
    test_file: str = ""
    
    # 混淆集路径 (用于数据增强)
    confusion_set_path: Optional[str] = None
    
    # 数据格式: "mucgec", "sighan", "ecspell", "custom", "clean" (干净语料)
    data_format: str = "mucgec"
    
    # 预处理选项
    normalize_text: bool = True          # 全角/半角统一
    split_sentence: bool = True          # 长句切分
    max_sentence_length: int = 128       # 切分后最大句长
    
    # 对齐算法: "levenshtein", "difflib"
    alignment_algorithm: str = "levenshtein"
    
    # ========== 数据增强配置（从干净语料生成训练数据）==========
    enable_augmentation: bool = False    # 是否启用数据增强（从干净语料生成）
    
    # 造错主参数（四个旋钮）
    aug_p_corrupt: float = 0.7           # 造错概率
    aug_lambda: float = 1.5              # 泊松参数（控制平均编辑数）
    aug_pi_skip: float = 0.2             # 删字概率 π_S
    aug_pi_multiply: float = 0.3         # 重复字概率 π_M
    aug_pi_replace: float = 0.5          # 错字概率 π_R
    aug_max_edits: int = 4               # 单句最大编辑数
    aug_max_insert_k: int = 3            # 单次最大重复数 K
    
    # 混淆集配置
    aug_use_default_shape: bool = True   # 使用默认形近字混淆集
    aug_use_default_pinyin: bool = True  # 使用默认音近字混淆集
    aug_custom_confusion_files: Optional[List[str]] = None  # 自定义混淆集文件
    
    # 保护约束配置
    aug_enable_protection: bool = True   # 启用保护约束
    aug_protect_doc_number: bool = True  # 保护文号
    aug_protect_date: bool = True        # 保护日期
    aug_protect_amount: bool = True      # 保护金额
    aug_protect_clause: bool = True      # 保护条款编号
    aug_protect_org: bool = True         # 保护机构名称
    aug_protect_law: bool = True         # 保护法规名称
    aug_protect_phrase: bool = True      # 保护固定格式
    aug_custom_protected_words: Optional[List[str]] = None  # 自定义保护词汇
    
    # 缓存
    cache_dir: str = "./cache"
    use_cache: bool = True
    
    # DataLoader配置
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    seed: int = 42
    num_epochs: int = 10
    batch_size: int = 32                 # 每GPU的batch size
    gradient_accumulation_steps: int = 1
    
    # 学习率
    learning_rate: float = 2e-5
    planner_lr: float = 5e-5             # Planner可以用不同学习率
    infiller_lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # 混合精度训练
    fp16: bool = True
    bf16: bool = False                   # 如果GPU支持BF16
    
    # 梯度裁剪
    max_grad_norm: float = 1.0
    
    # 分阶段训练配置
    current_stage: str = "infiller_pretrain"
    stage_a_epochs: int = 3              # Infiller预训练轮数
    stage_b_epochs: int = 3              # Planner训练轮数
    stage_c_epochs: int = 4              # 联合微调轮数
    
    # Scheduled Sampling (Stage C)
    scheduled_sampling_start: float = 0.0
    scheduled_sampling_end: float = 0.5
    
    # 损失函数权重
    lambda_infill: float = 1.0           # L_infill 权重
    mu_aux: float = 0.15                 # L_aux (辅助MLM) 权重
    
    # 保存和日志
    output_dir: str = "./outputs"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100
    eval_steps: int = 500
    
    # 早停
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001


@dataclass 
class F2OptimizationConfig:
    """F2优化配置 (召回优先)"""
    # 是否启用F2优化
    enable_f2_optimization: bool = True
    
    # 代价敏感权重
    op_delete_weight: float = 3.0        # DELETE 类别权重 alpha
    op_replace_weight: float = 2.0       # REPLACE 类别权重
    op_keep_weight: float = 1.0          # KEEP 类别权重
    
    insert_positive_weight: float = 5.0  # k>0 时的权重 beta
    insert_zero_weight: float = 1.0      # k=0 时的权重
    
    # 阈值校准
    enable_threshold_calibration: bool = True
    delete_threshold: float = 0.5        # DELETE 阈值 τ_del
    insert_threshold: float = 0.5        # INSERT 阈值 τ_ins
    
    # 风险约束 (防止过纠)
    max_insert_per_sentence: int = 6     # 每句最大插入总数 B
    max_insert_ratio: float = 0.1        # 最大插入比例 α (相对于原句长度)


@dataclass
class AblationConfig:
    """消融实验配置 - 控制各模块开关"""
    # 核心模块开关
    enable_gap: bool = True              # 启用Gap (关闭则退化为原始ReLM)
    enable_insert: bool = True           # 启用插入操作
    enable_delete: bool = True           # 启用删除操作
    
    # P-Tuning 配置（论文中对性能有贡献）
    enable_ptuning: bool = True          # 启用 P-Tuning（默认开启）
    ptuning_prompt_length: int = 10      # Prompt 长度（虚拟 token 数量）
    ptuning_use_lstm: bool = True        # 是否使用 LSTM 编码 prompt
    ptuning_use_mlp: bool = True         # 是否使用 MLP 编码 prompt
    ptuning_shared: bool = False         # 是否 Planner/Infiller 共享 prompt（False=各自独立）
    
    # 辅助模块开关
    enable_aux_mlm: bool = True          # 启用辅助MLM任务 (L_aux)
    aux_mlm_prob: float = 0.15           # 辅助MLM的mask比例
    
    # 可选增强模块
    enable_iterative_refinement: bool = False  # 迭代精炼 (Mask-Predict风格)
    refinement_rounds: int = 2                 # 精炼轮数 R
    refinement_mask_ratio: float = 0.15        # 每轮mask比例
    
    enable_verifier: bool = False        # 启用Verifier模块
    verifier_threshold: float = 0.5      # Verifier接受阈值
    
    # Scheduled Sampling (Stage C)
    enable_scheduled_sampling: bool = True
    enable_noisy_template: bool = False  # 噪声模板增强
    noisy_template_prob: float = 0.1     # 噪声概率
    
    # 术语保护 (公文场景)
    enable_term_protection: bool = False
    term_dict_path: Optional[str] = None


@dataclass
class InferenceConfig:
    """推理配置"""
    # 基础推理参数
    batch_size: int = 64
    max_seq_length: int = 512
    
    # 迭代精炼 (推理时)
    use_iterative_refinement: bool = False
    refinement_rounds: int = 2
    refinement_mask_ratio: float = 0.15
    
    # 解码策略
    use_confidence_threshold: bool = True
    min_confidence: float = 0.5
    
    # Verifier (推理时)
    use_verifier: bool = False
    verifier_model_path: Optional[str] = None
    
    # 输出格式
    output_format: str = "json"          # "json", "txt", "parallel"
    output_edits: bool = True            # 是否输出编辑操作


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    # DDP配置
    use_ddp: bool = True
    local_rank: int = -1
    world_size: int = 1
    
    # 通信后端
    backend: str = "nccl"                # "nccl", "gloo"
    
    # 同步BN
    sync_bn: bool = False
    
    # 找到未使用的参数（P-Tuning等）
    find_unused_parameters: bool = True


@dataclass
class GapReLMConfig:
    """Gap-ReLM 总配置"""
    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    f2_optimization: F2OptimizationConfig = field(default_factory=F2OptimizationConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # 实验名称
    experiment_name: str = "gap_relm_base"
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    
    # 日志级别
    log_level: str = "INFO"
    
    def save(self, path: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "GapReLMConfig":
        """从JSON文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "training": asdict(self.training),
            "f2_optimization": asdict(self.f2_optimization),
            "ablation": asdict(self.ablation),
            "inference": asdict(self.inference),
            "distributed": asdict(self.distributed),
            "experiment_name": self.experiment_name,
            "use_tensorboard": self.use_tensorboard,
            "tensorboard_dir": self.tensorboard_dir,
            "log_level": self.log_level,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GapReLMConfig":
        """从字典创建配置"""
        config = cls()
        
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "f2_optimization" in config_dict:
            config.f2_optimization = F2OptimizationConfig(**config_dict["f2_optimization"])
        if "ablation" in config_dict:
            config.ablation = AblationConfig(**config_dict["ablation"])
        if "inference" in config_dict:
            config.inference = InferenceConfig(**config_dict["inference"])
        if "distributed" in config_dict:
            config.distributed = DistributedConfig(**config_dict["distributed"])
        
        config.experiment_name = config_dict.get("experiment_name", "gap_relm_base")
        config.use_tensorboard = config_dict.get("use_tensorboard", True)
        config.tensorboard_dir = config_dict.get("tensorboard_dir", "./runs")
        config.log_level = config_dict.get("log_level", "INFO")
        
        return config
    
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        for key, value in vars(args).items():
            if value is not None:
                # 尝试更新各子配置
                for sub_config_name in ["model", "data", "training", "f2_optimization", 
                                        "ablation", "inference", "distributed"]:
                    sub_config = getattr(self, sub_config_name)
                    if hasattr(sub_config, key):
                        setattr(sub_config, key, value)
                
                # 更新顶层配置
                if hasattr(self, key):
                    setattr(self, key, value)


def get_default_config() -> GapReLMConfig:
    """获取默认配置"""
    return GapReLMConfig()


def get_ablation_config_no_gap() -> GapReLMConfig:
    """消融实验: 无Gap (退化为ReLM)"""
    config = GapReLMConfig()
    config.experiment_name = "ablation_no_gap"
    config.ablation.enable_gap = False
    config.ablation.enable_insert = False
    config.ablation.enable_delete = False
    return config


def get_ablation_config_no_insert() -> GapReLMConfig:
    """消融实验: 只删除不插入"""
    config = GapReLMConfig()
    config.experiment_name = "ablation_no_insert"
    config.ablation.enable_insert = False
    return config


def get_ablation_config_no_delete() -> GapReLMConfig:
    """消融实验: 只插入不删除"""
    config = GapReLMConfig()
    config.experiment_name = "ablation_no_delete"
    config.ablation.enable_delete = False
    return config


def get_ablation_config_no_aux_mlm() -> GapReLMConfig:
    """消融实验: 无辅助MLM"""
    config = GapReLMConfig()
    config.experiment_name = "ablation_no_aux_mlm"
    config.ablation.enable_aux_mlm = False
    return config


def get_ablation_config_with_refinement() -> GapReLMConfig:
    """启用迭代精炼"""
    config = GapReLMConfig()
    config.experiment_name = "with_iterative_refinement"
    config.ablation.enable_iterative_refinement = True
    config.ablation.refinement_rounds = 2
    return config


def get_ablation_config_with_verifier() -> GapReLMConfig:
    """启用Verifier"""
    config = GapReLMConfig()
    config.experiment_name = "with_verifier"
    config.ablation.enable_verifier = True
    return config


def get_ablation_config_no_f2() -> GapReLMConfig:
    """消融实验: 无F2优化"""
    config = GapReLMConfig()
    config.experiment_name = "ablation_no_f2"
    config.f2_optimization.enable_f2_optimization = False
    config.f2_optimization.enable_threshold_calibration = False
    return config


def get_ablation_config_no_ptuning() -> GapReLMConfig:
    """消融实验: 无P-Tuning"""
    config = GapReLMConfig()
    config.experiment_name = "ablation_no_ptuning"
    config.ablation.enable_ptuning = False
    return config


def get_ablation_config_ptuning_no_lstm() -> GapReLMConfig:
    """消融实验: P-Tuning 不使用 LSTM"""
    config = GapReLMConfig()
    config.experiment_name = "ablation_ptuning_no_lstm"
    config.ablation.enable_ptuning = True
    config.ablation.ptuning_use_lstm = False
    return config


def get_ablation_config_ptuning_shared() -> GapReLMConfig:
    """消融实验: P-Tuning 共享 prompt"""
    config = GapReLMConfig()
    config.experiment_name = "ablation_ptuning_shared"
    config.ablation.enable_ptuning = True
    config.ablation.ptuning_shared = True
    return config


# 预定义配置字典
PREDEFINED_CONFIGS = {
    "default": get_default_config,
    "no_gap": get_ablation_config_no_gap,
    "no_insert": get_ablation_config_no_insert,
    "no_delete": get_ablation_config_no_delete,
    "no_aux_mlm": get_ablation_config_no_aux_mlm,
    "with_refinement": get_ablation_config_with_refinement,
    "with_verifier": get_ablation_config_with_verifier,
    "no_f2": get_ablation_config_no_f2,
    "no_ptuning": get_ablation_config_no_ptuning,
    "ptuning_no_lstm": get_ablation_config_ptuning_no_lstm,
    "ptuning_shared": get_ablation_config_ptuning_shared,
}


def get_config(name: str = "default") -> GapReLMConfig:
    """根据名称获取预定义配置"""
    if name in PREDEFINED_CONFIGS:
        return PREDEFINED_CONFIGS[name]()
    else:
        raise ValueError(f"Unknown config name: {name}. Available: {list(PREDEFINED_CONFIGS.keys())}")
