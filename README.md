# Gap-ReLM: 可变长重述式中文纠错框架 - 详细使用手册

基于 ReLM (Rephrasing Language Model) 改进，支持插入/删除操作的中文纠错模型。

## 目录

1. [项目概述](#项目概述)
2. [安装配置](#安装配置)
3. [配置系统详解](#配置系统详解)
4. [数据准备](#数据准备)
5. [训练指南](#训练指南)
   - [完整训练流程](#完整训练流程)
   - [单独训练Planner](#单独训练planner)
   - [单独训练Infiller](#单独训练infiller)
   - [多任务联合训练](#多任务联合训练)
6. [数据增强与混淆集](#数据增强与混淆集)
7. [推理使用](#推理使用)
8. [消融实验](#消融实验)
9. [常见问题](#常见问题)

---

## 项目概述

### 核心特点

- **可变长纠错**：在 ReLM 的基础上扩展，支持 Missing (少字) 和 Redundant (多字) 错误类型
- **F2 优化**：召回优先的训练策略，适合高召回场景
- **分阶段训练**：Infiller 预训练 → Planner 训练 → 联合微调
- **模块化设计**：通过配置文件控制各组件开关，支持消融实验
- **多卡训练**：支持 DDP 分布式训练
- **混合精度**：支持 FP16/BF16 训练
- **数据增强**：从干净语料自动生成训练数据，支持自定义混淆集

### 模型架构

```
Input X → Encoder → Edit Planner (Op Head + Insert Head)
                          ↓
                   Template Builder
                          ↓
        [CLS] X [SEP] Template [SEP] → Infiller → Output Y
```

### 核心组件

1. **共享编码器** (MacBERT-Base)
2. **Edit Planner**：预测 KEEP/DELETE/REPLACE 和插入数量
3. **Template Builder**：根据规划构建变长模板
4. **ReLM Infiller**：填充模板中的 [MASK]
5. **Verifier** (可选)：降低过纠风险
6. **迭代精炼** (可选)：多轮优化低置信度 token

---

## 安装配置

### 环境要求

```bash
pip install torch transformers python-Levenshtein tensorboard tqdm
```

### 项目结构

```
gap_relm/
├── __init__.py
├── config.py              # 配置模块
├── inference.py           # 推理模块
├── data/                  # 数据处理
│   ├── alignment.py       # 字符级对齐
│   ├── augmentation.py    # 数据增强管道
│   ├── confusion_set.py   # 混淆集管理
│   ├── error_generator.py # 规则造错器
│   ├── protected_span.py  # 保护约束
│   ├── label_generator.py # 标签生成
│   ├── dataset.py         # PyTorch Dataset
│   └── data_loader.py     # DataLoader 工厂
├── models/                # 模型架构
│   ├── gap_relm.py        # 主模型
│   ├── planner.py         # Edit Planner
│   ├── infiller.py        # ReLM Infiller
│   ├── template_builder.py # Template Builder
│   └── verifier.py        # Verifier
├── trainers/              # 训练器
│   ├── trainer.py         # 训练器
│   └── scheduler.py       # 调度器
└── scripts/               # 启动脚本
    ├── train.py           # 训练脚本
    ├── predict.py         # 推理脚本
    └── run_ddp.sh         # DDP 多卡训练脚本
```

---

## 配置系统详解

Gap-ReLM 使用分层配置系统，通过 `config.py` 中的 `GapReLMConfig` 管理所有参数。

### 配置结构

```python
GapReLMConfig
├── model: ModelConfig           # 模型架构配置
├── data: DataConfig            # 数据处理配置
├── training: TrainingConfig    # 训练配置
├── f2_optimization: F2OptimizationConfig  # F2优化配置
├── ablation: AblationConfig    # 消融实验配置
├── inference: InferenceConfig  # 推理配置
└── distributed: DistributedConfig  # 分布式配置
```

### 核心配置参数说明

#### 1. ModelConfig - 模型架构配置

```python
pretrained_model_name: str = "hfl/chinese-macbert-base"  # 预训练模型
max_seq_length: int = 512                                 # 最大序列长度
num_op_labels: int = 3                                    # 操作类型数 (KEEP/DELETE/REPLACE)
max_insert_num: int = 3                                   # 最大插入数量 K
share_encoder: bool = True                                # Planner和Infiller是否共享编码器
```

#### 2. DataConfig - 数据处理配置

```python
train_file: str                          # 训练文件路径
dev_file: str                            # 验证文件路径
data_format: str = "mucgec"              # 数据格式: mucgec/sighan/ecspell/parallel
alignment_algorithm: str = "levenshtein" # 对齐算法
max_sentence_length: int = 128           # 句子最大长度

# 数据增强配置
enable_augmentation: bool = False        # 是否启用数据增强
confusion_set_path: Optional[str] = None # 混淆集路径
aug_p_corrupt: float = 0.7               # 造错概率
aug_lambda: float = 1.5                  # 泊松参数（控制平均编辑数）
aug_pi_skip: float = 0.2                 # 删字概率
aug_pi_multiply: float = 0.3             # 重复字概率
aug_pi_replace: float = 0.5              # 错字概率
```

#### 3. TrainingConfig - 训练配置

```python
num_epochs: int = 10                     # 训练轮数
batch_size: int = 32                     # 批大小
learning_rate: float = 2e-5              # 学习率
planner_lr: float = 5e-5                 # Planner专用学习率
infiller_lr: float = 2e-5                # Infiller专用学习率
gradient_accumulation_steps: int = 1     # 梯度累积步数
fp16: bool = True                        # 是否使用FP16
bf16: bool = False                       # 是否使用BF16

# 分阶段训练配置
current_stage: str = "infiller_pretrain" # 当前训练阶段
stage_a_epochs: int = 3                  # Stage A: Infiller预训练轮数
stage_b_epochs: int = 3                  # Stage B: Planner训练轮数
stage_c_epochs: int = 4                  # Stage C: 联合微调轮数
```

#### 4. AblationConfig - 消融实验配置

```python
enable_gap: bool = True                  # 启用Gap (关闭则退化为原始ReLM)
enable_insert: bool = True               # 启用插入操作
enable_delete: bool = True               # 启用删除操作
enable_aux_mlm: bool = True              # 启用辅助MLM任务
enable_iterative_refinement: bool = False # 启用迭代精炼
enable_verifier: bool = False            # 启用Verifier模块
```

### 使用预定义配置

项目提供多种预定义配置：

```python
# 默认配置
config = get_config("default")

# 消融实验: 无Gap (退化为ReLM)
config = get_config("no_gap")

# 消融实验: 只删除不插入
config = get_config("no_insert")

# 消融实验: 只插入不删除
config = get_config("no_delete")

# 启用迭代精炼
config = get_config("with_refinement")

# 启用Verifier
config = get_config("with_verifier")
```

### 从文件加载/保存配置

```python
# 保存配置
config = GapReLMConfig()
config.save("configs/my_config.json")

# 加载配置
config = GapReLMConfig.load("configs/my_config.json")
```

---

## 数据准备

### 支持的数据格式

#### 1. MuCGEC 格式 (推荐)

JSON Lines 格式，每行一个样本：
```json
{"source": "我今天很开兴", "target": "我今天很开心"}
{"source": "这是一个测是", "target": "这是一个测试"}
```

#### 2. Parallel 格式

制表符分隔：
```
我今天很开兴	我今天很开心
这是一个测是	这是一个测试
```

#### 3. SIGHAN/ECSpell 格式

制表符分隔，支持错误位置标注：
```
1	我今天很开兴	我今天很开心	5
2	这是一个测是	这是一个测试	6
```

### 数据目录组织

```
data/
├── mucgec/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── sighan/
│   ├── train.txt
│   └── test.txt
└── custom/
    └── my_data.jsonl
```

---

## 训练指南

### 完整训练流程

Gap-ReLM 采用**三阶段训练策略**：

```
Stage A: Infiller 预训练 (Gold Template Teacher Forcing)
    ↓
Stage B: Planner 训练 (纯监督序列标注)
    ↓
Stage C: 联合微调 (解决训练/推理不一致)
```

#### 完整三阶段训练

```bash
# 自动完成三阶段训练
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/gap_relm_full \
    --experiment_name gap_relm_full \
    --training_stage infiller_pretrain \
    --stage_a_epochs 3 \
    --stage_b_epochs 3 \
    --stage_c_epochs 4 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --planner_lr 5e-5 \
    --infiller_lr 2e-5 \
    --fp16
```

### 单独训练Planner

如果你只想训练 Planner 模块（例如已有预训练的 Infiller）：

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/planner_only \
    --experiment_name planner_only \
    --training_stage planner_train \
    --stage_b_epochs 5 \
    --batch_size 32 \
    --planner_lr 5e-5 \
    --fp16
```

**配置要点：**
- `--training_stage planner_train`: 指定为 Planner 训练阶段
- `--stage_b_epochs 5`: 设置 Planner 训练轮数
- `--planner_lr 5e-5`: Planner 通常需要更大的学习率

**Python 代码方式：**

```python
from gap_relm.config import GapReLMConfig
from gap_relm.models import GapReLMModel
from gap_relm.trainers import GapReLMTrainer
from gap_relm.data import create_data_loaders

# 创建配置
config = GapReLMConfig()
config.training.current_stage = "planner_train"
config.training.stage_b_epochs = 5
config.training.planner_lr = 5e-5

# 创建数据加载器
train_loader, dev_loader, _, tokenizer = create_data_loaders(
    train_file="data/mucgec/train.jsonl",
    dev_file="data/mucgec/dev.jsonl",
    tokenizer_name="hfl/chinese-macbert-base",
    batch_size=32
)

# 创建模型
model = GapReLMModel(config)

# 创建训练器
trainer = GapReLMTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    dev_loader=dev_loader,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()
```

### 单独训练Infiller

如果你只想训练 Infiller 模块：

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/infiller_only \
    --experiment_name infiller_only \
    --training_stage infiller_pretrain \
    --stage_a_epochs 5 \
    --batch_size 32 \
    --infiller_lr 2e-5 \
    --fp16
```

**配置要点：**
- `--training_stage infiller_pretrain`: 指定为 Infiller 预训练阶段
- `--stage_a_epochs 5`: 设置 Infiller 训练轮数
- 此阶段 Planner 会被冻结，使用 Gold Template 进行 Teacher Forcing

### 多任务联合训练

联合训练 Planner 和 Infiller（Stage C）：

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/joint_finetune \
    --experiment_name joint_finetune \
    --training_stage joint_finetune \
    --stage_c_epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --planner_lr 3e-5 \
    --infiller_lr 2e-5 \
    --fp16
```

**配置要点：**
- `--training_stage joint_finetune`: 联合微调阶段
- 可以为 Planner 和 Infiller 设置不同的学习率
- 此阶段会使用 Scheduled Sampling 来解决训练/推理不一致问题

**启用 Scheduled Sampling：**

```python
config = GapReLMConfig()
config.training.current_stage = "joint_finetune"
config.ablation.enable_scheduled_sampling = True
config.training.scheduled_sampling_start = 0.0  # 初始使用 Gold Template 的比例
config.training.scheduled_sampling_end = 0.5    # 最终使用 Gold Template 的比例
```

### 多卡 DDP 训练

使用 `torchrun` 进行多卡分布式训练：

```bash
# 双卡训练
torchrun --nproc_per_node=2 gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/gap_relm_ddp \
    --experiment_name gap_relm_ddp \
    --batch_size 32 \
    --num_epochs 10 \
    --fp16

# 四卡训练
torchrun --nproc_per_node=4 gap_relm/scripts/train.py \
    --batch_size 16 \
    --fp16
```

或使用脚本：

```bash
bash gap_relm/scripts/run_ddp.sh \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --num_gpus 2
```

### 推理

```bash
python gap_relm/scripts/predict.py \
    --model_path outputs/gap_relm/best_stage_c \
    --input test.txt \
    --output results.json \
    --output_format json \
    --batch_size 64
```

### Python API

```python
from gap_relm.inference import GapReLMPipeline

# 加载模型
pipeline = GapReLMPipeline("outputs/gap_relm/best_stage_c")

# 单句纠错
result = pipeline("今天天汽很好")
print(result.prediction)  # "今天天气很好"

# 批量纠错
texts = ["今天天汽很好", "我去了北京。"]
results = pipeline(texts)
for r in results:
    print(f"{r.source} → {r.prediction}")
```

## 配置说明

### 消融实验配置

```python
from gap_relm.config import GapReLMConfig

config = GapReLMConfig()

# 关闭 Gap (退化为原始 ReLM)
config.ablation.enable_gap = False
config.ablation.enable_insert = False
config.ablation.enable_delete = False

# 只启用插入，不启用删除
config.ablation.enable_insert = True
config.ablation.enable_delete = False

# 启用迭代精炼
config.ablation.enable_iterative_refinement = True
config.ablation.refinement_rounds = 2

# 启用 Verifier
config.ablation.enable_verifier = True
```

### F2 优化配置

```python
# 代价敏感权重
config.f2_optimization.op_delete_weight = 3.0  # DELETE 权重
config.f2_optimization.op_replace_weight = 2.0  # REPLACE 权重
config.f2_optimization.insert_positive_weight = 5.0  # 插入权重

# 阈值校准
config.f2_optimization.delete_threshold = 0.5
config.f2_optimization.insert_threshold = 0.5

# 风险约束
config.f2_optimization.max_insert_per_sentence = 6
config.f2_optimization.max_insert_ratio = 0.1
```

### 预定义配置

```python
from gap_relm.config import get_config

# 默认配置
config = get_config("default")

# 消融实验配置
config = get_config("no_gap")      # 无 Gap (退化为 ReLM)
config = get_config("no_insert")   # 只删除不插入
config = get_config("no_delete")   # 只插入不删除
config = get_config("no_aux_mlm")  # 无辅助 MLM
config = get_config("no_f2")       # 无 F2 优化
config = get_config("with_refinement")  # 启用迭代精炼
config = get_config("with_verifier")    # 启用 Verifier
```

## 训练阶段

Gap-ReLM 采用分阶段训练策略：

### Stage A: Infiller 预训练
- 使用 Gold Template (Teacher Forcing)
- 冻结 Planner
- 训练 Infiller 学会填空

### Stage B: Planner 训练
- 监督序列标注
- 训练 Op Head 和 Insert Head

### Stage C: 联合微调
- Scheduled Sampling
- 解决训练/推理不一致

### Stage D: 质量增强 (可选)
- 迭代精炼
- Verifier 训练

## 数据处理流水线

```
原始语料 → 规范化/切句 → (X, Y) → 字符级对齐 → Planner 标签 → Gold Template
```

### 字符级对齐

使用 `python-Levenshtein` 进行编辑距离对齐：

```python
from gap_relm.data.alignment import CharacterAligner

aligner = CharacterAligner(algorithm="levenshtein")
result = aligner.align("今天天汽很好", "今天天气很好")

print(result.operations)
# [KEEP('今'), KEEP('天'), KEEP('天'), REPLACE('汽'->'气'), KEEP('很'), KEEP('好')]
```

### 标签生成

```python
from gap_relm.data.label_generator import create_sample_processor

processor = create_sample_processor(max_insert_num=3)
sample = processor.process("今天天汽很好", "今天天气很好")

print(sample.planner_labels.op_labels)     # [0, 0, 0, 2, 0, 0] (KEEP=0, REPLACE=2)
print(sample.planner_labels.insert_labels) # [0, 0, 0, 0, 0, 0]
print(sample.gold_template.template_tokens)
print(sample.gold_template.gold_tokens)
```

## 依赖项

- Python >= 3.8
- PyTorch >= 1.10
- transformers >= 4.20
- python-Levenshtein
- tensorboard
- tqdm

## 多卡 DDP 训练详解

使用 `torchrun` 进行多卡分布式训练：

```bash
# 双卡训练
torchrun --nproc_per_node=2 gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/gap_relm_ddp \
    --experiment_name gap_relm_ddp \
    --batch_size 32 \
    --num_epochs 10 \
    --fp16

# 四卡训练
torchrun --nproc_per_node=4 gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/gap_relm_ddp \
    --batch_size 16 \
    --num_epochs 10 \
    --fp16

# 使用脚本（推荐）
bash gap_relm/scripts/run_ddp.sh
```

**DDP 配置说明：**
- 数据会自动分片到各个 GPU
- 梯度会在各 GPU 间同步
- 每个 GPU 的有效 batch size = `batch_size`
- 全局有效 batch size = `batch_size × num_gpus`

---

## 数据增强与混淆集使用

Gap-ReLM 支持从干净语料自动生成训练数据，使用混淆集进行规则造错。

### 使用内置混淆集

项目内置了形近字和音近字混淆集：

```python
from gap_relm.data.augmentation import DataAugmentor, AugmentationConfig

# 使用默认混淆集
config = AugmentationConfig(
    use_default_shape_confusion=True,   # 使用形近字混淆集
    use_default_pinyin_confusion=True,  # 使用音近字混淆集
    p_corrupt=0.7,                       # 70% 的句子会被造错
    lambda_=1.5,                         # 平均编辑数
    pi_skip=0.2,                         # 20% 删字
    pi_multiply=0.3,                     # 30% 重复字
    pi_replace=0.5,                      # 50% 错字
)

augmentor = DataAugmentor(config)

# 对单个句子增强
result = augmentor.augment("这是一个正确的句子")
print(f"原句: {result.target}")
print(f"错句: {result.source}")
print(f"编辑: {result.edits}")

# 批量增强
clean_sentences = ["句子1", "句子2", "句子3"]
training_pairs = augmentor.generate_training_pairs(clean_sentences)
for src, tgt in training_pairs:
    print(f"{src} -> {tgt}")
```

### 自定义混淆集

#### 1. 创建混淆集文件

**JSON 格式** (`my_confusion.json`)：
```json
{
  "的": ["得", "地", "德"],
  "得": ["的", "地"],
  "在": ["再", "载"],
  "做": ["作", "座", "坐"]
}
```

**TSV 格式** (`my_confusion.tsv`)：
```
的	得	地	德
得	的	地
在	再	载
做	作	座	坐
```

#### 2. 加载自定义混淆集

```python
from gap_relm.data.confusion_set import ConfusionSet

# 方法1: 加载单个文件
confusion_set = ConfusionSet(
    use_default_shape=False,
    use_default_pinyin=False,
    custom_confusion_files=["my_confusion.json"]
)

# 方法2: 合并多个混淆集
confusion_set = ConfusionSet(
    use_default_shape=True,              # 使用内置形近字
    use_default_pinyin=True,             # 使用内置音近字
    custom_confusion_files=[
        "domain_specific.json",          # 领域特定混淆集
        "typo_patterns.tsv"              # 常见打字错误
    ]
)

# 查询混淆字符
confusions = confusion_set.get_confusions("的")
print(confusions)  # ['得', '地', '德']
```

#### 3. 在数据增强中使用自定义混淆集

```python
config = AugmentationConfig(
    use_default_shape_confusion=True,
    use_default_pinyin_confusion=True,
    custom_confusion_files=["my_confusion.json"],
    p_corrupt=0.7,
    lambda_=1.5,
)

augmentor = DataAugmentor(config)
```

#### 4. 在训练中使用自定义混淆集

**命令行方式：**
```bash
python gap_relm/scripts/train.py \
    --train_file data/clean_corpus.txt \
    --dev_file data/dev.jsonl \
    --data_format clean \
    --enable_augmentation \
    --confusion_set_path my_confusion.json \
    --aug_p_corrupt 0.7 \
    --aug_lambda 1.5 \
    --output_dir outputs/with_custom_confusion
```

**配置文件方式：**
```python
config = GapReLMConfig()
config.data.enable_augmentation = True
config.data.confusion_set_path = "my_confusion.json"
config.data.aug_p_corrupt = 0.7
config.data.aug_lambda = 1.5
config.data.aug_pi_skip = 0.2
config.data.aug_pi_multiply = 0.3
config.data.aug_pi_replace = 0.5
```

### 保护约束

在数据增强时，可以保护特定的文本片段不被修改：

```python
config = AugmentationConfig(
    enable_protection=True,
    enable_doc_number_protection=True,    # 保护文号 (京政发〔2023〕1号)
    enable_date_protection=True,          # 保护日期 (2023年12月13日)
    enable_amount_protection=True,        # 保护金额 (1000.00元)
    enable_clause_protection=True,        # 保护条款编号 (第一条)
    enable_org_protection=True,           # 保护机构名称
    enable_law_protection=True,           # 保护法规名称
    enable_phrase_protection=True,        # 保护固定格式
    custom_protected_words=["特定词汇", "专有名词"],  # 自定义保护词汇
)

augmentor = DataAugmentor(config)
```

### 数据增强参数调优

使用网格搜索找到最佳参数：

```python
from gap_relm.data.augmentation import DataAugmentor, AugmentationConfig

# 定义搜索空间
param_grid = {
    'p_corrupt': [0.5, 0.7, 0.9],
    'lambda_': [1.0, 1.5, 2.0],
    'pi_skip': [0.1, 0.2, 0.3],
    'pi_multiply': [0.2, 0.3, 0.4],
    'pi_replace': [0.4, 0.5, 0.6],
}

clean_corpus = ["句子1", "句子2", ...]  # 你的干净语料

for p in [0.5, 0.7, 0.9]:
    for lam in [1.0, 1.5, 2.0]:
        config = AugmentationConfig(
            p_corrupt=p,
            lambda_=lam,
            pi_skip=0.2,
            pi_multiply=0.3,
            pi_replace=0.5,
        )
        augmentor = DataAugmentor(config)
        results = augmentor.augment_batch(clean_corpus)
        stats = augmentor.get_stats(results)
        
        print(f"p={p}, λ={lam}")
        print(f"  平均编辑数: {stats['avg_edits']:.2f}")
        print(f"  造错率: {stats['corruption_rate']:.2%}")
```

---

## 推理使用

### 命令行推理

```bash
# 基础推理
python gap_relm/scripts/predict.py \
    --model_path outputs/gap_relm_full/best_model \
    --input test.txt \
    --output predictions.json \
    --batch_size 64

# 启用迭代精炼
python gap_relm/scripts/predict.py \
    --model_path outputs/gap_relm_full/best_model \
    --input test.txt \
    --output predictions.json \
    --use_refinement \
    --refinement_rounds 2

# 启用Verifier
python gap_relm/scripts/predict.py \
    --model_path outputs/gap_relm_full/best_model \
    --input test.txt \
    --output predictions.json \
    --use_verifier

# 不同输出格式
python gap_relm/scripts/predict.py \
    --model_path outputs/gap_relm_full/best_model \
    --input test.txt \
    --output predictions.txt \
    --output_format txt  # json/txt/parallel

# 只输出有修改的句子
python gap_relm/scripts/predict.py \
    --model_path outputs/gap_relm_full/best_model \
    --input test.txt \
    --output predictions.json \
    --only_changed
```

### Python API 推理

```python
from gap_relm.inference import GapReLMPipeline

# 加载模型
pipeline = GapReLMPipeline(
    model_path="outputs/gap_relm_full/best_model",
    device="cuda"
)

# 单句推理
text = "我今天很开兴"
result = pipeline(text)
print(f"原句: {result.source}")
print(f"纠正: {result.prediction}")
print(f"是否修改: {result.is_changed}")
print(f"置信度: {result.confidence:.4f}")

# 批量推理
texts = ["句子1", "句子2", "句子3"]
results = pipeline.predict_batch(texts, batch_size=32)
for r in results:
    if r.is_changed:
        print(f"{r.source} -> {r.prediction}")

# 启用迭代精炼
pipeline.predictor.inference_config.use_iterative_refinement = True
pipeline.predictor.ablation_config.refinement_rounds = 2
result = pipeline(text)

# 获取编辑操作
result = pipeline(text)
print(result.edits)
# [{"pos": 5, "type": "replace", "old": "兴", "new": "心"}]
```

---

## 消融实验

Gap-ReLM 支持多种消融实验配置，用于分析各模块的贡献。

### 1. 退化为原始 ReLM (无 Gap)

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/ablation_no_gap \
    --experiment_name ablation_no_gap \
    --no_gap \
    --num_epochs 10 \
    --batch_size 32 \
    --fp16
```

或使用预定义配置：
```python
config = get_config("no_gap")
```

### 2. 只启用插入，不启用删除

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/ablation_no_delete \
    --experiment_name ablation_no_delete \
    --no_delete \
    --num_epochs 10 \
    --fp16
```

### 3. 只启用删除，不启用插入

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/ablation_no_insert \
    --experiment_name ablation_no_insert \
    --no_insert \
    --num_epochs 10 \
    --fp16
```

### 4. 禁用辅助 MLM 任务

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/ablation_no_aux_mlm \
    --experiment_name ablation_no_aux_mlm \
    --no_aux_mlm \
    --num_epochs 10 \
    --fp16
```

### 5. 启用迭代精炼

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/with_refinement \
    --experiment_name with_refinement \
    --enable_refinement \
    --num_epochs 10 \
    --fp16
```

### 6. 禁用 F2 优化

```bash
python gap_relm/scripts/train.py \
    --train_file data/mucgec/train.jsonl \
    --dev_file data/mucgec/dev.jsonl \
    --output_dir outputs/ablation_no_f2 \
    --experiment_name ablation_no_f2 \
    --no_f2 \
    --num_epochs 10 \
    --fp16
```

### 批量消融实验脚本

```bash
#!/bin/bash

configs=("default" "no_gap" "no_insert" "no_delete" "no_aux_mlm" "no_f2")

for config in "${configs[@]}"; do
    python gap_relm/scripts/train.py \
        --config $config \
        --train_file data/mucgec/train.jsonl \
        --dev_file data/mucgec/dev.jsonl \
        --output_dir outputs/ablation_$config \
        --experiment_name ablation_$config \
        --num_epochs 10 \
        --batch_size 32 \
        --fp16
done
```

---

## 常见问题 (FAQ)

### Q1: 如何调整模型以适应特定领域？

**A:** 有几种方式：

1. **使用领域特定的混淆集**：
```python
config.data.confusion_set_path = "domain_confusion.json"
```

2. **调整造错参数**：
```python
config.data.aug_lambda = 2.0  # 增加编辑数量
config.data.aug_pi_replace = 0.7  # 提高错字比例
```

3. **使用领域语料进行预训练**：
```bash
python gap_relm/scripts/train.py \
    --train_file domain_clean_corpus.txt \
    --data_format clean \
    --enable_augmentation \
    --training_stage infiller_pretrain
```

### Q2: 训练时显存不足怎么办？

**A:** 可以尝试：

1. **减小 batch size**：
```bash
--batch_size 16
```

2. **启用梯度累积**：
```bash
--batch_size 16 \
--gradient_accumulation_steps 2  # 有效 batch size = 16*2 = 32
```

3. **减小最大序列长度**：
```bash
--max_seq_length 256
```

4. **使用 FP16 混合精度**：
```bash
--fp16
```

### Q3: 如何在新数据集上微调已训练的模型？

**A:**

```bash
python gap_relm/scripts/train.py \
    --train_file new_domain_data.jsonl \
    --dev_file new_domain_dev.jsonl \
    --resume_from outputs/gap_relm_full/best_model \
    --output_dir outputs/gap_relm_finetuned \
    --experiment_name gap_relm_finetuned \
    --training_stage joint_finetune \
    --num_epochs 5 \
    --learning_rate 1e-5 \
    --fp16
```

### Q4: 如何修改最大插入数量 K？

**A:**

```bash
python gap_relm/scripts/train.py \
    --max_insert_num 5 \  # 默认是 3，可以改为 5
    ...
```

或修改配置：
```python
config.model.max_insert_num = 5
```

注意：修改 K 后需要重新训练模型。

### Q5: 如何使用不同的预训练模型？

**A:**

```bash
python gap_relm/scripts/train.py \
    --pretrained_model hfl/chinese-roberta-wwm-ext \  # 使用 RoBERTa
    ...
```

支持的预训练模型：
- `hfl/chinese-macbert-base` (默认)
- `hfl/chinese-roberta-wwm-ext`
- `hfl/chinese-bert-wwm-ext`
- `bert-base-chinese`

### Q6: 如何查看训练日志和可视化？

**A:**

启动 TensorBoard：
```bash
tensorboard --logdir runs/gap_relm_base
```

然后在浏览器访问 `http://localhost:6006`

### Q7: 数据增强生成的数据质量如何保证？

**A:**

1. **使用保护约束**：保护不应该被修改的片段
2. **调整造错参数**：降低 `p_corrupt` 和 `lambda_` 来减少错误数量
3. **人工审核样本**：检查生成的训练数据质量
4. **混合真实数据**：将生成数据与真实标注数据混合使用

```python
# 生成数据后进行质量检查
results = augmentor.augment_batch(clean_sentences)
stats = augmentor.get_stats(results)

print(f"平均编辑数: {stats['avg_edits']}")
print(f"造错率: {stats['corruption_rate']}")
print(f"删除操作占比: {stats['skip_ratio']}")
print(f"重复操作占比: {stats['multiply_ratio']}")
print(f"替换操作占比: {stats['replace_ratio']}")

# 如果质量不满意，调整参数
if stats['avg_edits'] > 3:
    config.lambda_ = 1.0  # 降低编辑数量
```

---

## 参考论文

- ReLM: Chinese Spelling Correction as Rephrasing Language Model
- GECToR: Grammatical Error Correction: Tag, Not Rewrite
- Levenshtein Transformer
- Mask-Predict: Parallel Decoding of Conditional Masked Language Models

## License

MIT License
