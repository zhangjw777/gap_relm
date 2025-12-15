# Gap-ReLM 项目修改计划

## 最后修改日期：2025-12-14

---

## 〇-0、静态训练数据生成功能（2025-12-14 新增 ✅）

### 0.0.1 背景与需求
在线动态数据增强存在CPU瓶颈（实时编辑距离对齐），改用预生成静态数据并**直接保存预计算标签**。

**核心优化**：
- 造错时已知编辑操作（位置、类型、字符），直接生成Planner标签和Gold Template
- 训练时零CPU开销，直接读取标签，跳过对齐算法
- 兼容旧格式：检测到预计算字段则直接使用，否则走传统对齐流程

### 0.0.2 代码修改

| 文件 | 修改 |
|------|------|
| `data/error_generator.py` | 新增 `corrupt_with_type()` 方法 |
| `data/augmentation.py` | 精简重构，`StaticDataGenerator` 支持生成预计算标签 |
| `data/dataset.py` | `_load_mucgec` 检测预计算格式，`_load_and_process` 支持跳过对齐 |
| `scripts/generate_training_data.py` | 精简为纯静态数据生成脚本 |

### 0.0.3 数据格式

**预计算格式（推荐）**：
```json
{
  "source": "错误句子",
  "target": "正确句子",
  "op_labels": [0, 0, 2, 0, 0],
  "insert_labels": [0, 0, 0, 0, 0],
  "template_tokens": ["错", "误", "[MASK]", "子"],
  "gold_tokens": ["句"],
  "mask_positions": [2]
}
```

**基础格式（兼容）**：
```json
{"source": "错误句子", "target": "正确句子"}
```

### 0.0.4 使用方法

```bash
python scripts/generate_training_data.py \
    --clean_file data/clean_sentences.txt \
    --output_dir ./static_training_data \
    --confusion_file data/confusion_sets/pycorrect_merged.json \
    --num_negative 2 \
    --pi_skip 0.2 --pi_multiply 0.3 --pi_replace 0.5 \
    --seed 42
```

### 0.0.5 待办：开源数据离线编译

对于没有预计算标签的开源数据集（如MuCGEC、SIGHAN），可以一次性"离线编译"成预计算格式：

```python
# TODO: 实现离线编译脚本
# python scripts/compile_dataset.py \
#     --input data/mucgec/train.json \
#     --output data/mucgec/train_compiled.jsonl
```

### 0.0.6 脚本兼容性检查（2025-12-14 完成 ✅）

已验证 `scripts/` 下所有脚本的导入兼容性：

| 脚本 | 导入 | 状态 |
|------|------|------|
| `generate_training_data.py` | `StaticDataGenerator, StaticSampleConfig` | ✅ |
| `train.py` | `create_data_loaders, create_online_data_loaders` | ✅ |
| `generate_frozen_dev.py` | `DataAugmentor, AugmentationConfig` | ✅ |
| `predict.py` | 无 `data` 模块依赖 | ✅ |
| `convert_pycorrect_confusion.py` | 无 `data` 模块依赖 | ✅ |

所有脚本无需修改，`data/__init__.py` 已正确导出所有必需的类和函数。

---

## 〇、关键缺陷修复与 API 更新（2025-12-14 修复 ✅）

### 0.1 Scheduled Sampling 逻辑缺陷修复

**问题描述**：
在 `trainers/trainer.py` 的 `_apply_scheduled_sampling` 方法中存在严重逻辑缺陷：
- 当使用 Planner 预测的模板替换 Gold Template 时，模板长度和 [MASK] 位置会发生变化
- 但 `infill_labels` 仍然是基于 Gold Template 生成的标签
- 两者长度不一致，传入 CrossEntropyLoss 时会导致形状不匹配
- 在某些 CUDA 环境下，这种不匹配会直接导致 `CUDA driver error: unknown error`

**修复方案**：
1. 修改 `_apply_scheduled_sampling` 方法签名，返回 `Tuple[Dict, bool]`
2. 第二个返回值 `skip_infiller_loss` 标识是否跳过 Infiller Loss
3. 当使用预测模板时，设置 `skip_infiller_loss=True`
4. 在前向传播时，如果 `skip_infiller_loss=True`，传入 `infill_labels=None`
5. 此时只计算 Planner Loss，符合 Scheduled Sampling 的设计目的

**修改文件**：`trainers/trainer.py`

### 0.2 PyTorch AMP API 更新

**问题描述**：
`torch.cuda.amp.autocast` 和 `torch.cuda.amp.GradScaler` 已弃用，需要使用新 API。

**修复方案**：
```python
# 旧 API (已弃用)
from torch.cuda.amp import autocast, GradScaler
with autocast(enabled=..., dtype=...):
scaler = GradScaler()

# 新 API
from torch.amp import autocast, GradScaler
with autocast('cuda', enabled=..., dtype=...):
scaler = GradScaler('cuda')
```

**修改文件**：`trainers/trainer.py`

### 0.3 辅助 MLM 损失维度不匹配修复

**问题描述**：
运行时报错：`ValueError: Expected input batch_size (7616) to match target batch_size (8192)`
- 位置：`models/infiller.py` 第 132 行的 `aux_mlm_loss` 计算
- 原因：`aux_mlm_labels` 是基于源序列生成的，形状 `[batch, seq_len=128]`
- 但 Infiller 的输入是模板序列，logits 形状 `[batch, template_len, vocab]`
- 当模板长度 ≠ 源序列长度时（例如 119 vs 128），会导致维度不匹配

**修复方案**：
1. Infiller 只计算 `infill_loss`（基于模板序列）
2. `aux_mlm_loss` 单独在源序列上计算（使用 `encoder_hidden`）
3. 最后将两个损失相加

**修改文件**：`models/gap_relm.py`

```python
# 修复后：aux_mlm_loss 在源序列上单独计算
if aux_mlm_labels is not None and self.ablation_config.enable_aux_mlm:
    aux_logits = self.infiller.lm_head(encoder_hidden)  # [batch, seq_len, vocab]
    aux_mlm_loss = F.cross_entropy(aux_logits.view(-1, vocab), aux_mlm_labels.view(-1), ignore_index=-100)
    infiller_loss = infiller_loss + mu_aux * aux_mlm_loss
```

### 0.4 DDP 梯度同步问题修复

**问题描述**：
运行时报错：`RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one`
- 原因：P-Tuning 有独立的 `planner_ptuning` 和 `infiller_ptuning`
- 当 `training_stage="planner"` 时，`infiller_ptuning` 没有被使用
- DDP 要求所有参数都参与 loss 计算

**修复方案**：
1. 在 forward 开始时，无论训练阶段如何，都调用两个 P-Tuning 模块
2. 使用 `dummy_loss = prompt.sum() * 0.0` 确保梯度流动但不影响实际损失
3. 将 dummy_loss 加到 total_loss 中

**修改文件**：`models/gap_relm.py`

```python
# DDP 兼容性：确保所有 P-Tuning 参数都参与梯度计算
ptuning_dummy_loss = torch.tensor(0.0, device=device)
if self.ablation_config.enable_ptuning and not self.ablation_config.ptuning_shared:
    if self.planner_ptuning is not None:
        planner_prompt = self.planner_ptuning(batch_size, device)
        ptuning_dummy_loss = ptuning_dummy_loss + planner_prompt.sum() * 0.0
    if self.infiller_ptuning is not None:
        infiller_prompt = self.infiller_ptuning(batch_size, device)
        ptuning_dummy_loss = ptuning_dummy_loss + infiller_prompt.sum() * 0.0

# 最后添加到 total_loss
total_loss = total_loss + ptuning_dummy_loss
```

### 0.5 BatchSize 与显存分析

**双 4090 (24GB×2) + MacBERT-base + BatchSize=128 显存估算（修正版）**：

Gap-ReLM 是 **Planner + Infiller 双模型**架构，显存需求比单模型高很多：

| 组件 | FP16 估算 |
|------|----------|
| 模型参数 (×2) | ~400 MB |
| 梯度 (×2) | ~400 MB |
| 优化器状态 | ~1.6 GB |
| **Planner 激活值** | ~15 GB |
| **Infiller 激活值** | ~15 GB |
| **Infiller Logits 峰值** (B×L×V) | ~35 GB |
| **总计（峰值）** | **~67 GB** |

**建议配置**：
- `batch_size=16` (per-GPU=8) + `gradient_accumulation_steps=8`
- 或 `batch_size=32` (per-GPU=16) + `gradient_accumulation_steps=4`

### 0.6 数据加载性能优化

**问题**：在线数据增强导致训练速度慢（1.4s/it）

**解决方案**：
1. 添加 `--prefetch_factor` 参数支持（默认 2，可增大到 4-8）
2. 建议增大 `--num_workers` 到 8-16

**修改文件**：
- `scripts/train.py` - 添加 `--prefetch_factor` 参数
- `data/data_loader.py` - 传递 `prefetch_factor` 给 DataLoader
- `scripts/run_ddp.sh` - 添加 `PREFETCH_FACTOR` 配置

### 0.7 断点续训支持

项目已支持断点续训，使用方式：
```bash
./scripts/run_ddp.sh  # 正常训练

# 中断后恢复
# 在 run_ddp.sh 中添加：--resume_from ./outputs/checkpoint-1500
```

---

## 零-1、导入路径修复（2025-12-14 修复 ✅）

### 0.0 问题
由于项目根目录本身就叫 `gap_relm`，而代码中使用了 `from gap_relm.xxx` 的导入方式，导致在服务器上运行时出现 `ModuleNotFoundError: No module named 'gap_relm'`。

### 0.1 修复
修改所有模块文件的导入语句，去掉 `gap_relm.` 前缀和相对导入：

| 文件 | 修改前 | 修改后 | 状态 |
|------|--------|--------|------|
| `scripts/generate_frozen_dev.py` | `from gap_relm.data.augmentation import ...` | `from data.augmentation import ...` | ✅ |
| `scripts/train.py` | `from gap_relm.config import ...` | `from config import ...` | ✅ |
| `scripts/generate_training_data.py` | `from gap_relm.data.augmentation import ...` | `from data.augmentation import ...` | ✅ |
| `scripts/predict.py` | `from gap_relm.inference import ...` | `from inference import ...` | ✅ |
| `trainers/trainer.py` | `from ..models import ...` | `from models import ...` | ✅ |
| `trainers/trainer.py` | `from ..config import ...` | `from config import ...` | ✅ |
| `inference.py` | `from ..models import ...` | `from models import ...` | ✅ |
| `inference.py` | `from ..config import ...` | `from config import ...` | ✅ |
| `models/gap_relm.py` | `from ..config import ...` | `from config import ...` | ✅ |

现在可以直接在项目根目录运行脚本：
```bash
cd /path/to/gap_relm
python scripts/generate_frozen_dev.py ...
python scripts/train.py ...
```

---

## 一、代码质量改进（2025-12-14 新增 ✅）

### 1.1 背景
在代码检查过程中发现了5个潜在问题，这些问题不会影响正常运行，但在边界条件下可能导致错误或不一致。

### 1.2 修复的问题

| 问题 | 位置 | 修复内容 | 状态 |
|------|------|---------|------|
| **随机种子不确定性** | `data/dataset.py` | 在线数据集使用可控RNG，结合idx+worker_id+torch.initial_seed()生成确定性随机种子 | ✅ |
| **错误处理逻辑** | `data/dataset.py` | 样本处理失败时作为正例使用（source==target），并添加日志记录 | ✅ |
| **概率归一化验证** | `data/error_generator.py` | 添加概率和为0的检查，抛出ValueError | ✅ |
| **混合精度配置冲突** | `scripts/train.py` | 添加fp16和bf16互斥检查，禁止同时使用 | ✅ |
| **模板构建空指针** | `trainers/trainer.py` | 在scheduled sampling中检查template_result是否为None | ✅ |

### 1.3 详细修改

#### 1. 随机种子可控性（最重要）
**修改文件**：
- `data/dataset.py` - OnlineAugmentedDataset.__getitem__
- `data/error_generator.py` - ErrorGenerator.corrupt, TruncatedPoisson.sample, _sample_error_type, _apply_error
- `data/confusion_set.py` - ConfusionSet.get_random_confusion

**修改内容**：
- 在`__getitem__`中使用`hash((idx, worker_id, torch.initial_seed()))`生成样本级RNG
- ErrorGenerator支持传入可选的`rng`参数
- 所有使用random的方法都支持可控RNG
- 保证分布式训练时各worker数据不同但可复现

**优势**：
- 解决分布式训练时各GPU数据不一致问题
- 保证训练可复现性
- 每个epoch数据仍然不同（因为torch.initial_seed()会变化）

#### 2. 错误处理改进
**修改文件**：`data/dataset.py`

**修改内容**：
- 样本处理失败时使用干净句子作为正例（source==target, 全KEEP标签）
- 添加debug级别的日志记录，便于追踪问题

**优势**：
- 正例有助于模型学习"不修改正确句子"
- 不会因为个别样本失败而中断训练

#### 3. 概率验证
**修改文件**：`data/error_generator.py`

**修改内容**：
- 在概率归一化前检查`pi_total < 1e-6`
- 抛出详细的ValueError错误信息

#### 4. 混合精度互斥
**修改文件**：`scripts/train.py`

**修改内容**：
- 在main函数开始处检查`args.fp16 and args.bf16`
- 如果同时设置则抛出ValueError

#### 5. 空指针检查
**修改文件**：`trainers/trainer.py`

**修改内容**：
- 在`_apply_scheduled_sampling`中检查`template_result is None or template_result.template_ids is None`
- 失败时记录warning并使用原始Gold Template

---

## 一、P-Tuning 功能添加（已完成 ✅）

### 1.1 背景
原始 ReLM 论文中使用了 P-Tuning 技术，对模型性能有贡献。当前项目缺失该功能，需要添加作为消融实验。

### 1.2 实现方案
采用 **方案 B：Planner/Infiller 各自独立 P-Tuning**

原因：
- 实现更简洁，不需要大量修改现有代码结构
- 能有效隔离 Planner 和 Infiller 的梯度冲突
- Planner 输入只有 X（序列标注），Infiller 输入是 X+T（模板填空），输入分布差异很大
- 后续如需要可以轻松扩展为共享 + 独立的分层结构

### 1.3 修改的文件

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `models/ptuning.py` | 新建，实现 PTuningEncoder、PTuningWrapper、TaskSpecificPTuning | ✅ |
| `models/__init__.py` | 添加 P-Tuning 模块导出 | ✅ |
| `models/gap_relm.py` | 集成 P-Tuning，添加 `_encode_with_ptuning`、`_infiller_predict_with_ptuning` 方法 | ✅ |
| `models/infiller.py` | 添加 `inputs_embeds` 参数支持 | ✅ |
| `config.py` | 添加 P-Tuning 配置项和消融实验配置 | ✅ |
| `scripts/train.py` | 添加 P-Tuning 命令行参数 | ✅ |
| `scripts/run_ddp.sh` | 添加 P-Tuning 配置选项 | ✅ |

### 1.4 新增配置项（在 AblationConfig 中）

```python
# P-Tuning 配置（论文中对性能有贡献）
enable_ptuning: bool = True          # 启用 P-Tuning（默认开启）
ptuning_prompt_length: int = 10      # Prompt 长度（虚拟 token 数量）
ptuning_use_lstm: bool = True        # 是否使用 LSTM 编码 prompt
ptuning_use_mlp: bool = True         # 是否使用 MLP 编码 prompt
ptuning_shared: bool = False         # 是否 Planner/Infiller 共享 prompt
```

### 1.5 新增消融实验配置

| 配置名 | 功能 |
|--------|------|
| `no_ptuning` | 关闭 P-Tuning |
| `ptuning_no_lstm` | P-Tuning 不使用 LSTM |
| `ptuning_shared` | P-Tuning 共享 prompt |

---

## 二、在线动态数据增强功能（已完成 ✅）

### 2.1 背景
原始数据增强策略是每个句子只生成一次（正例或负例），实际错误数由泊松分布采样。新方案在 `Dataset.__getitem__` 中实时造错，每个 Epoch 数据都不同。

### 2.2 优势
1. **无限数据**：只要训练不停止，模型永远在看新的错误组合，对提升 F2 极有帮助
2. **防止过拟合**：模型无法死记硬背，只能学习语法和语义规律
3. **充分利用 GPU**：计算密集型方法，能充分利用双 4090

### 2.3 实现方案

#### 核心组件
1. **OnlineAugmentedDataset**: 在线动态增强数据集类
   - 在 `__getitem__` 中实时调用 ErrorGenerator 造错
   - 支持长度自适应 λ（短句少错，长句多错）
   
2. **LengthAdaptiveLambda**: 长度自适应 λ 参数类
   - 线性插值模式：根据句子长度线性调整 λ
   - 比例模式：错误数占句子长度的固定比例

3. **Frozen-Dev-Synth**: 固定验证集
   - 预生成的固定错误数据，用于稳定评估
   - 确保训练曲线可比，避免数据变化带来的噪声

### 2.4 修改的文件

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `data/dataset.py` | 新增 `OnlineAugmentedDataset`、`LengthAdaptiveLambda`、`load_clean_sentences` | ✅ |
| `data/data_loader.py` | 新增 `create_online_data_loaders` 函数 | ✅ |
| `data/__init__.py` | 导出新类和函数 | ✅ |
| `scripts/train.py` | 添加在线增强相关命令行参数 | ✅ |
| `scripts/generate_frozen_dev.py` | **新建**：Frozen-Dev-Synth 生成脚本 | ✅ |
| `scripts/run_ddp.sh` | 添加在线增强相关参数 | ✅ |
| `scripts/quick_start.sh` | 更新支持在线/静态两种模式 | ✅ |

### 2.5 新增命令行参数

#### 基础参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--online_augment` | True | 启用在线动态数据增强（默认开启） |
| `--no_online_augment` | - | 关闭在线增强，使用预生成数据 |
| `--clean_train_file` | None | 干净句子文件路径（在线增强用） |
| `--frozen_dev_file` | None | 固定验证集路径 |
| `--clean_file_format` | "txt" | 干净文件格式（txt/json/jsonl） |
| `--clean_text_field` | "text" | JSON 中的文本字段名 |

#### 造错参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--p_corrupt` | 0.7 | 造错概率 |
| `--base_lambda` | 1.5 | 基础泊松参数 |
| `--pi_skip` | 0.2 | 删字概率 |
| `--pi_multiply` | 0.3 | 重复字概率 |
| `--pi_replace` | 0.5 | 错字概率 |
| `--max_edits_per_sent` | 4 | 单句最大编辑数 |
| `--max_insert_k` | 3 | 单次最大重复字符数 |

#### 长度自适应参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_length_adaptive` | True | 启用长度自适应 λ |
| `--no_length_adaptive` | - | 关闭长度自适应 |
| `--min_length_for_lambda` | 20 | λ 最小值对应的句子长度 |
| `--max_length_for_lambda` | 80 | λ 最大值对应的句子长度 |
| `--min_lambda` | 1.0 | 最小 λ 值 |
| `--max_lambda` | 3.0 | 最大 λ 值 |
| `--use_ratio_mode` | False | 使用错误比例模式 |
| `--error_ratio` | 0.05 | 错误比例（比例模式） |

### 2.6 使用示例

#### 生成 Frozen-Dev-Synth（固定验证集）
```bash
# 基础用法
python scripts/generate_frozen_dev.py \
    --clean_file data/clean_dev.txt \
    --output_file data/frozen_dev.jsonl \
    --num_samples 20000 \
    --seed 42

# 指定造错参数
python scripts/generate_frozen_dev.py \
    --clean_file data/clean_dev.txt \
    --output_file data/frozen_dev.jsonl \
    --p_corrupt 0.7 \
    --lambda_ 1.5 \
    --pi_skip 0.2 \
    --pi_multiply 0.3 \
    --pi_replace 0.5

# 从 JSONL 文件读取
python scripts/generate_frozen_dev.py \
    --clean_file data/corpus.jsonl \
    --file_format jsonl \
    --text_field sentence \
    --output_file data/frozen_dev.jsonl
```

#### 在线动态增强（默认模式）
```bash
# 使用干净句子文件 + 固定验证集
python scripts/train.py \
    --clean_train_file data/clean_sentences.txt \
    --frozen_dev_file data/frozen_dev.jsonl \
    --online_augment \
    --p_corrupt 0.7 \
    --base_lambda 1.5 \
    --enable_length_adaptive \
    --min_lambda 1.0 \
    --max_lambda 3.0

# 如果 train_file 就是干净句子，可以直接用
python scripts/train.py \
    --train_file data/clean_sentences.txt \
    --frozen_dev_file data/frozen_dev.jsonl
```

#### 使用 DDP 多卡训练（在线增强）
```bash
bash scripts/run_ddp.sh \
    --train_file data/clean_train.txt \
    --frozen_dev_file data/frozen_dev.jsonl \
    --online_augment \
    --num_gpus 2 \
    --p_corrupt 0.7 \
    --base_lambda 1.5
```

#### 静态预生成数据模式
```bash
# 关闭在线增强，使用预生成的训练数据
python scripts/train.py \
    --train_file data/train.jsonl \
    --dev_file data/dev.jsonl \
    --no_online_augment
```

### 2.7 架构说明

```
在线动态增强流程:

__getitem__(idx) 被调用时:
  1. 获取干净句子 clean_sentences[idx]
  2. 根据句子长度计算当前 λ（长度自适应）
  3. 以 p_corrupt 概率决定是否造错
  4. 若造错，从截断泊松采样错误数量
  5. 按 π 分布选择错误类型（删/重复/替）
  6. 执行造错，生成 (source, target) 对
  7. 对齐并生成 Planner/Infiller 标签
  8. 返回模型输入特征

训练时:
  - 每个 Epoch，同一个 idx 对应不同的错误
  - Epoch 1: 句子 A 可能是正确的
  - Epoch 2: 句子 A 可能有删字错误
  - Epoch 3: 句子 A 可能有重复字错误

验证时:
  - 使用 Frozen-Dev-Synth（固定验证集）
  - 每次验证数据相同，确保曲线可比
```

---

## 三、待办事项

### 3.1 后续优化（可选）
- [ ] 实现方案 C：共享域 prompt + 任务独立 prompt（分层）
- [ ] 添加 prompt 初始化策略选项（随机、从词表采样等）
- [ ] 支持 deep prompt tuning（每层都有 prompt）
- [ ] 添加更多错误类型（如语序错误）
- [ ] 实现基于困惑度的自适应造错

### 3.2 测试验证
- [ ] 在服务器上运行测试，验证代码正确性
- [ ] 进行消融实验，对比有无 P-Tuning 的性能差异
- [ ] 对比在线增强 vs 静态数据的训练效果
- [ ] 验证长度自适应 λ 的效果
- [ ] 记录实验结果

---

## 〇-1、内存溢出问题修复 - 惰性加载数据集（2025-12-15 新增 ✅）

### 问题描述
在使用预计算格式的大规模数据集（700万条样本）训练时，系统 RAM 溢出导致进程被杀死（exitcode: -9）。

**根本原因**：
- `GapReLMDataset` 在 `__init__` 时调用 `_load_and_process`，一次性将所有数据加载到 `self.samples` 列表
- 700万条 `ProcessedSample` 对象，每条约 1-2KB 内存
- 粗略估计：700万 × 1.5KB ≈ 10GB+，加上 Python 对象开销、pickle 加载临时内存、DDP 多进程等，轻松超过 32GB RAM

### 解决方案：惰性加载数据集 `LazyGapReLMDataset`

新增 `LazyGapReLMDataset` 类，采用**按需读取**策略：

1. **索引扫描**：初始化时只扫描文件，记录每行的字节偏移（约 700万 × 8B = 56MB）
2. **按需读取**：`__getitem__` 时通过 `file.seek(offset)` 定位并读取单行
3. **索引缓存**：首次扫描后保存索引文件，后续启动秒加载

**内存占用对比**：
| 模式 | 内存占用 | 适用场景 |
|------|---------|---------|
| `GapReLMDataset` | ~10GB+ | 小数据集（<100万样本） |
| `LazyGapReLMDataset` | ~100MB | 大数据集（>100万样本） |

### 代码修改

| 文件 | 修改内容 |
|------|---------|
| `data/dataset.py` | 新增 `LazyGapReLMDataset` 类 |
| `data/data_loader.py` | `create_data_loaders` 添加 `lazy_load` 参数 |
| `data/__init__.py` | 导出 `LazyGapReLMDataset` |
| `scripts/train.py` | 添加 `--lazy_load` 命令行参数 |

### 使用方法

```bash
# 使用惰性加载（推荐用于大数据集）
torchrun --nproc_per_node=2 scripts/train.py \
    --train_file data/train_7m.jsonl \
    --dev_file data/dev.jsonl \
    --no_online_augment \
    --lazy_load \
    --batch_size 64 \
    --num_epochs 10
```

### 注意事项
1. `--lazy_load` 仅适用于**预计算标签格式**的 JSONL 文件
2. 首次运行会扫描文件建立索引，后续启动直接加载索引缓存
3. 验证集通常较小，不使用惰性加载（直接全部加载）
4. 索引缓存默认保存在 `--cache_dir` 目录下

---

## 四、修改总结

### 已完成任务
1. ✅ 创建 `models/ptuning.py`，实现完整的 P-Tuning 模块
2. ✅ 更新 `config.py`，添加 P-Tuning 相关配置项
3. ✅ 修改 `models/gap_relm.py`，集成 P-Tuning
4. ✅ 修改 `models/infiller.py`，支持 inputs_embeds 参数
5. ✅ 更新 `models/__init__.py`，导出新模块
6. ✅ 添加消融实验配置函数
7. ✅ 更新 `scripts/train.py`，添加 P-Tuning 命令行参数
8. ✅ 更新 `scripts/run_ddp.sh`，添加 P-Tuning 配置选项
9. ✅ 创建 `OnlineAugmentedDataset` 类，支持在线动态数据增强
10. ✅ 实现 `LengthAdaptiveLambda` 类，支持长度自适应 λ
11. ✅ 创建 `create_online_data_loaders` 函数
12. ✅ 更新 `data/__init__.py`，导出新组件
13. ✅ 更新 `scripts/train.py`，添加在线增强命令行参数
14. ✅ 创建 `scripts/generate_frozen_dev.py`，Frozen-Dev-Synth 生成脚本
15. ✅ 更新 `scripts/run_ddp.sh`，添加在线增强参数
16. ✅ 更新 `scripts/quick_start.sh`，支持在线/静态两种模式
17. ✅ **新增 `LazyGapReLMDataset` 惰性加载数据集，解决大规模数据内存溢出问题**
18. ✅ 更新本文档

### 未完成任务
- 需要在服务器上测试代码
- 需要生成 Frozen-Dev-Synth 验证集

---

## 五、注意事项

### P-Tuning
1. **默认行为**：P-Tuning 默认开启（`enable_ptuning=True`）
2. **参数量增加**：P-Tuning 会引入额外参数（约 prompt_length × hidden_size × 4）
3. **训练时**：P-Tuning 参数会自动加入可训练参数列表
4. **推理时**：P-Tuning 会影响编码过程，确保模型保存时包含 P-Tuning 权重

### 在线动态增强
1. **默认行为**：在线增强默认开启（`--online_augment`）
2. **Frozen-Dev-Synth**：务必使用固定验证集，否则训练曲线不可比
3. **计算开销**：每次 `__getitem__` 都会执行造错和对齐，CPU 开销较大
4. **num_workers**：建议使用较大的 num_workers 以并行处理数据
5. **分布式训练**：每个进程独立造错，同一句子在不同 GPU 上可能有不同错误

### 惰性加载数据集
1. **适用场景**：大规模预计算数据集（>100万样本），内存受限环境
2. **首次启动**：需要扫描文件建立索引，耗时约几分钟（700万条约3-5分钟）
3. **后续启动**：直接加载索引缓存，秒启动
4. **数据格式**：仅支持预计算标签格式的 JSONL 文件
