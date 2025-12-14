# Gap-ReLM 项目修改计划

## 最后修改日期：2025-12-14

---

## 零、导入路径修复（2025-12-14 修复 ✅）

### 0.0 问题
由于项目根目录本身就叫 `gap_relm`，而代码中使用了 `from gap_relm.xxx` 的导入方式，导致在服务器上运行时出现 `ModuleNotFoundError: No module named 'gap_relm'`。

### 0.1 修复
修改所有脚本的导入语句，去掉 `gap_relm.` 前缀：

| 文件 | 修改前 | 修改后 | 状态 |
|------|--------|--------|------|
| `scripts/generate_frozen_dev.py` | `from gap_relm.data.augmentation import ...` | `from data.augmentation import ...` | ✅ |
| `scripts/train.py` | `from gap_relm.config import ...` | `from config import ...` | ✅ |
| `scripts/generate_training_data.py` | `from gap_relm.data.augmentation import ...` | `from data.augmentation import ...` | ✅ |
| `scripts/predict.py` | `from gap_relm.inference import ...` | `from inference import ...` | ✅ |

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
17. ✅ 更新本文档

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
