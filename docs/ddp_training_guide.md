# DDP多卡训练使用指南

## 关键确认：joint训练时Infiller使用Gold Template

### ✅ 已确认的训练逻辑

在 `joint_finetune` 阶段（Planner + Infiller多任务联合训练）：

1. **Planner训练**：使用源序列，预测op_labels和insert_labels
2. **Infiller训练**：使用**Gold Template**（从标注数据生成的正确模板），而不是Planner预测的模板

**代码确认**（[models/gap_relm.py](models\gap_relm.py) Line 150-230）：

```python
def forward(
    self,
    input_ids,           # 源序列
    template_input_ids,  # Gold Template（从数据集预生成）
    training_stage="joint",
    ...
):
    # Planner前向：在源序列上预测
    if training_stage in ["planner", "joint"]:
        planner_output = self.planner(encoder_hidden, ...)
    
    # Infiller前向：使用Gold Template（不是Planner构建的）
    if training_stage in ["infiller", "joint"]:
        infiller_output = self.infiller(
            input_ids=template_input_ids,  # ← 这是Gold Template！
            labels=infill_labels,           # 正确的填充标签
            ...
        )
    
    # 联合损失
    total_loss = planner_loss + lambda_infill * infiller_loss
```

**数据流确认**（[data/dataset.py](data\dataset.py) Line 300-400）：

```python
def __getitem__(self, idx):
    sample = self.samples[idx]
    
    # 从对齐结果生成Gold Template
    template_tokens = sample.gold_template.template_tokens  # 包含MASK
    gold_tokens = sample.gold_template.gold_tokens          # MASK的答案
    
    # 返回数据
    return {
        'input_ids': source_ids,              # 源序列
        'template_input_ids': template_ids,   # Gold Template（预生成）
        'infill_labels': infill_labels,       # MASK位置的正确答案
        ...
    }
```

### 🎯 训练策略说明

#### 训练阶段对比

| 阶段 | Planner | Infiller | Template来源 |
|------|---------|----------|------------|
| Stage A: infiller_pretrain | ❌ 冻结 | ✅ 训练 | Gold Template |
| Stage B: planner_train | ✅ 训练 | ❌ 可选冻结 | Gold Template |
| Stage C: joint_finetune | ✅ 训练 | ✅ 训练 | **Gold Template** |

**关键点**：所有阶段都使用Gold Template训练Infiller，避免训练-推理不一致。

#### joint_finetune的优势

1. **Planner学习预测**：op和insert标签
2. **Infiller学习填充**：在正确的模板上填充MASK
3. **联合优化**：两个任务互相促进
4. **避免误差累积**：训练时不用Planner的错误预测

---

## 快速开始：多卡联合训练

### 方式1：使用默认配置

```bash
# 最简单的方式
bash scripts/run_ddp.sh \
    --train_file ./data/mucgec_train.json \
    --dev_file ./data/mucgec_dev.json \
    --num_gpus 2
```

### 方式2：自定义配置

```bash
bash scripts/run_ddp.sh \
    --train_file ./data/mucgec_train.json \
    --dev_file ./data/mucgec_dev.json \
    --num_gpus 2 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --output_dir ./outputs/exp1 \
    --experiment_name mucgec_joint_training
```

### 方式3：修改脚本中的默认参数

编辑 `scripts/run_ddp.sh`，修改以下参数：

```bash
# 必填参数（运行时提供）
TRAIN_FILE=""                              # 通过命令行指定

# 基础配置（可在脚本中修改默认值）
NUM_GPUS=2                                 # 你的GPU数量
BATCH_SIZE=32                              # 每个GPU的batch size
NUM_EPOCHS=10                              # 训练轮数
LEARNING_RATE=2e-5                         # 学习率

# 训练策略
TRAINING_STAGE="joint_finetune"            # 联合训练（推荐）
```

然后运行：
```bash
bash scripts/run_ddp.sh \
    --train_file ./data/mucgec_train.json \
    --dev_file ./data/mucgec_dev.json
```

---

## 必填参数说明

### 🔴 必须提供的参数

1. **--train_file**：训练数据文件路径
   ```bash
   --train_file ./data/mucgec_train.json
   ```
   
   格式要求：
   ```json
   {"source": "错误句", "target": "正确句"}
   {"source": "错误句", "target": "正确句"}
   ```

### 🟡 强烈推荐提供的参数

2. **--dev_file**：验证数据文件路径
   ```bash
   --dev_file ./data/mucgec_dev.json
   ```
   - 用于每个epoch后评估模型
   - 用于保存最佳模型
   - 如果不提供，只进行训练不评估

3. **--num_gpus**：GPU数量
   ```bash
   --num_gpus 2  # 根据你的实际GPU数量
   ```
   - 默认是4，根据实际情况修改
   - 查看GPU: `nvidia-smi`

### 🟢 可选参数（有默认值）

其他参数都有合理的默认值，可以不指定：

```bash
--data_format mucgec              # 数据格式
--batch_size 32                   # batch size
--num_epochs 10                   # 训练轮数
--learning_rate 2e-5              # 学习率
--output_dir ./outputs            # 输出目录
--experiment_name gap_relm        # 实验名称
```

---

## 完整训练流程示例

### 示例1：使用MuCGEC数据训练

```bash
# 1. 准备数据
ls ./data/mucgec_train.json
ls ./data/mucgec_dev.json

# 2. 开始训练（4卡）
bash scripts/run_ddp.sh \
    --train_file ./data/mucgec_train.json \
    --dev_file ./data/mucgec_dev.json \
    --num_gpus 2 \
    --batch_size 32 \
    --num_epochs 10 \
    --output_dir ./outputs/mucgec_exp1

# 3. 查看训练日志
tensorboard --logdir=./outputs/mucgec_exp1/tensorboard
```

### 示例2：使用SIGHAN数据训练

```bash
bash scripts/run_ddp.sh \
    --train_file ./data/sighan_train.tsv \
    --dev_file ./data/sighan_dev.tsv \
    --data_format sighan \
    --num_gpus 2 \
    --output_dir ./outputs/sighan_exp1
```

### 示例3：使用生成的数据训练

```bash
# 先生成数据
python scripts/generate_training_data.py

# 然后训练
bash scripts/run_ddp.sh \
    --train_file ./generated_data/train.jsonl \
    --dev_file ./generated_data/dev.jsonl \
    --num_gpus 2
```

---

## 训练过程中的输出

### 启动时的输出

```
==========================================
  Gap-ReLM DDP 多卡联合训练
==========================================

【数据配置】
  训练文件: ./data/mucgec_train.json
  验证文件: ./data/mucgec_dev.json
  数据格式: mucgec

【训练配置】
  训练阶段: joint_finetune
  GPU数量:  4
  Batch Size: 32 (per GPU)
  总Batch: 128
  训练轮数: 10
  学习率:   2e-5

【模型配置】
  预训练模型: hfl/chinese-macbert-base
  最大序列长度: 128
  最大插入数: 3

【输出配置】
  输出目录: ./outputs
  实验名称: gap_relm_joint_training

【功能开关】
  启用插入: true
  启用删除: true
  辅助MLM:  true
  F2优化:   true
  FP16:     true

==========================================

🚀 Starting training...
```

### 训练中的输出

```
Loading data...
Processing 10000 samples...
100%|████████████| 10000/10000 [01:23<00:00]
Loaded 9847 samples

Starting Gap-ReLM Training
Experiment: gap_relm_joint_training
Device: cuda
Distributed: True
World size: 4
==========================================

Epoch 0 [joint]:
100%|████████| 77/77 [05:42<00:00, loss=2.34, lr=1.2e-05]

Evaluating:
100%|████████| 10/10 [00:15<00:00]

Validation metrics:
  total_loss: 2.156
  planner_loss: 1.234
  infill_loss: 0.922
✓ New best! Saved checkpoint to ./outputs/best_stage_c

Epoch 1 [joint]:
100%|████████| 77/77 [05:38<00:00, loss=1.87, lr=1.8e-05]
...
```

### 完成后的输出

```
==========================================
  ✅ Training completed successfully!
==========================================

【输出目录】
  模型检查点: ./outputs/
  TensorBoard: tensorboard --logdir=./outputs/tensorboard
```

---

## 训练后的文件结构

```
./outputs/
├── best_stage_c/              # 最佳模型
│   ├── pytorch_model.bin
│   ├── config.json
│   └── training_args.json
├── checkpoint-1000/           # 定期检查点
├── checkpoint-2000/
├── tensorboard/               # TensorBoard日志
│   └── events.out.tfevents...
└── training.log               # 训练日志
```

---

## 常见问题

### Q1: 如何确认使用了几张GPU？

```bash
# 训练前查看GPU
nvidia-smi

# 训练时查看GPU使用情况
watch -n 1 nvidia-smi
```

### Q2: 内存不足怎么办？

```bash
# 方法1: 减小batch size
--batch_size 16  # 或更小

# 方法2: 减小序列长度
--max_seq_length 64

# 方法3: 使用梯度累积
--gradient_accumulation_steps 2
```

修改脚本中的参数：
```bash
BATCH_SIZE=16                  # 减小
GRADIENT_ACCUMULATION_STEPS=2  # 增大（等效batch size = 16*2*4 = 128）
```

### Q3: 如何恢复训练？

```bash
python scripts/train.py \
    --resume_from ./outputs/checkpoint-1000 \
    ...其他参数...
```

### Q4: 如何只使用部分GPU？

```bash
# 方法1: 指定可见GPU
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_ddp.sh \
    --train_file ... \
    --num_gpus 2

# 方法2: 修改脚本
--num_gpus 2  # 使用前2张GPU
```

### Q5: 训练太慢怎么办？

优化策略：
1. 增加batch size（如果内存允许）
2. 使用FP16混合精度（已默认启用）
3. 增加num_workers（数据加载线程）
4. 使用更快的GPU
5. 启用数据缓存（已默认启用）

---

## 高级配置

### 修改F2优化参数

编辑脚本中的F2参数：
```bash
# 提高召回（F2优化）
DELETE_THRESHOLD=0.3  # 降低删除阈值（更激进）
INSERT_THRESHOLD=0.3  # 降低插入阈值（更激进）
```

### 分阶段训练

```bash
# Stage A: Infiller预训练
TRAINING_STAGE="infiller_pretrain"

# Stage B: Planner训练
TRAINING_STAGE="planner_train"

# Stage C: 联合微调（推荐）
TRAINING_STAGE="joint_finetune"
```

### 消融实验

```bash
# 禁用插入操作
ENABLE_INSERT=false

# 禁用删除操作
ENABLE_DELETE=false

# 禁用辅助MLM
ENABLE_AUX_MLM=false

# 禁用F2优化
ENABLE_F2=false
```

---

## 总结

### ✅ 已确认：joint训练使用Gold Template

- **Planner**：在源序列上预测op和insert
- **Infiller**：在Gold Template上训练填充MASK
- **联合损失**：planner_loss + lambda * infiller_loss
- **避免误差传播**：训练时不用Planner预测的模板

### 🚀 开始训练只需3步

```bash
# 1. 准备数据
# 2. 运行脚本
bash scripts/run_ddp.sh \
    --train_file ./data/train.json \
    --dev_file ./data/dev.json \
    --num_gpus 2

# 3. 等待完成
```

### 📊 监控训练

```bash
# 实时查看日志
tail -f ./outputs/training.log

# 查看TensorBoard
tensorboard --logdir=./outputs/tensorboard

# 查看GPU使用
watch -n 1 nvidia-smi
```

**祝训练顺利！** 🎉
