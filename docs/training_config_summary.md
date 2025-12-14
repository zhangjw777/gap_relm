# 训练配置总结与确认

## ✅ 核心确认：joint训练使用Gold Template

### 代码证据链

#### 1. 数据加载时生成Gold Template

**文件**: [data/label_generator.py](data\label_generator.py)

```python
@dataclass
class GoldTemplate:
    """Gold Template (用于训练 Infiller)
    
    模板结构: 根据 op 和 insert 标签构建的 token 序列
    - template_tokens: 模板 token 序列 (包含 [MASK] 和保留的字符)
    - gold_tokens: 每个 [MASK] 位置对应的正确 token
    - mask_positions: [MASK] 在模板中的位置列表
    """
    template_tokens: List[str]     # 模板序列
    gold_tokens: List[str]         # [MASK] 位置的正确答案
    mask_positions: List[int]      # [MASK] 的位置索引
    source: str
    target: str
```

Gold Template是在**数据预处理**时就生成的，不是训练时动态生成。

#### 2. Dataset返回Gold Template

**文件**: [data/dataset.py](data\dataset.py) Line ~320-350

```python
def __getitem__(self, idx: int) -> Dict[str, Any]:
    """获取单个样本"""
    sample = self.samples[idx]  # ProcessedSample包含gold_template
    
    # 构建模板序列
    template_text = ''.join(sample.gold_template.template_tokens)
    template_encoding = self.tokenizer(...)
    
    # 构建infill标签（MASK位置的正确答案）
    gold_tokens = sample.gold_template.gold_tokens
    for i, pos in enumerate(mask_positions):
        if i < len(gold_tokens):
            infill_labels[pos] = tokenizer.convert_tokens_to_ids(gold_tokens[i])
    
    return {
        'input_ids': source_encoding,        # 源序列
        'template_input_ids': template_ids,  # Gold Template ← 这是预生成的！
        'infill_labels': infill_labels,      # MASK位置答案
        ...
    }
```

#### 3. Model前向使用Gold Template

**文件**: [models/gap_relm.py](models\gap_relm.py) Line ~140-230

```python
def forward(
    self,
    input_ids: torch.Tensor,           # [batch, seq_len] 源序列
    template_input_ids: torch.Tensor,  # [batch, template_len] Gold Template
    training_stage: str = "joint",
    ...
):
    # Planner训练：在源序列上预测
    if training_stage in ["planner", "joint"]:
        planner_output = self.planner(
            hidden_states=encoder_hidden,  # 从源序列编码
            op_labels=op_labels,
            insert_labels=insert_labels,
        )
    
    # Infiller训练：在Gold Template上训练
    if training_stage in ["infiller", "joint"]:
        infiller_output = self.infiller(
            input_ids=template_input_ids,  # ← 使用Gold Template，不是Planner构建的！
            labels=infill_labels,          # MASK位置的正确答案
            ...
        )
    
    # 联合损失
    total_loss = planner_loss + lambda_infill * infiller_loss
```

**关键点**：
- `template_input_ids` 来自Dataset，是**预生成的Gold Template**
- **不是**在训练时用Planner的预测构建的模板
- Planner和Infiller是**独立训练**的，只是损失联合

#### 4. Trainer调用确认

**文件**: [trainers/trainer.py](trainers\trainer.py) Line ~330-350

```python
def _train_epoch(self, epoch, training_stage="joint", ...):
    for batch in train_loader:
        # batch来自Dataset，包含预生成的Gold Template
        outputs = self.model(
            input_ids=batch['input_ids'],              # 源序列
            template_input_ids=batch['template_input_ids'],  # Gold Template
            infill_labels=batch['infill_labels'],      # 正确答案
            training_stage=training_stage,              # "joint"
        )
        
        loss = outputs.total_loss  # planner_loss + infiller_loss
        loss.backward()
```

### 🎯 训练流程图

```
数据文件(json)
    ↓
加载(source, target)对
    ↓
Levenshtein对齐
    ↓
生成标签
    ├─ op_labels (KEEP/DELETE/REPLACE)
    ├─ insert_labels (0~K)
    └─ Gold Template (包含MASK)
    ↓
存入Dataset
    ↓
DataLoader批处理
    ↓
训练循环（joint阶段）
    ├─ Planner(source) → 预测op + insert
    │    ↓
    │   planner_loss（与真实op/insert标签对比）
    │
    └─ Infiller(Gold Template) → 填充MASK
         ↓
        infiller_loss（与正确答案对比）
    ↓
total_loss = planner_loss + λ * infiller_loss
    ↓
反向传播
```

### 📝 为什么这样设计？

#### 优点

1. **训练稳定**：Infiller在正确模板上训练，不受Planner错误影响
2. **避免误差传播**：Planner的预测错误不会累积到Infiller
3. **独立优化**：两个任务可以独立收敛
4. **Teacher Forcing**：Infiller学习在完美模板上填充

#### 推理时的差异

```python
# 训练时（joint）
planner预测op/insert → 计算planner_loss
infiller填充Gold Template → 计算infiller_loss

# 推理时（predict）
planner预测op/insert → 构建预测模板 → infiller填充预测模板 → 输出
```

这就是为什么需要Stage C（joint_finetune）来缓解训练-推理不一致。

---

## 📋 DDP训练配置总结

### 必填参数

```bash
--train_file <path>    # 训练数据（必须）
```

### 推荐参数

```bash
--dev_file <path>      # 验证数据（强烈推荐）
--num_gpus <N>         # GPU数量（默认4）
```

### 关键默认值

```bash
# 训练策略
TRAINING_STAGE="joint_finetune"    # 联合训练（Planner + Infiller）
BATCH_SIZE=32                       # 每GPU batch size
NUM_EPOCHS=10                       # 训练轮数
LEARNING_RATE=2e-5                  # 学习率

# 模型配置
PRETRAINED_MODEL="hfl/chinese-macbert-base"
MAX_SEQ_LENGTH=128
MAX_INSERT_NUM=3

# 功能开关
ENABLE_INSERT=true                  # 启用插入操作
ENABLE_DELETE=true                  # 启用删除操作
ENABLE_AUX_MLM=true                 # 启用辅助MLM
ENABLE_F2=true                      # 启用F2优化
USE_FP16=true                       # 使用混合精度

# 数据格式
DATA_FORMAT="mucgec"                # 支持mucgec/sighan/ecspell/custom/parallel
```

### 脚本位置

```
scripts/
├── run_ddp.sh          # 主训练脚本（DDP多卡）
├── quick_start.sh      # 快速启动模板
└── train.py            # 训练入口程序
```

---

## 🚀 立即开始训练

### 方法1：修改quick_start.sh（最简单）

```bash
# 1. 编辑 scripts/quick_start.sh
vim scripts/quick_start.sh

# 修改这几行：
TRAIN_FILE="./data/mucgec_train.json"  # 你的训练文件
DEV_FILE="./data/mucgec_dev.json"      # 你的验证文件
NUM_GPUS=2                              # 你的GPU数量

# 2. 运行
bash scripts/quick_start.sh
```

### 方法2：直接使用run_ddp.sh

```bash
bash scripts/run_ddp.sh \
    --train_file ./data/mucgec_train.json \
    --dev_file ./data/mucgec_dev.json \
    --num_gpus 2
```

### 方法3：Python API

```python
from gap_relm.config import GapReLMConfig, get_config
from gap_relm.models import GapReLMModel
from gap_relm.data import create_data_loaders
from gap_relm.trainers import GapReLMTrainer

# 配置
config = get_config("default")
config.data.train_file = "./data/mucgec_train.json"
config.data.dev_file = "./data/mucgec_dev.json"
config.training.num_epochs = 10

# 数据
train_loader, dev_loader, _, tokenizer = create_data_loaders(
    train_file=config.data.train_file,
    dev_file=config.data.dev_file,
    ...
)

# 模型
model = GapReLMModel(config)

# 训练
trainer = GapReLMTrainer(model, config, train_loader, dev_loader)
trainer.train()
```

---

## 📊 训练监控

### 查看日志

```bash
# 实时查看
tail -f ./outputs/training.log

# 查看TensorBoard
tensorboard --logdir=./outputs/tensorboard
```

### 查看GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看进程
nvidia-smi pmon -i 0,1,2,3
```

---

## ❓ 常见问题

### Q: joint训练时Infiller用的是什么模板？

**A: Gold Template（预生成的正确模板），不是Planner预测的模板。**

详见本文档"核心确认"部分的代码证据链。

### Q: 那推理时呢？

**A: 推理时才用Planner预测的模板。**

```python
# 训练
infiller(Gold Template) → 与正确答案对比 → loss

# 推理
planner预测 → 构建模板 → infiller填充 → 输出
```

### Q: 为什么不在训练时也用Planner预测的模板？

**A: 避免误差累积。**

如果训练时也用Planner预测的模板：
- Planner预测错误 → 模板错误 → Infiller学到错误映射
- 误差累积 → 训练不稳定

使用Gold Template：
- Planner独立学习预测
- Infiller独立学习填充
- 两个任务互相促进，不互相干扰

### Q: 如何查看当前配置？

运行训练脚本时会打印：

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
  ...
```

---

## 📚 相关文档

- [docs/ddp_training_guide.md](ddp_training_guide.md) - DDP训练详细指南
- [docs/training_workflow_complete_guide.md](training_workflow_complete_guide.md) - 完整训练流程
- [docs/data_processing_guide.md](data_processing_guide.md) - 数据处理指南
- [README.md](../README.md) - 项目总览

---

## 🎉 总结

### ✅ 已确认

1. **joint训练使用Gold Template** - 代码证据充分
2. **Planner和Infiller独立训练** - 避免误差累积
3. **训练配置已优化** - 默认值合理
4. **DDP脚本ready** - 可立即使用

### 🚀 快速开始

```bash
# 只需3步
bash scripts/quick_start.sh
```

或

```bash
bash scripts/run_ddp.sh \
    --train_file <你的训练文件> \
    --dev_file <你的验证文件> \
    --num_gpus <你的GPU数量>
```

**祝训练顺利！** 🎊
