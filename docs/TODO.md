---

## 三、待办事项

### 3.1 **【核心重构】架构调整为整句重述模式（2025-12-16）**

#### 当前问题
**当前Gap-ReLM（稀疏MASK）：**
- Infiller输入：`[CLS] template [SEP]`（❌ 缺少source上下文）
- Template：`这 个 [MASK] 子`（只在错误位置有MASK）
- Infiller：只预测少量MASK位置（填字任务）
- **问题：** 与原版ReLM差异过大，Infiller无法利用source信息

#### 目标架构：整句重述（Sentence Paraphrase）

**设计思路：**
保持原版ReLM的`src+trg`拼接结构，同时支持变长序列的插入/删除能力。

**核心改动：**

1️⃣ **Planner（保持不变）**
```python
输入：[CLS] source [SEP]
输出：op_labels（KEEP/DELETE/REPLACE） + insert_labels
作用：预测编辑操作，并由此计算目标序列长度N
```
**设计理由：** 让模型通过预测细粒度的编辑操作（而非直接预测长度N），可以更好地学习句子结构变化规律。

2️⃣ **Template Builder（关键修改）**
```python
# 当前（稀疏MASK）
template = "这 个 [MASK] 子"  # 只在REPLACE位置放MASK

# 新版（全句MASK）
target_length = calculate_length(op_labels, insert_labels)
template = "[MASK] " * target_length  # N个MASK
```

3️⃣ **Infiller（重构输入格式）**
```python
# 当前输入
[CLS] template [SEP]

# 新版输入（对齐原版ReLM）
[CLS] [P] source [SEP] [P] [MASK]*N [SEP]
         ↑ prompt      ↑ prompt

输出：并行预测整句的每个token（所有位置都有label）
```

4️⃣ **训练方式（保持Teacher Forcing）**
```python
# Infiller预训练阶段
gold_template = [CLS] [P] source [SEP] [P] 这 个 句 子 [SEP]
labels = 所有target位置都计算loss（不再用-100忽略KEEP位置）

# 联合训练阶段
仍使用Scheduled Sampling解决训练推理不一致
```

#### 架构对比

| 维度 | 当前Gap-ReLM | 新架构 | 原版ReLM |
|------|-------------|--------|---------|
| **Infiller输入** | `[template]` | `[src] [SEP] [template]` | `[src] [SEP] [trg]` |
| **Template类型** | 稀疏MASK | 全MASK | 部分MASK |
| **预测模式** | 填字（少量位置） | 整句重述（全并行） | 整句重述 |
| **变长能力** | ✅ 支持 | ✅ 支持 | ❌ 不支持 |
| **源上下文** | ❌ 缺失 | ✅ 完整 | ✅ 完整 |

#### 实现清单

**核心文件修改：**
- [ ] `models/template_builder.py` - 改为生成全MASK的template
- [ ] `data/dataset.py` - Infiller输入改为`source + template`拼接
- [ ] `data/data_loader.py` - Collator需调整label对齐逻辑
- [ ] `models/infiller.py` - 更新输入格式说明和forward逻辑
- [ ] `models/gap_relm.py` - P-Tuning需为Infiller构建拼接输入
- [ ] `trainers/trainer.py` - Scheduled Sampling的batch构建

**数据重新生成：**
- [ ] 重新生成gold_template（全句标签，非稀疏MASK）
- [ ] 更新label_generator.py的模板生成逻辑

**测试验证：**
- [ ] 单元测试：验证template长度计算正确性
- [ ] 集成测试：验证Infiller输入格式正确
- [ ] 对比实验：与原版ReLM在等长样本上的效果对比

#### 设计优势分析

✅ **为什么op_labels不直接预测长度N？**
- 预测KEEP/DELETE/REPLACE等细粒度操作，模型能更好地学习句子结构变化
- 直接预测长度N太抽象，BERT难以捕捉这种全局数值规律
- 编辑操作是局部特征，更符合BERT的token-level建模能力

✅ **为什么Infiller要看到source？**
- 原版ReLM设计精髓：通过同时编码src和trg，让模型学习上下文对应关系
- 整句重述需要源句语义信息，不能只靠template
- 对齐原版架构，便于后续与baseline对比

---

### 3.2 后续优化（可选）
- [ ] 实现方案 C：共享域 prompt + 任务独立 prompt（分层）
- [ ] 添加 prompt 初始化策略选项（随机、从词表采样等）
- [ ] 支持 deep prompt tuning（每层都有 prompt）
- [ ] 添加更多错误类型（如语序错误）
- [ ] 实现基于困惑度的自适应造错

---