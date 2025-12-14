#!/bin/bash
# 快速启动示例 - 多卡联合训练
# 支持在线动态数据增强和静态数据两种模式

# ========== 修改这里的参数 ==========

# === 模式选择 ===
# 模式1: 在线动态增强（推荐，默认）
#   需要: 干净训练句子 + 固定验证集
# 模式2: 静态预生成数据
#   需要: 已生成好的训练/验证数据

USE_ONLINE_AUGMENT=true           # true=在线增强, false=静态数据

# === 模式1: 在线动态增强 ===
CLEAN_TRAIN_FILE="./data/clean_train.txt"   # 干净训练句子文件
FROZEN_DEV_FILE="./data/frozen_dev.jsonl"   # 固定验证集（用 generate_frozen_dev.py 生成）

# === 模式2: 静态数据 ===
TRAIN_FILE="./data/train.jsonl"   # 预生成的训练数据
DEV_FILE="./data/dev.jsonl"       # 预生成的验证数据

# === 通用配置 ===
NUM_GPUS=2                        # 你的GPU数量
BATCH_SIZE=32                     # 每个GPU的batch size
NUM_EPOCHS=10                     # 训练轮数
OUTPUT_DIR="./outputs/exp1"       # 输出目录

# ========== 不需要修改下面的内容 ==========

if [ "$USE_ONLINE_AUGMENT" = true ]; then
    echo "===========================================" 
    echo "  模式: 在线动态数据增强"
    echo "==========================================="
    
    # 检查干净训练文件
    if [ ! -f "$CLEAN_TRAIN_FILE" ]; then
        echo "❌ 错误：干净训练文件不存在: $CLEAN_TRAIN_FILE"
        echo ""
        echo "请修改脚本中的 CLEAN_TRAIN_FILE 变量"
        echo "或使用 generate_training_data.py 从语料中提取干净句子"
        exit 1
    fi
    
    # 检查固定验证集
    if [ ! -f "$FROZEN_DEV_FILE" ]; then
        echo "⚠️ 警告：固定验证集不存在: $FROZEN_DEV_FILE"
        echo ""
        echo "建议使用 generate_frozen_dev.py 生成固定验证集："
        echo "  python scripts/generate_frozen_dev.py \\"
        echo "      --clean_file data/clean_dev.txt \\"
        echo "      --output_file $FROZEN_DEV_FILE"
        echo ""
        echo "将使用训练文件作为验证集（不推荐）"
        FROZEN_DEV_FILE=""
    fi
    
    echo "✅ 数据文件检查通过"
    echo "  干净训练文件: $CLEAN_TRAIN_FILE"
    echo "  固定验证集:   ${FROZEN_DEV_FILE:-无}"
    echo ""
    
    # 运行训练（在线增强模式）
    bash scripts/run_ddp.sh \
        --train_file "$CLEAN_TRAIN_FILE" \
        --clean_train_file "$CLEAN_TRAIN_FILE" \
        --frozen_dev_file "$FROZEN_DEV_FILE" \
        --online_augment \
        --num_gpus $NUM_GPUS \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --output_dir "$OUTPUT_DIR" \
        --experiment_name "online_augment_$(date +%Y%m%d_%H%M%S)"

else
    echo "===========================================" 
    echo "  模式: 静态预生成数据"
    echo "==========================================="
    
    # 检查训练数据文件
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "❌ 错误：训练文件不存在: $TRAIN_FILE"
        echo ""
        echo "请修改脚本中的 TRAIN_FILE 变量为你的训练数据路径"
        echo "或使用 generate_training_data.py 生成训练数据"
        exit 1
    fi
    
    echo "✅ 数据文件检查通过"
    echo "  训练文件: $TRAIN_FILE"
    echo "  验证文件: ${DEV_FILE:-无}"
    echo ""
    
    # 运行训练（静态数据模式）
    bash scripts/run_ddp.sh \
        --train_file "$TRAIN_FILE" \
        --dev_file "$DEV_FILE" \
        --no_online_augment \
        --num_gpus $NUM_GPUS \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --output_dir "$OUTPUT_DIR" \
        --experiment_name "static_data_$(date +%Y%m%d_%H%M%S)"
fi
