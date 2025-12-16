#!/bin/bash
# Gap-ReLM DDP 多卡训练启动脚本
# 用于Planner + Infiller联合多任务训练

# ========== 必填参数 ==========
TRAIN_FILE="./static_training_data/train.jsonl"  # 预生成的静态训练数据（带预计算标签）
DEV_FILE="./static_training_data/dev.jsonl"      # 预生成的静态验证数据

# ========== 基础配置 ==========
OUTPUT_DIR="./outputs"                     # 输出目录
EXPERIMENT_NAME="gap_relm_joint_training"  # 实验名称
NUM_GPUS=2                                 # GPU数量（根据实际情况修改）
DATA_FORMAT="mucgec"                       # 数据格式（mucgec/sighan/ecspell/custom/parallel）

# ========== 训练策略 ==========
TRAINING_STAGE="joint_finetune"            # 训练阶段（joint_finetune=联合训练）
NUM_EPOCHS=10                              # 训练轮数
BATCH_SIZE=64                              # 每个GPU的batch size
GRADIENT_ACCUMULATION_STEPS=2              # 梯度累积步数

# ========== 优化器参数 ==========
LEARNING_RATE=2e-5                         # 学习率
WARMUP_RATIO=0.1                           # 预热比例
WEIGHT_DECAY=0.01                          # 权重衰减
MAX_GRAD_NORM=1.0                          # 梯度裁剪

# ========== 模型参数 ==========
PRETRAINED_MODEL="hfl/chinese-macbert-base"  # 预训练模型
MAX_SEQ_LENGTH=128                         # 最大序列长度
MAX_INSERT_NUM=3                           # 最大插入数量K

# ========== 混合精度训练 ==========
USE_FP16=true                              # 使用FP16混合精度（推荐）
USE_BF16=false                             # 使用BF16混合精度（如果GPU支持）

# ========== 数据加载 ==========
NUM_WORKERS=4                              # 数据加载进程数（建议 4-16，在线增强时增大）
PREFETCH_FACTOR=4                          # 每个worker预取的batch数（默认2，可增大到4-8）
CACHE_DIR="./cache"                        # 缓存目录
USE_CACHE=true                             # 是否使用缓存
LAZY_LOAD=true                            # 惰性加载模式（推荐大数据集>100万样本使用，节省内存）

# ========== 预计算 tokenize 数据（最高效模式） ==========
# 如果使用预计算的二进制数据，设置 USE_TOKENIZED_DATA=true
# 并指定数据文件前缀（不含 .bin/.idx 后缀）
USE_TOKENIZED_DATA=true                   # 是否使用预计算 tokenize 数据
TRAIN_DATA_PREFIX="./tokenized_data/train"                       # 训练数据前缀，如 ./tokenized_data/train
DEV_DATA_PREFIX="./tokenized_data/dev"                         # 验证数据前缀，如 ./tokenized_data/dev
TEST_DATA_PREFIX="./tokenized_data/test"                        # 测试数据前缀（可选）

# ========== 在线动态数据增强 ==========
# 注意：使用预生成静态数据时，设置 ONLINE_AUGMENT=false
ONLINE_AUGMENT=false                       # 关闭在线动态数据增强（使用预生成静态数据）
CLEAN_TRAIN_FILE=""                        # 干净训练句子文件（留空：使用预生成数据）
FROZEN_DEV_FILE=""                         # 固定验证集文件（留空：使用DEV_FILE）
CLEAN_FILE_FORMAT="txt"                    # 干净文件格式（txt/json/jsonl）
P_CORRUPT=0.7                              # 造错概率
BASE_LAMBDA=1.5                            # 基础泊松参数
PI_SKIP=0.2                                # 删字概率
PI_MULTIPLY=0.3                            # 重复字概率
PI_REPLACE=0.5                             # 错字概率

# 长度自适应 λ
ENABLE_LENGTH_ADAPTIVE=true                # 启用长度自适应λ
MIN_LENGTH_FOR_LAMBDA=20                   # λ最小值对应的句子长度
MAX_LENGTH_FOR_LAMBDA=80                   # λ最大值对应的句子长度
MIN_LAMBDA=1.0                             # 最小λ值
MAX_LAMBDA=3.0                             # 最大λ值

# ========== 消融实验开关 ==========
ENABLE_INSERT=true                         # 启用插入操作
ENABLE_DELETE=true                         # 启用删除操作
ENABLE_AUX_MLM=true                        # 启用辅助MLM任务

# ========== MASK 模式 ==========
# Full MASK 模式（ReLM 风格）：模板格式为 [CLS] source [SEP] [MASK]*N [SEP]
# 稀疏 MASK 模式：只在编辑位置放置 [MASK]
FULL_MASK_MODE=true                        # true=Full MASK模式（默认），false=稀疏MASK模式

# ========== P-Tuning 配置 ==========
ENABLE_PTUNING=true                        # 启用P-Tuning（默认开启）
PTUNING_PROMPT_LENGTH=10                   # Prompt长度
PTUNING_USE_LSTM=true                      # 使用LSTM编码prompt
PTUNING_SHARED=false                       # Planner/Infiller共享prompt（false=各自独立）

# ========== F2优化 ==========
ENABLE_F2=true                             # 启用F2优化
DELETE_THRESHOLD=0.3                       # 删除阈值
INSERT_THRESHOLD=0.3                       # 插入阈值

# ========== 日志和保存 ==========
LOGGING_STEPS=100                          # 日志输出步数
SAVE_STEPS=500                             # 保存检查点步数
EVAL_STEPS=500                             # 评估步数

# ========== 其他 ==========
SEED=42                                    # 随机种子

# ========== 解析命令行参数（覆盖默认值）==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --dev_file)
            DEV_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --data_format)
            DATA_FORMAT="$2"
            shift 2
            ;;
        --training_stage)
            TRAINING_STAGE="$2"
            shift 2
            ;;
        --no_ptuning)
            ENABLE_PTUNING=false
            shift
            ;;
        --ptuning_prompt_length)
            PTUNING_PROMPT_LENGTH="$2"
            shift 2
            ;;
        --ptuning_no_lstm)
            PTUNING_USE_LSTM=false
            shift
            ;;
        --ptuning_shared)
            PTUNING_SHARED=true
            shift
            ;;
        # 在线动态数据增强参数
        --online_augment)
            ONLINE_AUGMENT=true
            shift
            ;;
        --no_online_augment)
            ONLINE_AUGMENT=false
            shift
            ;;
        --clean_train_file)
            CLEAN_TRAIN_FILE="$2"
            shift 2
            ;;
        --frozen_dev_file)
            FROZEN_DEV_FILE="$2"
            shift 2
            ;;
        --clean_file_format)
            CLEAN_FILE_FORMAT="$2"
            shift 2
            ;;
        --p_corrupt)
            P_CORRUPT="$2"
            shift 2
            ;;
        --base_lambda)
            BASE_LAMBDA="$2"
            shift 2
            ;;
        --pi_skip)
            PI_SKIP="$2"
            shift 2
            ;;
        --pi_multiply)
            PI_MULTIPLY="$2"
            shift 2
            ;;
        --pi_replace)
            PI_REPLACE="$2"
            shift 2
            ;;
        --no_length_adaptive)
            ENABLE_LENGTH_ADAPTIVE=false
            shift
            ;;
        --min_lambda)
            MIN_LAMBDA="$2"
            shift 2
            ;;
        --max_lambda)
            MAX_LAMBDA="$2"
            shift 2
            ;;
        --lazy_load)
            LAZY_LOAD=true
            shift
            ;;
        --full_mask_mode)
            FULL_MASK_MODE=true
            shift
            ;;
        --sparse_mask_mode)
            FULL_MASK_MODE=false
            shift
            ;;
        --tokenized_data)
            USE_TOKENIZED_DATA=true
            shift
            ;;
        --train_data_prefix)
            TRAIN_DATA_PREFIX="$2"
            shift 2
            ;;
        --dev_data_prefix)
            DEV_DATA_PREFIX="$2"
            shift 2
            ;;
        --test_data_prefix)
            TEST_DATA_PREFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Available options:"
            echo "  --train_file <path>          训练数据文件（必填）"
            echo "  --dev_file <path>            验证数据文件（可选）"
            echo "  --output_dir <path>          输出目录"
            echo "  --experiment_name <name>     实验名称"
            echo "  --num_gpus <N>               GPU数量"
            echo "  --batch_size <N>             batch size"
            echo "  --num_epochs <N>             训练轮数"
            echo "  --learning_rate <float>      学习率"
            echo "  --data_format <format>       数据格式（mucgec/sighan/...）"
            echo "  --training_stage <stage>     训练阶段（joint_finetune/infiller_pretrain/planner_train）"
            echo "  --no_ptuning                 关闭P-Tuning（消融实验）"
            echo "  --ptuning_prompt_length <N>  P-Tuning prompt长度"
            echo "  --ptuning_no_lstm            P-Tuning不使用LSTM"
            echo "  --ptuning_shared             P-Tuning使用共享prompt"
            echo ""
            echo "在线动态数据增强选项:"
            echo "  --online_augment             启用在线动态数据增强（默认）"
            echo "  --no_online_augment          关闭在线增强，使用预生成数据"
            echo "  --clean_train_file <path>    干净训练句子文件"
            echo "  --frozen_dev_file <path>     固定验证集文件"
            echo "  --clean_file_format <fmt>    干净文件格式（txt/json/jsonl）"
            echo "  --p_corrupt <float>          造错概率（0-1）"
            echo "  --base_lambda <float>        基础泊松参数"
            echo "  --pi_skip <float>            删字概率"
            echo "  --pi_multiply <float>        重复字概率"
            echo "  --pi_replace <float>         错字概率"
            echo "  --no_length_adaptive         关闭长度自适应λ"
            echo "  --min_lambda <float>         最小λ值"
            echo "  --max_lambda <float>         最大λ值"
            echo ""
            echo "大数据集内存优化选项:"
            echo "  --lazy_load                  启用惰性加载（推荐>100万样本数据集使用）"
            echo ""
            echo "MASK 模式选项:"
            echo "  --full_mask_mode             Full MASK 模式（ReLM 风格，默认）"
            echo "  --sparse_mask_mode           稀疏 MASK 模式（只在编辑位置放 MASK）"
            echo ""
            echo "预计算 tokenize 数据选项（最高效模式）:"
            echo "  --tokenized_data             使用预计算的二进制数据"
            echo "  --train_data_prefix <path>   训练数据前缀（不含 .bin/.idx）"
            echo "  --dev_data_prefix <path>     验证数据前缀（不含 .bin/.idx）"
            echo "  --test_data_prefix <path>    测试数据前缀（可选）"
            exit 1
            ;;
    esac
done

# ========== 检查必需参数 ==========
if [ -z "$TRAIN_FILE" ]; then
    echo "❌ Error: --train_file is required"
    echo ""
    echo "Usage example:"
    echo "  bash scripts/run_ddp.sh \\"
    echo "    --train_file ./data/mucgec_train.json \\"
    echo "    --dev_file ./data/mucgec_dev.json \\"
    echo "    --num_gpus 2"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [ -n "$DEV_FILE" ] && [ ! -f "$DEV_FILE" ]; then
    echo "⚠️ Warning: Dev file not found: $DEV_FILE"
    echo "Will skip validation during training."
    DEV_FILE=""
fi

# ========== 创建输出目录 ==========
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# ========== 打印训练配置 ==========
echo ""
echo "=========================================="
echo "  Gap-ReLM DDP 多卡联合训练"
echo "=========================================="
echo ""
echo "【数据配置】"
echo "  训练文件: $TRAIN_FILE"
echo "  验证文件: ${DEV_FILE:-None}"
echo "  数据格式: $DATA_FORMAT"
echo ""
echo "【训练配置】"
echo "  训练阶段: $TRAINING_STAGE"
echo "  GPU数量:  $NUM_GPUS"
echo "  Batch Size: $BATCH_SIZE (per GPU)"
echo "  总Batch: $((NUM_GPUS * BATCH_SIZE))"
echo "  训练轮数: $NUM_EPOCHS"
echo "  学习率:   $LEARNING_RATE"
echo ""
echo "【模型配置】"
echo "  预训练模型: $PRETRAINED_MODEL"
echo "  最大序列长度: $MAX_SEQ_LENGTH"
echo "  最大插入数: $MAX_INSERT_NUM"
echo ""
echo "【输出配置】"
echo "  输出目录: $OUTPUT_DIR"
echo "  实验名称: $EXPERIMENT_NAME"
echo ""
echo "【功能开关】"
echo "  启用插入: $ENABLE_INSERT"
echo "  启用删除: $ENABLE_DELETE"
echo "  辅助MLM:  $ENABLE_AUX_MLM"
echo "  P-Tuning: $ENABLE_PTUNING"
echo "  Prompt长度: $PTUNING_PROMPT_LENGTH"
echo "  F2优化:   $ENABLE_F2"
echo "  FP16:     $USE_FP16"
echo "  Full MASK模式: $FULL_MASK_MODE"
echo ""
echo "【在线动态增强】"
echo "  启用在线增强: $ONLINE_AUGMENT"
if [ "$USE_TOKENIZED_DATA" = true ]; then
    echo ""
    echo "【预计算 tokenize 数据】"
    echo "  训练数据前缀: $TRAIN_DATA_PREFIX"
    echo "  验证数据前缀: ${DEV_DATA_PREFIX:-None}"
    echo "  测试数据前缀: ${TEST_DATA_PREFIX:-None}"
elif [ "$ONLINE_AUGMENT" = true ]; then
    echo "  干净训练文件: ${CLEAN_TRAIN_FILE:-$TRAIN_FILE}"
    echo "  固定验证集:   ${FROZEN_DEV_FILE:-$DEV_FILE}"
    echo "  造错概率:     $P_CORRUPT"
    echo "  基础λ:        $BASE_LAMBDA"
    echo "  长度自适应:   $ENABLE_LENGTH_ADAPTIVE"
    if [ "$ENABLE_LENGTH_ADAPTIVE" = true ]; then
        echo "  λ范围:        [$MIN_LAMBDA, $MAX_LAMBDA]"
    fi
fi
echo ""
echo "=========================================="
echo ""

# ========== 构建训练命令 ==========
CMD="torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/train.py \
    --train_file \"$TRAIN_FILE\" \
    --data_format $DATA_FORMAT \
    --output_dir \"$OUTPUT_DIR\" \
    --experiment_name \"$EXPERIMENT_NAME\" \
    --training_stage $TRAINING_STAGE \
    --pretrained_model $PRETRAINED_MODEL \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_insert_num $MAX_INSERT_NUM \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_workers $NUM_WORKERS \
    --prefetch_factor $PREFETCH_FACTOR \
    --cache_dir \"$CACHE_DIR\" \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --seed $SEED"

# 添加可选参数
if [ -n "$DEV_FILE" ]; then
    CMD="$CMD --dev_file \"$DEV_FILE\""
fi

if [ "$USE_CACHE" = true ]; then
    CMD="$CMD"  # 默认启用缓存
else
    CMD="$CMD --no_cache"
fi

if [ "$USE_FP16" = true ]; then
    CMD="$CMD --fp16"
fi

if [ "$USE_BF16" = true ]; then
    CMD="$CMD --bf16"
fi

if [ "$ENABLE_INSERT" = false ]; then
    CMD="$CMD --no_insert"
fi

if [ "$ENABLE_DELETE" = false ]; then
    CMD="$CMD --no_delete"
fi

if [ "$ENABLE_AUX_MLM" = false ]; then
    CMD="$CMD --no_aux_mlm"
fi

if [ "$ENABLE_F2" = false ]; then
    CMD="$CMD --no_f2"
fi

# P-Tuning 配置
if [ "$ENABLE_PTUNING" = false ]; then
    CMD="$CMD --no_ptuning"
fi

if [ "$PTUNING_USE_LSTM" = false ]; then
    CMD="$CMD --ptuning_no_lstm"
fi

if [ "$PTUNING_SHARED" = true ]; then
    CMD="$CMD --ptuning_shared"
fi

CMD="$CMD --ptuning_prompt_length $PTUNING_PROMPT_LENGTH"

# MASK 模式配置
if [ "$FULL_MASK_MODE" = true ]; then
    CMD="$CMD --full_mask_mode"
else
    CMD="$CMD --sparse_mask_mode"
fi

# 预计算 tokenize 数据配置（最高效模式）
if [ "$USE_TOKENIZED_DATA" = true ]; then
    CMD="$CMD --tokenized_data"

    if [ -n "$TRAIN_DATA_PREFIX" ]; then
        CMD="$CMD --train_data_prefix \"$TRAIN_DATA_PREFIX\""
    else
        echo "❌ Error: --train_data_prefix is required when using --tokenized_data"
        exit 1
    fi

    if [ -n "$DEV_DATA_PREFIX" ]; then
        CMD="$CMD --dev_data_prefix \"$DEV_DATA_PREFIX\""
    fi

    if [ -n "$TEST_DATA_PREFIX" ]; then
        CMD="$CMD --test_data_prefix \"$TEST_DATA_PREFIX\""
    fi
# 在线动态数据增强配置
elif [ "$ONLINE_AUGMENT" = true ]; then
    CMD="$CMD --online_augment"

    # 干净训练文件（如果指定）
    if [ -n "$CLEAN_TRAIN_FILE" ]; then
        CMD="$CMD --clean_train_file \"$CLEAN_TRAIN_FILE\""
    fi

    # 固定验证集（如果指定）
    if [ -n "$FROZEN_DEV_FILE" ]; then
        CMD="$CMD --frozen_dev_file \"$FROZEN_DEV_FILE\""
    fi

    # 造错参数
    CMD="$CMD --p_corrupt $P_CORRUPT"
    CMD="$CMD --base_lambda $BASE_LAMBDA"
    CMD="$CMD --pi_skip $PI_SKIP"
    CMD="$CMD --pi_multiply $PI_MULTIPLY"
    CMD="$CMD --pi_replace $PI_REPLACE"
    CMD="$CMD --clean_file_format $CLEAN_FILE_FORMAT"

    # 长度自适应配置
    if [ "$ENABLE_LENGTH_ADAPTIVE" = true ]; then
        CMD="$CMD --enable_length_adaptive"
        CMD="$CMD --min_lambda $MIN_LAMBDA"
        CMD="$CMD --max_lambda $MAX_LAMBDA"
        CMD="$CMD --min_length_for_lambda $MIN_LENGTH_FOR_LAMBDA"
        CMD="$CMD --max_length_for_lambda $MAX_LENGTH_FOR_LAMBDA"
    else
        CMD="$CMD --no_length_adaptive"
    fi
# 静态数据模式（不使用预计算 tokenize，也不使用在线增强）
else
    CMD="$CMD --no_online_augment"

    # 惰性加载（仅在静态 JSONL 数据模式下生效）
    if [ "$LAZY_LOAD" = true ]; then
        CMD="$CMD --lazy_load"
    fi
fi

# ========== 运行训练 ==========
echo "🚀 Starting training..."
echo ""

eval $CMD

# ========== 训练完成 ==========
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✅ Training completed successfully!"
    echo "=========================================="
    echo ""
    echo "【输出目录】"
    echo "  模型检查点: $OUTPUT_DIR/"
    echo "  TensorBoard: tensorboard --logdir=$OUTPUT_DIR/tensorboard"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "  ❌ Training failed!"
    echo "=========================================="
    exit 1
fi
