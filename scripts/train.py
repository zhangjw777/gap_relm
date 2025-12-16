"""
Gap-ReLM 训练启动脚本
支持单卡和多卡 DDP 训练
"""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GapReLMConfig, get_config
from models import GapReLMModel
from data import create_data_loaders, create_online_data_loaders, create_tokenized_data_loaders
from trainers import GapReLMTrainer


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        return None, None, None
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train Gap-ReLM model")
    
    # 基础参数
    parser.add_argument("--config", type=str, default="default",
                        help="Configuration name or path to config file")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data file")
    parser.add_argument("--dev_file", type=str, default=None,
                        help="Path to validation data file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="gap_relm",
                        help="Experiment name")
    
    # 模型参数
    parser.add_argument("--pretrained_model", type=str, default="hfl/chinese-macbert-base",
                        help="Pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--max_insert_num", type=int, default=3,
                        help="Maximum insert number K")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # 混合精度
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 training")
    
    # 分阶段训练
    parser.add_argument("--training_stage", type=str, default="infiller_pretrain",
                        choices=["infiller_pretrain", "planner_train", "joint_finetune"],
                        help="Training stage")
    parser.add_argument("--stage_a_epochs", type=int, default=3,
                        help="Stage A epochs")
    parser.add_argument("--stage_b_epochs", type=int, default=3,
                        help="Stage B epochs")
    parser.add_argument("--stage_c_epochs", type=int, default=4,
                        help="Stage C epochs")
    
    # 消融实验
    parser.add_argument("--enable_gap", action="store_true", default=True,
                        help="Enable Gap (insert/delete)")
    parser.add_argument("--no_gap", action="store_true",
                        help="Disable Gap (equivalent to ReLM)")
    parser.add_argument("--no_insert", action="store_true",
                        help="Disable insert operation")
    parser.add_argument("--no_delete", action="store_true",
                        help="Disable delete operation")
    parser.add_argument("--no_aux_mlm", action="store_true",
                        help="Disable auxiliary MLM")
    parser.add_argument("--enable_refinement", action="store_true",
                        help="Enable iterative refinement")
    parser.add_argument("--enable_verifier", action="store_true",
                        help="Enable verifier")
    
    # Full MASK 模式（ReLM 风格）
    mask_mode_group = parser.add_mutually_exclusive_group()
    mask_mode_group.add_argument("--full_mask_mode", action="store_true", default=True,
                                  help="Use full MASK mode (ReLM style: [CLS] src [SEP] [MASK]*N [SEP]) (default)")
    mask_mode_group.add_argument("--sparse_mask_mode", action="store_true",
                                  help="Use sparse MASK mode (only MASK at edit positions)")
    
    # P-Tuning 配置
    parser.add_argument("--no_ptuning", action="store_true",
                        help="Disable P-Tuning (for ablation study)")
    parser.add_argument("--ptuning_prompt_length", type=int, default=10,
                        help="P-Tuning prompt length")
    parser.add_argument("--ptuning_no_lstm", action="store_true",
                        help="Disable LSTM in P-Tuning")
    parser.add_argument("--ptuning_shared", action="store_true",
                        help="Use shared P-Tuning for Planner and Infiller")
    
    # F2 优化
    parser.add_argument("--no_f2", action="store_true",
                        help="Disable F2 optimization")
    parser.add_argument("--delete_threshold", type=float, default=0.5,
                        help="Delete threshold")
    parser.add_argument("--insert_threshold", type=float, default=0.5,
                        help="Insert threshold")
    
    # 数据参数
    parser.add_argument("--data_format", type=str, default="mucgec",
                        choices=["mucgec", "sighan", "ecspell", "custom", "parallel"],
                        help="Data format")
    parser.add_argument("--alignment_algorithm", type=str, default="levenshtein",
                        choices=["levenshtein", "difflib"],
                        help="Alignment algorithm")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches to prefetch per worker (increase for faster data loading)")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable data caching")
    
    # ========== 惰性加载参数（大规模数据集内存优化） ==========
    parser.add_argument("--lazy_load", action="store_true",
                        help="Enable lazy loading for large precomputed datasets (memory-efficient, recommended for >1M samples)")
    
    # ========== 预计算 tokenize 数据参数（最高效模式） ==========
    parser.add_argument("--tokenized_data", action="store_true",
                        help="Use pre-tokenized binary data format (most efficient, recommended for large-scale training)")
    parser.add_argument("--train_data_prefix", type=str, default=None,
                        help="Path prefix for tokenized training data (without .bin/.idx suffix)")
    parser.add_argument("--dev_data_prefix", type=str, default=None,
                        help="Path prefix for tokenized dev data (without .bin/.idx suffix)")
    parser.add_argument("--test_data_prefix", type=str, default=None,
                        help="Path prefix for tokenized test data (without .bin/.idx suffix)")

    
    # ========== 在线动态数据增强参数 ==========
    parser.add_argument("--online_augment", action="store_true", default=True,
                        help="Enable online dynamic data augmentation (default: True)")
    parser.add_argument("--no_online_augment", action="store_true",
                        help="Disable online augmentation, use pre-generated training data")
    parser.add_argument("--clean_train_file", type=str, default=None,
                        help="Path to clean sentences file for online augmentation (required if --online_augment)")
    parser.add_argument("--frozen_dev_file", type=str, default=None,
                        help="Path to frozen dev set (pre-generated errors) for stable evaluation")
    parser.add_argument("--clean_file_format", type=str, default="txt",
                        choices=["txt", "json", "jsonl"],
                        help="Format of clean sentences file")
    parser.add_argument("--clean_text_field", type=str, default="text",
                        help="Text field name in JSON/JSONL file")
    
    # 在线增强造错参数
    parser.add_argument("--p_corrupt", type=float, default=0.7,
                        help="Probability of corrupting a sentence (online augment)")
    parser.add_argument("--base_lambda", type=float, default=1.5,
                        help="Base Poisson lambda for number of edits (online augment)")
    parser.add_argument("--pi_skip", type=float, default=0.2,
                        help="Probability of skip (delete) error type")
    parser.add_argument("--pi_multiply", type=float, default=0.3,
                        help="Probability of multiply (insert) error type")
    parser.add_argument("--pi_replace", type=float, default=0.5,
                        help="Probability of replace error type")
    parser.add_argument("--max_edits_per_sent", type=int, default=4,
                        help="Maximum edits per sentence (online augment)")
    parser.add_argument("--max_insert_k", type=int, default=3,
                        help="Maximum characters to insert in multiply error")
    
    # 长度自适应 λ 参数
    parser.add_argument("--enable_length_adaptive", action="store_true", default=True,
                        help="Enable length-adaptive lambda (default: True)")
    parser.add_argument("--no_length_adaptive", action="store_true",
                        help="Disable length-adaptive lambda")
    parser.add_argument("--min_length_for_lambda", type=int, default=20,
                        help="Sentence length for minimum lambda")
    parser.add_argument("--max_length_for_lambda", type=int, default=80,
                        help="Sentence length for maximum lambda")
    parser.add_argument("--min_lambda", type=float, default=1.0,
                        help="Minimum lambda value")
    parser.add_argument("--max_lambda", type=float, default=3.0,
                        help="Maximum lambda value")
    parser.add_argument("--use_ratio_mode", action="store_true",
                        help="Use error ratio mode instead of length-adaptive lambda")
    parser.add_argument("--error_ratio", type=float, default=0.05,
                        help="Error ratio (percentage of sentence length) for ratio mode")
    
    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation steps")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Log level")
    
    # 恢复训练
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")
    
    # 分布式
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    # 种子
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    
    # 验证混合精度配置
    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both --fp16 and --bf16 at the same time. Please choose one.")
    
    # 设置分布式
    rank, world_size, local_rank = setup_distributed()
    is_distributed = rank is not None
    is_main_process = not is_distributed or rank == 0
    
    if is_distributed:
        args.local_rank = local_rank
    
    # 设置日志（只在主进程设置详细日志）
    if is_main_process:
        setup_logging(args.log_level)
    else:
        # 非主进程：只显示WARNING及以上
        logging.basicConfig(level=logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    # 设置种子
    set_seed(args.seed)
    
    # 加载或创建配置
    if os.path.isfile(args.config):
        config = GapReLMConfig.load(args.config)
    else:
        config = get_config(args.config)
    
    # 更新配置
    config.experiment_name = args.experiment_name
    config.model.pretrained_model_name = args.pretrained_model
    config.model.max_seq_length = args.max_seq_length
    config.model.max_insert_num = args.max_insert_num
    
    config.data.train_file = args.train_file
    config.data.dev_file = args.dev_file
    config.data.data_format = args.data_format
    config.data.alignment_algorithm = args.alignment_algorithm
    config.data.num_workers = args.num_workers
    config.data.cache_dir = args.cache_dir
    config.data.use_cache = not args.no_cache
    
    config.training.num_epochs = args.num_epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.warmup_ratio = args.warmup_ratio
    config.training.weight_decay = args.weight_decay
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.fp16 = args.fp16
    config.training.bf16 = args.bf16
    config.training.output_dir = args.output_dir
    config.training.logging_steps = args.logging_steps
    config.training.save_steps = args.save_steps
    config.training.eval_steps = args.eval_steps
    config.training.current_stage = args.training_stage
    config.training.stage_a_epochs = args.stage_a_epochs
    config.training.stage_b_epochs = args.stage_b_epochs
    config.training.stage_c_epochs = args.stage_c_epochs
    config.training.seed = args.seed
    
    # 消融实验配置
    if args.no_gap:
        config.ablation.enable_gap = False
        config.ablation.enable_insert = False
        config.ablation.enable_delete = False
    else:
        config.ablation.enable_gap = True
        config.ablation.enable_insert = not args.no_insert
        config.ablation.enable_delete = not args.no_delete
    
    config.ablation.enable_aux_mlm = not args.no_aux_mlm
    config.ablation.enable_iterative_refinement = args.enable_refinement
    config.ablation.enable_verifier = args.enable_verifier
    
    # Full MASK 模式配置（ReLM 风格）
    config.ablation.full_mask_mode = not args.sparse_mask_mode
    
    # P-Tuning 配置
    config.ablation.enable_ptuning = not args.no_ptuning
    config.ablation.ptuning_prompt_length = args.ptuning_prompt_length
    config.ablation.ptuning_use_lstm = not args.ptuning_no_lstm
    config.ablation.ptuning_shared = args.ptuning_shared
    
    # F2 优化配置
    config.f2_optimization.enable_f2_optimization = not args.no_f2
    config.f2_optimization.delete_threshold = args.delete_threshold
    config.f2_optimization.insert_threshold = args.insert_threshold
    
    # 分布式配置
    config.distributed.use_ddp = is_distributed
    config.distributed.local_rank = args.local_rank if is_distributed else -1
    config.distributed.world_size = world_size if is_distributed else 1
    
    if is_main_process:
        logger.info(f"Configuration: {config.experiment_name}")
        logger.info(f"Training file: {args.train_file}")
        logger.info(f"Validation file: {args.dev_file}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Distributed: {is_distributed}, World size: {world_size if is_distributed else 1}")
    
    # 确定数据加载模式
    use_tokenized_data = args.tokenized_data and args.train_data_prefix
    use_online_augment = args.online_augment and not args.no_online_augment and not use_tokenized_data
    
    # 创建数据加载器
    if use_tokenized_data:
        # 预计算 tokenize 模式（最高效）
        if is_main_process:
            logger.info("Using PRE-TOKENIZED binary data format (most efficient)")
            logger.info(f"Train data prefix: {args.train_data_prefix}")
            logger.info(f"Dev data prefix: {args.dev_data_prefix}")
        
        train_loader, dev_loader, _, tokenizer = create_tokenized_data_loaders(
            train_prefix=args.train_data_prefix,
            dev_prefix=args.dev_data_prefix,
            test_prefix=args.test_data_prefix,
            tokenizer_name=args.pretrained_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            enable_aux_mlm=config.ablation.enable_aux_mlm,
            distributed=is_distributed,
            world_size=world_size if is_distributed else 1,
            rank=rank if is_distributed else 0,
        )
    elif use_online_augment:
        # 在线动态数据增强模式
        # 需要提供干净句子文件
        clean_train_file = args.clean_train_file or args.train_file
        frozen_dev_file = args.frozen_dev_file or args.dev_file
        
        if is_main_process:
            logger.info("Using ONLINE dynamic data augmentation")
            logger.info(f"Clean train file: {clean_train_file}")
            logger.info(f"Frozen dev file: {frozen_dev_file}")
            logger.info(f"p_corrupt={args.p_corrupt}, base_lambda={args.base_lambda}")
            logger.info(f"pi_skip={args.pi_skip}, pi_multiply={args.pi_multiply}, pi_replace={args.pi_replace}")
            if args.enable_length_adaptive and not args.no_length_adaptive:
                logger.info(f"Length adaptive λ: [{args.min_lambda}, {args.max_lambda}]")
        
        train_loader, dev_loader, _, tokenizer = create_online_data_loaders(
            clean_train_file=clean_train_file,
            frozen_dev_file=frozen_dev_file,
            test_file=None,
            tokenizer_name=args.pretrained_model,
            max_seq_length=args.max_seq_length,
            max_insert_num=args.max_insert_num,
            enable_insert=config.ablation.enable_insert,
            enable_delete=config.ablation.enable_delete,
            alignment_algorithm=args.alignment_algorithm,
            data_format=args.data_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            normalize_text=True,
            enable_aux_mlm=config.ablation.enable_aux_mlm,
            # 在线增强参数
            p_corrupt=args.p_corrupt,
            base_lambda=args.base_lambda,
            pi_skip=args.pi_skip,
            pi_multiply=args.pi_multiply,
            pi_replace=args.pi_replace,
            max_edits_per_sent=args.max_edits_per_sent,
            max_insert_k=args.max_insert_k,
            # 长度自适应配置
            enable_length_adaptive=args.enable_length_adaptive and not args.no_length_adaptive,
            min_length_for_lambda=args.min_length_for_lambda,
            max_length_for_lambda=args.max_length_for_lambda,
            min_lambda=args.min_lambda,
            max_lambda=args.max_lambda,
            use_ratio_mode=args.use_ratio_mode,
            error_ratio=args.error_ratio,
            # 分布式
            distributed=is_distributed,
            world_size=world_size if is_distributed else 1,
            rank=rank if is_distributed else 0,
            # 文件格式
            clean_file_format=args.clean_file_format,
            clean_text_field=args.clean_text_field,
            # MASK 模式
            full_mask_mode=config.ablation.full_mask_mode,
        )
    else:
        # 静态数据模式（使用预生成的训练数据）
        if is_main_process:
            if args.lazy_load:
                logger.info("Using STATIC pre-generated training data with LAZY LOADING (memory-efficient)")
            else:
                logger.info("Using STATIC pre-generated training data")
        
        train_loader, dev_loader, _, tokenizer = create_data_loaders(
            train_file=args.train_file,
            dev_file=args.dev_file,
            tokenizer_name=args.pretrained_model,
            max_seq_length=args.max_seq_length,
            max_insert_num=args.max_insert_num,
            enable_insert=config.ablation.enable_insert,
            enable_delete=config.ablation.enable_delete,
            alignment_algorithm=args.alignment_algorithm,
            data_format=args.data_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            enable_aux_mlm=config.ablation.enable_aux_mlm,
            distributed=is_distributed,
            world_size=world_size if is_distributed else 1,
            rank=rank if is_distributed else 0,
            lazy_load=args.lazy_load,
            full_mask_mode=config.ablation.full_mask_mode,
        )
    
    if is_main_process:
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        if dev_loader:
            logger.info(f"Dev samples: {len(dev_loader.dataset)}")
    
    # 创建模型
    model = GapReLMModel(config, pretrained_model_name=args.pretrained_model)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 创建训练器
    trainer = GapReLMTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        dev_loader=dev_loader,
        tokenizer=tokenizer,
    )
    
    # 恢复训练
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        if is_main_process:
            logger.info(f"Resumed from {args.resume_from}")
    
    # 开始训练
    trainer.train()
    
    # 清理
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
