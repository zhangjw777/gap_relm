"""
Gap-ReLM 训练器
支持 DDP 多卡训练、混合精度、分阶段训练、TensorBoard 日志
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..models import GapReLMModel
from ..config import GapReLMConfig, TrainingStage
from .scheduler import get_optimizer, get_scheduler, ScheduledSamplingScheduler


logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """训练状态"""
    global_step: int = 0
    epoch: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    early_stopping_counter: int = 0


class GapReLMTrainer:
    """
    Gap-ReLM 训练器
    
    支持特性:
    - DDP 多卡训练
    - 混合精度训练 (FP16/BF16)
    - 分阶段训练 (Infiller预训练 -> Planner训练 -> 联合微调)
    - Scheduled Sampling
    - TensorBoard 日志
    - 早停
    - 检查点保存/恢复
    """
    
    def __init__(
        self,
        model: GapReLMModel,
        config: GapReLMConfig,
        train_loader,
        dev_loader=None,
        tokenizer=None,
    ):
        """
        Args:
            model: GapReLMModel 实例
            config: 配置对象
            train_loader: 训练数据加载器
            dev_loader: 验证数据加载器
            tokenizer: tokenizer
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.tokenizer = tokenizer
        
        self.training_config = config.training
        self.distributed_config = config.distributed
        
        # 分布式设置
        self.is_distributed = self.distributed_config.use_ddp and dist.is_initialized()
        self.local_rank = self.distributed_config.local_rank
        self.world_size = self.distributed_config.world_size if self.is_distributed else 1
        self.is_main_process = not self.is_distributed or self.local_rank == 0
        
        # 设备
        if torch.cuda.is_available():
            if self.is_distributed:
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # DDP 包装
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.distributed_config.find_unused_parameters
            )
        
        # 获取原始模型引用
        self.raw_model = self.model.module if self.is_distributed else self.model
        
        # 混合精度
        self.use_amp = self.training_config.fp16 or self.training_config.bf16
        self.amp_dtype = torch.bfloat16 if self.training_config.bf16 else torch.float16
        self.scaler = GradScaler() if self.training_config.fp16 else None
        
        # 优化器和调度器
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()
        
        # Scheduled Sampling 调度器
        self.ss_scheduler = None
        if config.ablation.enable_scheduled_sampling:
            total_steps = len(train_loader) * self.training_config.num_epochs
            self.ss_scheduler = ScheduledSamplingScheduler(
                start_ratio=self.training_config.scheduled_sampling_start,
                end_ratio=self.training_config.scheduled_sampling_end,
                num_steps=total_steps
            )
        
        # TensorBoard
        self.writer = None
        if config.use_tensorboard and self.is_main_process:
            log_dir = os.path.join(config.tensorboard_dir, config.experiment_name)
            self.writer = SummaryWriter(log_dir=log_dir)
        
        # 训练状态
        self.state = TrainingState()
        
        # 输出目录
        self.output_dir = self.training_config.output_dir
        if self.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def _setup_optimizer(self):
        """设置优化器和调度器"""
        self.optimizer = get_optimizer(
            self.raw_model,
            learning_rate=self.training_config.learning_rate,
            planner_lr=self.training_config.planner_lr,
            infiller_lr=self.training_config.infiller_lr,
            weight_decay=self.training_config.weight_decay,
        )
        
        # 计算总步数
        num_training_steps = (
            len(self.train_loader) * self.training_config.num_epochs //
            self.training_config.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.training_config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_type="linear"
        )
    
    def train(self):
        """
        完整训练流程
        按阶段进行: A -> B -> C -> D
        """
        logger.info("=" * 60)
        logger.info("Starting Gap-ReLM Training")
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Distributed: {self.is_distributed}")
        logger.info(f"World size: {self.world_size}")
        logger.info("=" * 60)
        
        current_stage = self.training_config.current_stage
        
        # 根据当前阶段决定训练流程
        if current_stage == "infiller_pretrain":
            self._train_stage_a()
            current_stage = "planner_train"
        
        if current_stage == "planner_train":
            self._train_stage_b()
            current_stage = "joint_finetune"
        
        if current_stage == "joint_finetune":
            self._train_stage_c()
        
        # Stage D 是可选的质量增强
        if self.config.ablation.enable_iterative_refinement or self.config.ablation.enable_verifier:
            self._train_stage_d()
        
        # 关闭 TensorBoard
        if self.writer:
            self.writer.close()
        
        logger.info("Training completed!")
    
    def _train_stage_a(self):
        """
        Stage A: Infiller 预训练
        使用 Gold Template (Teacher Forcing)
        """
        logger.info("\n" + "=" * 40)
        logger.info("Stage A: Infiller Pretraining")
        logger.info("=" * 40)
        
        # 冻结 Planner
        self.raw_model.freeze_planner(freeze=True)
        
        for epoch in range(self.training_config.stage_a_epochs):
            self._train_epoch(epoch, training_stage="infiller")
            
            # 验证
            if self.dev_loader and self.is_main_process:
                metrics = self._evaluate(training_stage="infiller")
                self._log_metrics(metrics, epoch, "val")
                
                # 保存最佳模型
                if metrics.get('infill_loss', float('inf')) < self.state.best_metric or self.state.best_metric == 0:
                    self.state.best_metric = metrics.get('infill_loss', 0)
                    self.state.best_epoch = epoch
                    self._save_checkpoint("best_stage_a")
        
        # 解冻 Planner
        self.raw_model.freeze_planner(freeze=False)
    
    def _train_stage_b(self):
        """
        Stage B: Planner 训练
        纯监督序列标注
        """
        logger.info("\n" + "=" * 40)
        logger.info("Stage B: Planner Training")
        logger.info("=" * 40)
        
        # 冻结 Infiller (可选)
        # self.raw_model.freeze_infiller(freeze=True)
        
        for epoch in range(self.training_config.stage_b_epochs):
            self._train_epoch(epoch, training_stage="planner")
            
            if self.dev_loader and self.is_main_process:
                metrics = self._evaluate(training_stage="planner")
                self._log_metrics(metrics, epoch, "val")
                
                if metrics.get('planner_loss', float('inf')) < self.state.best_metric or self.state.best_metric == 0:
                    self.state.best_metric = metrics.get('planner_loss', 0)
                    self.state.best_epoch = epoch
                    self._save_checkpoint("best_stage_b")
    
    def _train_stage_c(self):
        """
        Stage C: 联合微调
        使用 Scheduled Sampling 解决训练/推理不一致
        """
        logger.info("\n" + "=" * 40)
        logger.info("Stage C: Joint Fine-tuning")
        logger.info("=" * 40)
        
        for epoch in range(self.training_config.stage_c_epochs):
            self._train_epoch(epoch, training_stage="joint", use_scheduled_sampling=True)
            
            if self.dev_loader and self.is_main_process:
                metrics = self._evaluate(training_stage="joint")
                self._log_metrics(metrics, epoch, "val")
                
                total_loss = metrics.get('total_loss', float('inf'))
                
                # 早停检查
                if total_loss < self.state.best_metric - self.training_config.early_stopping_threshold:
                    self.state.best_metric = total_loss
                    self.state.best_epoch = epoch
                    self.state.early_stopping_counter = 0
                    self._save_checkpoint("best_stage_c")
                else:
                    self.state.early_stopping_counter += 1
                    if self.state.early_stopping_counter >= self.training_config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
    
    def _train_stage_d(self):
        """
        Stage D: 质量增强
        迭代精炼和 Verifier 训练
        """
        logger.info("\n" + "=" * 40)
        logger.info("Stage D: Quality Enhancement")
        logger.info("=" * 40)
        
        # TODO: 实现迭代精炼训练和 Verifier 训练
        pass
    
    def _train_epoch(
        self,
        epoch: int,
        training_stage: str = "joint",
        use_scheduled_sampling: bool = False
    ):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        planner_loss_sum = 0.0
        infiller_loss_sum = 0.0
        num_batches = 0
        
        # 设置 sampler 的 epoch（用于分布式训练）
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [{training_stage}]",
            disable=not self.is_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = self._move_batch_to_device(batch)
            
            # Scheduled Sampling
            if use_scheduled_sampling and self.ss_scheduler:
                use_predicted = self.ss_scheduler.should_use_predicted_template()
                if use_predicted:
                    batch = self._apply_scheduled_sampling(batch)
                self.ss_scheduler.step()
            
            # 前向传播
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    op_labels=batch['op_labels'],
                    insert_labels=batch['insert_labels'],
                    template_input_ids=batch['template_input_ids'],
                    template_attention_mask=batch['template_attention_mask'],
                    infill_labels=batch['infill_labels'],
                    aux_mlm_labels=batch.get('aux_mlm_labels'),
                    training_stage=training_stage,
                )
                
                loss = outputs.total_loss
                
                if self.training_config.gradient_accumulation_steps > 1:
                    loss = loss / self.training_config.gradient_accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 累积梯度
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                
                # 优化器步进
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1
                
                # 日志
                if self.state.global_step % self.training_config.logging_steps == 0:
                    self._log_training_step(outputs, step)
            
            # 累积损失
            total_loss += loss.item() * self.training_config.gradient_accumulation_steps
            if outputs.planner_loss is not None:
                planner_loss_sum += outputs.planner_loss.item()
            if outputs.infiller_loss is not None:
                infiller_loss_sum += outputs.infiller_loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.training_config.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 保存检查点
            if self.is_main_process and self.state.global_step % self.training_config.save_steps == 0:
                self._save_checkpoint(f"checkpoint-{self.state.global_step}")
        
        # 记录 epoch 指标
        avg_loss = total_loss / num_batches
        if self.writer and self.is_main_process:
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
            if planner_loss_sum > 0:
                self.writer.add_scalar("train/planner_loss", planner_loss_sum / num_batches, epoch)
            if infiller_loss_sum > 0:
                self.writer.add_scalar("train/infiller_loss", infiller_loss_sum / num_batches, epoch)
        
        self.state.epoch = epoch
    
    def _evaluate(self, training_stage: str = "joint") -> Dict[str, float]:
        """验证评估"""
        self.model.eval()
        
        total_loss = 0.0
        planner_loss_sum = 0.0
        infiller_loss_sum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Evaluating", disable=not self.is_main_process):
                batch = self._move_batch_to_device(batch)
                
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        op_labels=batch['op_labels'],
                        insert_labels=batch['insert_labels'],
                        template_input_ids=batch['template_input_ids'],
                        template_attention_mask=batch['template_attention_mask'],
                        infill_labels=batch['infill_labels'],
                        aux_mlm_labels=batch.get('aux_mlm_labels'),
                        training_stage=training_stage,
                    )
                
                total_loss += outputs.total_loss.item()
                if outputs.planner_loss is not None:
                    planner_loss_sum += outputs.planner_loss.item()
                if outputs.infiller_loss is not None:
                    infiller_loss_sum += outputs.infiller_loss.item()
                num_batches += 1
        
        metrics = {
            'total_loss': total_loss / num_batches,
            'planner_loss': planner_loss_sum / num_batches if planner_loss_sum > 0 else 0,
            'infill_loss': infiller_loss_sum / num_batches if infiller_loss_sum > 0 else 0,
        }
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将批次数据移动到设备"""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result
    
    def _apply_scheduled_sampling(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用 Scheduled Sampling
        使用 Planner 预测的模板替换 Gold Template
        """
        # 获取 Planner 预测
        with torch.no_grad():
            encoder_output = self.raw_model.encoder(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_dict=True
            )
            op_preds, insert_preds = self.raw_model.planner.predict(
                encoder_output.last_hidden_state,
                batch['attention_mask']
            )
        
        # 构建预测模板
        from ..models.template_builder import TemplateBuilder
        template_builder = TemplateBuilder(
            tokenizer=self.tokenizer,
            max_seq_length=self.config.model.max_seq_length,
        )
        
        template_result = template_builder.build_template(
            batch['input_ids'],
            batch['attention_mask'],
            op_preds,
            insert_preds
        )
        
        # 更新 batch
        batch['template_input_ids'] = template_result.template_ids
        batch['template_attention_mask'] = template_result.template_mask
        
        # 注意：infill_labels 需要根据新模板重新计算
        # 这里简化处理，保持原标签（实际应用中可能需要更复杂的处理）
        
        return batch
    
    def _log_training_step(self, outputs, step):
        """记录训练步骤日志"""
        if self.writer:
            self.writer.add_scalar("train/loss", outputs.total_loss.item(), self.state.global_step)
            if outputs.planner_loss is not None:
                self.writer.add_scalar("train/planner_loss", outputs.planner_loss.item(), self.state.global_step)
            if outputs.infiller_loss is not None:
                self.writer.add_scalar("train/infiller_loss", outputs.infiller_loss.item(), self.state.global_step)
            self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.state.global_step)
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int, prefix: str):
        """记录指标日志"""
        logger.info(f"Epoch {epoch} {prefix} metrics: {metrics}")
        
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, epoch)
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        self.raw_model.save_pretrained(checkpoint_dir)
        
        # 保存优化器和调度器状态
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'state': {
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'best_metric': self.state.best_metric,
                'best_epoch': self.state.best_epoch,
            }
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # 保存 tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """加载检查点"""
        # 加载模型
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
            map_location=self.device
        )
        self.raw_model.load_state_dict(state_dict)
        
        # 加载训练状态
        training_state = torch.load(
            os.path.join(checkpoint_dir, "training_state.pt"),
            map_location=self.device
        )
        
        self.optimizer.load_state_dict(training_state['optimizer'])
        self.scheduler.load_state_dict(training_state['scheduler'])
        if self.scaler and training_state['scaler']:
            self.scaler.load_state_dict(training_state['scaler'])
        
        self.state.global_step = training_state['state']['global_step']
        self.state.epoch = training_state['state']['epoch']
        self.state.best_metric = training_state['state']['best_metric']
        self.state.best_epoch = training_state['state']['best_epoch']
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
