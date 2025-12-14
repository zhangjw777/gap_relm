"""
学习率调度器和优化器
"""

import math
from typing import Optional, List
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(
    model,
    learning_rate: float = 2e-5,
    planner_lr: Optional[float] = None,
    infiller_lr: Optional[float] = None,
    weight_decay: float = 0.01,
    no_decay_keywords: List[str] = ["bias", "LayerNorm.weight", "layer_norm.weight"],
) -> AdamW:
    """
    创建优化器，支持不同模块使用不同学习率
    
    Args:
        model: GapReLMModel
        learning_rate: 默认学习率
        planner_lr: Planner 学习率（可选）
        infiller_lr: Infiller 学习率（可选）
        weight_decay: 权重衰减
        no_decay_keywords: 不应用权重衰减的参数关键词
        
    Returns:
        AdamW 优化器
    """
    # 分组参数
    optimizer_grouped_parameters = []
    
    # 编码器参数
    encoder_params_decay = []
    encoder_params_no_decay = []
    
    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_keywords):
            encoder_params_no_decay.append(param)
        else:
            encoder_params_decay.append(param)
    
    if encoder_params_decay:
        optimizer_grouped_parameters.append({
            "params": encoder_params_decay,
            "weight_decay": weight_decay,
            "lr": learning_rate
        })
    if encoder_params_no_decay:
        optimizer_grouped_parameters.append({
            "params": encoder_params_no_decay,
            "weight_decay": 0.0,
            "lr": learning_rate
        })
    
    # Planner 参数
    planner_lr = planner_lr or learning_rate
    planner_params_decay = []
    planner_params_no_decay = []
    
    for name, param in model.planner.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_keywords):
            planner_params_no_decay.append(param)
        else:
            planner_params_decay.append(param)
    
    if planner_params_decay:
        optimizer_grouped_parameters.append({
            "params": planner_params_decay,
            "weight_decay": weight_decay,
            "lr": planner_lr
        })
    if planner_params_no_decay:
        optimizer_grouped_parameters.append({
            "params": planner_params_no_decay,
            "weight_decay": 0.0,
            "lr": planner_lr
        })
    
    # Infiller 参数
    infiller_lr = infiller_lr or learning_rate
    infiller_params_decay = []
    infiller_params_no_decay = []
    
    for name, param in model.infiller.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_keywords):
            infiller_params_no_decay.append(param)
        else:
            infiller_params_decay.append(param)
    
    if infiller_params_decay:
        optimizer_grouped_parameters.append({
            "params": infiller_params_decay,
            "weight_decay": weight_decay,
            "lr": infiller_lr
        })
    if infiller_params_no_decay:
        optimizer_grouped_parameters.append({
            "params": infiller_params_no_decay,
            "weight_decay": 0.0,
            "lr": infiller_lr
        })
    
    # Verifier 参数（如果存在）
    if hasattr(model, 'verifier') and model.verifier is not None:
        verifier_params_decay = []
        verifier_params_no_decay = []
        
        for name, param in model.verifier.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay_keywords):
                verifier_params_no_decay.append(param)
            else:
                verifier_params_decay.append(param)
        
        if verifier_params_decay:
            optimizer_grouped_parameters.append({
                "params": verifier_params_decay,
                "weight_decay": weight_decay,
                "lr": learning_rate
            })
        if verifier_params_no_decay:
            optimizer_grouped_parameters.append({
                "params": verifier_params_no_decay,
                "weight_decay": 0.0,
                "lr": learning_rate
            })
    
    return AdamW(optimizer_grouped_parameters)


def get_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str = "linear",
) -> LambdaLR:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        scheduler_type: 调度器类型 ("linear", "cosine", "constant")
        
    Returns:
        学习率调度器
    """
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    线性学习率调度，带预热
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    余弦学习率调度，带预热
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    常数学习率，带预热
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class ScheduledSamplingScheduler:
    """
    Scheduled Sampling 调度器
    用于控制使用预测模板训练 Infiller 的比例
    """
    
    def __init__(
        self,
        start_ratio: float = 0.0,
        end_ratio: float = 0.5,
        num_steps: int = 10000,
        schedule_type: str = "linear"
    ):
        """
        Args:
            start_ratio: 起始比例
            end_ratio: 结束比例
            num_steps: 总步数
            schedule_type: 调度类型
        """
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.current_step = 0
    
    def get_ratio(self) -> float:
        """获取当前的采样比例"""
        if self.schedule_type == "linear":
            progress = min(1.0, self.current_step / self.num_steps)
            return self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        elif self.schedule_type == "cosine":
            progress = min(1.0, self.current_step / self.num_steps)
            return self.start_ratio + (self.end_ratio - self.start_ratio) * (
                1 - math.cos(math.pi * progress)
            ) / 2
        else:
            return self.end_ratio
    
    def step(self):
        """更新步数"""
        self.current_step += 1
    
    def should_use_predicted_template(self) -> bool:
        """是否应该使用预测模板"""
        import random
        return random.random() < self.get_ratio()
