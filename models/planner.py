"""
Edit Planner 模块
包含 Op Head 和 Insert Head，用于预测编辑操作
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PlannerOutput:
    """Planner 输出"""
    op_logits: torch.Tensor        # [batch, seq_len, 3]
    insert_logits: torch.Tensor    # [batch, seq_len, K+1]
    op_loss: Optional[torch.Tensor] = None
    insert_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None


class EditPlanner(nn.Module):
    """
    编辑规划器
    
    包含两个预测头:
    1. Op Head: 预测 KEEP/DELETE/REPLACE
    2. Insert Head: 预测每个位置后的插入数量 (0 ~ K)
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_op_labels: int = 3,        # KEEP=0, DELETE=1, REPLACE=2
        max_insert_num: int = 3,       # K
        dropout_prob: float = 0.1,
        enable_insert: bool = True,
        enable_delete: bool = True,
    ):
        """
        Args:
            hidden_size: 编码器隐藏层大小
            num_op_labels: 操作类型数量 (3)
            max_insert_num: 最大插入数量 K
            dropout_prob: Dropout 概率
            enable_insert: 是否启用插入预测
            enable_delete: 是否启用删除预测
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_op_labels = num_op_labels
        self.max_insert_num = max_insert_num
        self.enable_insert = enable_insert
        self.enable_delete = enable_delete
        
        # Op Head: 预测操作类型
        self.op_classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_op_labels)
        )
        
        # Insert Head: 预测插入数量
        if enable_insert:
            self.insert_classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, max_insert_num + 1)  # 0, 1, ..., K
            )
        else:
            self.insert_classifier = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        op_labels: Optional[torch.Tensor] = None,
        insert_labels: Optional[torch.Tensor] = None,
        op_weights: Optional[torch.Tensor] = None,
        insert_weights: Optional[torch.Tensor] = None,
    ) -> PlannerOutput:
        """
        前向传播
        
        Args:
            hidden_states: 编码器输出 [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch, seq_len]
            op_labels: 操作标签 [batch, seq_len]，用于训练
            insert_labels: 插入标签 [batch, seq_len]，用于训练
            op_weights: 操作损失权重 [num_op_labels]，用于 F2 优化
            insert_weights: 插入损失权重 [max_insert_num + 1]，用于 F2 优化
            
        Returns:
            PlannerOutput
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Op Head
        op_logits = self.op_classifier(hidden_states)  # [batch, seq_len, 3]
        
        # Insert Head
        if self.enable_insert and self.insert_classifier is not None:
            insert_logits = self.insert_classifier(hidden_states)  # [batch, seq_len, K+1]
        else:
            insert_logits = torch.zeros(
                batch_size, seq_len, self.max_insert_num + 1,
                device=hidden_states.device
            )
            insert_logits[..., 0] = 1.0  # 默认不插入
        
        # 计算损失
        op_loss = None
        insert_loss = None
        total_loss = None
        
        if op_labels is not None:
            op_loss = self._compute_op_loss(op_logits, op_labels, attention_mask, op_weights)
        
        if insert_labels is not None and self.enable_insert:
            insert_loss = self._compute_insert_loss(
                insert_logits, insert_labels, attention_mask, insert_weights
            )
        
        if op_loss is not None and insert_loss is not None:
            total_loss = op_loss + insert_loss
        elif op_loss is not None:
            total_loss = op_loss
        elif insert_loss is not None:
            total_loss = insert_loss
        
        return PlannerOutput(
            op_logits=op_logits,
            insert_logits=insert_logits,
            op_loss=op_loss,
            insert_loss=insert_loss,
            total_loss=total_loss
        )
    
    def _compute_op_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        weights: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """计算操作分类损失"""
        # 如果禁用删除，将 DELETE 标签转为 KEEP
        if not self.enable_delete:
            labels = labels.clone()
            labels[labels == 1] = 0  # DELETE -> KEEP
        
        if weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, self.num_op_labels)
        labels_flat = labels.view(-1)
        
        return loss_fn(logits_flat, labels_flat)
    
    def _compute_insert_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        weights: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """计算插入数量预测损失"""
        # 裁剪标签到有效范围
        labels = labels.clamp(0, self.max_insert_num)
        
        if weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, self.max_insert_num + 1)
        labels_flat = labels.view(-1)
        
        # 将填充位置标记为 ignore
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            labels_flat = labels_flat.clone()
            labels_flat[mask_flat == 0] = -100
        
        return loss_fn(logits_flat, labels_flat)
    
    def predict(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        delete_threshold: float = 0.5,
        insert_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测编辑操作
        
        Args:
            hidden_states: 编码器输出
            attention_mask: 注意力掩码
            delete_threshold: 删除阈值
            insert_threshold: 插入阈值
            
        Returns:
            (op_predictions, insert_predictions)
        """
        with torch.no_grad():
            output = self.forward(hidden_states, attention_mask)
            
            # Op 预测
            op_probs = torch.softmax(output.op_logits, dim=-1)
            op_preds = op_probs.argmax(dim=-1)
            
            # 应用删除阈值
            if delete_threshold > 0:
                delete_probs = op_probs[..., 1]  # DELETE 概率
                # 只有当概率超过阈值时才预测为 DELETE
                low_confidence_delete = (op_preds == 1) & (delete_probs < delete_threshold)
                op_preds[low_confidence_delete] = 0  # 改为 KEEP
            
            # Insert 预测
            if self.enable_insert:
                insert_probs = torch.softmax(output.insert_logits, dim=-1)
                insert_preds = insert_probs.argmax(dim=-1)
                
                # 应用插入阈值
                if insert_threshold > 0:
                    # 只有当 p(k>0) > threshold 时才插入
                    insert_any_prob = 1 - insert_probs[..., 0]
                    low_confidence_insert = insert_any_prob < insert_threshold
                    insert_preds[low_confidence_insert] = 0
            else:
                insert_preds = torch.zeros_like(op_preds)
            
            return op_preds, insert_preds
    
    def get_confidence(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取预测置信度
        
        Returns:
            (op_confidence, insert_confidence) - 最大类别概率
        """
        with torch.no_grad():
            output = self.forward(hidden_states)
            
            op_probs = torch.softmax(output.op_logits, dim=-1)
            op_confidence = op_probs.max(dim=-1)[0]
            
            if self.enable_insert:
                insert_probs = torch.softmax(output.insert_logits, dim=-1)
                insert_confidence = insert_probs.max(dim=-1)[0]
            else:
                insert_confidence = torch.ones_like(op_confidence)
            
            return op_confidence, insert_confidence


def create_f2_weights(
    num_op_labels: int = 3,
    max_insert_num: int = 3,
    op_delete_weight: float = 3.0,
    op_replace_weight: float = 2.0,
    op_keep_weight: float = 1.0,
    insert_positive_weight: float = 5.0,
    insert_zero_weight: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建 F2 优化的损失权重
    
    Returns:
        (op_weights, insert_weights)
    """
    # Op 权重: [KEEP, DELETE, REPLACE]
    op_weights = torch.tensor(
        [op_keep_weight, op_delete_weight, op_replace_weight],
        dtype=torch.float,
        device=device
    )
    
    # Insert 权重: [0, 1, 2, ..., K]
    insert_weights = torch.tensor(
        [insert_zero_weight] + [insert_positive_weight] * max_insert_num,
        dtype=torch.float,
        device=device
    )
    
    return op_weights, insert_weights
