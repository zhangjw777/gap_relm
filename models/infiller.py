"""
ReLM Infiller 模块
负责填充模板中的 [MASK] 位置
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from transformers import BertModel, BertLMHeadModel


@dataclass
class InfillerOutput:
    """Infiller 输出"""
    logits: torch.Tensor              # [batch, seq_len, vocab_size]
    predictions: Optional[torch.Tensor] = None  # [batch, seq_len]
    infill_loss: Optional[torch.Tensor] = None
    aux_mlm_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


class ReLMInfiller(nn.Module):
    """
    ReLM Infiller
    
    继承 ReLM 的核心思想：在模板上做 MLM 式填空
    输入: [CLS] source [SEP] template [SEP]
    输出: 对 template 中所有 [MASK] 位置的预测
    """
    
    def __init__(
        self,
        encoder: Optional[BertModel] = None,
        vocab_size: int = 21128,
        hidden_size: int = 768,
        share_encoder: bool = True,
        enable_aux_mlm: bool = True,
    ):
        """
        Args:
            encoder: 共享的 BERT 编码器（如果 share_encoder=True）
            vocab_size: 词表大小
            hidden_size: 隐藏层大小
            share_encoder: 是否与 Planner 共享编码器
            enable_aux_mlm: 是否启用辅助 MLM 任务
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.share_encoder = share_encoder
        self.enable_aux_mlm = enable_aux_mlm
        
        # 编码器
        if share_encoder and encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None  # 需要在外部设置或创建
        
        # MLM Head
        self.lm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def set_encoder(self, encoder: BertModel):
        """设置编码器"""
        self.encoder = encoder
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        aux_mlm_labels: Optional[torch.Tensor] = None,
        aux_mlm_weight: float = 0.15,
        return_hidden_states: bool = False,
    ) -> InfillerOutput:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
                       格式: [CLS] source [SEP] template [SEP]
                       或: [CLS] template [SEP] (简化版)
            attention_mask: 注意力掩码 [batch, seq_len]
            inputs_embeds: 输入的 embeddings [batch, seq_len, hidden_size]
                          如果提供，则跳过 encoder，直接使用这些 embeddings
            labels: [MASK] 位置的标签 [batch, seq_len]，非 mask 位置为 -100
            aux_mlm_labels: 辅助 MLM 标签 [batch, seq_len]
            aux_mlm_weight: 辅助 MLM 损失权重
            return_hidden_states: 是否返回隐藏状态
            
        Returns:
            InfillerOutput
        """
        # 编码
        if inputs_embeds is not None:
            # 直接使用传入的 hidden states（用于 P-Tuning 场景）
            hidden_states = inputs_embeds
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # LM Head
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        # 计算损失
        infill_loss = None
        aux_mlm_loss = None
        total_loss = None
        
        if labels is not None:
            infill_loss = self.loss_fn(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
        
        if aux_mlm_labels is not None and self.enable_aux_mlm:
            aux_mlm_loss = self.loss_fn(
                logits.view(-1, self.vocab_size),
                aux_mlm_labels.view(-1)
            )
        
        # 总损失
        if infill_loss is not None and aux_mlm_loss is not None:
            total_loss = infill_loss + aux_mlm_weight * aux_mlm_loss
        elif infill_loss is not None:
            total_loss = infill_loss
        elif aux_mlm_loss is not None:
            total_loss = aux_mlm_loss
        
        # 预测
        predictions = logits.argmax(dim=-1) if labels is None else None
        
        return InfillerOutput(
            logits=logits,
            predictions=predictions,
            infill_loss=infill_loss,
            aux_mlm_loss=aux_mlm_loss,
            total_loss=total_loss,
            hidden_states=hidden_states if return_hidden_states else None
        )
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_token_id: int = 103,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测 [MASK] 位置的 token
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            mask_token_id: [MASK] token ID
            
        Returns:
            (predictions, confidence_scores)
        """
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            
            # 获取预测
            probs = torch.softmax(output.logits, dim=-1)
            predictions = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1)[0]
            
            # 只在 [MASK] 位置替换
            is_mask = input_ids == mask_token_id
            result = input_ids.clone()
            result[is_mask] = predictions[is_mask]
            
            return result, confidence
    
    def get_mask_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_token_id: int = 103,
        top_k: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        获取 [MASK] 位置的 top-k 预测
        
        Returns:
            {
                'top_tokens': [batch, num_masks, top_k],
                'top_probs': [batch, num_masks, top_k],
                'mask_positions': [batch, num_masks]
            }
        """
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            probs = torch.softmax(output.logits, dim=-1)
            
            batch_size = input_ids.shape[0]
            results = {
                'top_tokens': [],
                'top_probs': [],
                'mask_positions': []
            }
            
            for b in range(batch_size):
                mask_positions = (input_ids[b] == mask_token_id).nonzero(as_tuple=True)[0]
                results['mask_positions'].append(mask_positions)
                
                if len(mask_positions) > 0:
                    mask_probs = probs[b, mask_positions]  # [num_masks, vocab_size]
                    top_probs, top_tokens = mask_probs.topk(top_k, dim=-1)
                    results['top_tokens'].append(top_tokens)
                    results['top_probs'].append(top_probs)
                else:
                    results['top_tokens'].append(torch.tensor([], device=input_ids.device))
                    results['top_probs'].append(torch.tensor([], device=input_ids.device))
            
            return results


class DualEncoderInfiller(ReLMInfiller):
    """
    双编码器 Infiller
    
    使用独立的编码器处理 source 和 template
    """
    
    def __init__(
        self,
        source_encoder: Optional[BertModel] = None,
        template_encoder: Optional[BertModel] = None,
        vocab_size: int = 21128,
        hidden_size: int = 768,
        enable_aux_mlm: bool = True,
    ):
        super().__init__(
            encoder=template_encoder,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            share_encoder=False,
            enable_aux_mlm=enable_aux_mlm
        )
        
        self.source_encoder = source_encoder
        self.template_encoder = template_encoder
        
        # 融合层
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
    
    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        template_ids: torch.Tensor,
        template_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> InfillerOutput:
        """
        双编码器前向传播
        
        Args:
            source_ids: 源序列 IDs
            source_mask: 源序列掩码
            template_ids: 模板 IDs
            template_mask: 模板掩码
            labels: 标签
        """
        # 编码源序列
        source_outputs = self.source_encoder(
            input_ids=source_ids,
            attention_mask=source_mask,
            return_dict=True
        )
        source_hidden = source_outputs.last_hidden_state
        
        # 编码模板
        template_outputs = self.template_encoder(
            input_ids=template_ids,
            attention_mask=template_mask,
            return_dict=True
        )
        template_hidden = template_outputs.last_hidden_state
        
        # 融合
        fused_hidden, _ = self.fusion(
            query=template_hidden,
            key=source_hidden,
            value=source_hidden,
            key_padding_mask=(source_mask == 0)
        )
        
        # 残差连接
        fused_hidden = fused_hidden + template_hidden
        
        # LM Head
        logits = self.lm_head(fused_hidden)
        
        # 计算损失
        infill_loss = None
        if labels is not None:
            infill_loss = self.loss_fn(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
        
        return InfillerOutput(
            logits=logits,
            infill_loss=infill_loss,
            total_loss=infill_loss,
            hidden_states=fused_hidden
        )
