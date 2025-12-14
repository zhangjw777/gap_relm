"""
Verifier 模块 (可选)
用于验证纠错结果，降低过纠风险
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass
from transformers import BertModel


@dataclass
class VerifierOutput:
    """Verifier 输出"""
    accept_logits: torch.Tensor      # [batch, 2] 接受/拒绝
    accept_probs: torch.Tensor       # [batch] 接受概率
    accepted: torch.Tensor           # [batch] bool
    loss: Optional[torch.Tensor] = None


class Verifier(nn.Module):
    """
    Verifier 模块
    
    输入原句 X 和候选纠正 Ŷ，判断是否接受这个纠正
    用于控制过纠风险
    """
    
    def __init__(
        self,
        encoder: Optional[BertModel] = None,
        hidden_size: int = 768,
        dropout_prob: float = 0.1,
        accept_threshold: float = 0.5,
    ):
        """
        Args:
            encoder: BERT 编码器
            hidden_size: 隐藏层大小
            dropout_prob: Dropout 概率
            accept_threshold: 接受阈值
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.accept_threshold = accept_threshold
        
        self.encoder = encoder
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 2)  # 接受/拒绝
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def set_encoder(self, encoder: BertModel):
        """设置编码器"""
        self.encoder = encoder
    
    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> VerifierOutput:
        """
        前向传播
        
        Args:
            source_ids: 原句 token IDs [batch, seq_len]
            source_mask: 原句掩码 [batch, seq_len]
            candidate_ids: 候选纠正 token IDs [batch, seq_len]
            candidate_mask: 候选纠正掩码 [batch, seq_len]
            labels: 是否接受的标签 [batch]，0=拒绝，1=接受
            
        Returns:
            VerifierOutput
        """
        # 编码原句
        source_outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=source_mask,
            return_dict=True
        )
        source_cls = source_outputs.last_hidden_state[:, 0]  # [batch, hidden_size]
        
        # 编码候选纠正
        candidate_outputs = self.encoder(
            input_ids=candidate_ids,
            attention_mask=candidate_mask,
            return_dict=True
        )
        candidate_cls = candidate_outputs.last_hidden_state[:, 0]  # [batch, hidden_size]
        
        # 拼接
        combined = torch.cat([source_cls, candidate_cls], dim=-1)  # [batch, hidden_size * 2]
        
        # 分类
        logits = self.classifier(combined)  # [batch, 2]
        probs = torch.softmax(logits, dim=-1)
        accept_probs = probs[:, 1]  # 接受概率
        accepted = accept_probs > self.accept_threshold
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return VerifierOutput(
            accept_logits=logits,
            accept_probs=accept_probs,
            accepted=accepted,
            loss=loss
        )
    
    def verify(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        验证候选纠正
        
        Returns:
            (accepted, accept_probs)
        """
        with torch.no_grad():
            output = self.forward(
                source_ids, source_mask,
                candidate_ids, candidate_mask
            )
            return output.accepted, output.accept_probs
    
    def batch_verify(
        self,
        sources: List[str],
        candidates: List[str],
        tokenizer,
        max_length: int = 128,
    ) -> Tuple[List[bool], List[float]]:
        """
        批量验证
        
        Args:
            sources: 原句列表
            candidates: 候选纠正列表
            tokenizer: tokenizer
            max_length: 最大长度
            
        Returns:
            (accepted_list, prob_list)
        """
        device = next(self.parameters()).device
        
        # 编码
        source_encoding = tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        
        candidate_encoding = tokenizer(
            candidates,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        
        # 验证
        accepted, probs = self.verify(
            source_encoding['input_ids'],
            source_encoding['attention_mask'],
            candidate_encoding['input_ids'],
            candidate_encoding['attention_mask']
        )
        
        return accepted.cpu().tolist(), probs.cpu().tolist()


class EditLevelVerifier(nn.Module):
    """
    编辑级别的 Verifier
    
    对每个编辑操作单独判断是否接受
    """
    
    def __init__(
        self,
        encoder: Optional[BertModel] = None,
        hidden_size: int = 768,
        dropout_prob: float = 0.1,
        accept_threshold: float = 0.5,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.accept_threshold = accept_threshold
        self.encoder = encoder
        
        # 编辑级别分类头
        self.edit_classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, 2)
        )
    
    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        edit_positions: List[List[int]],
        edit_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        对每个编辑位置进行验证
        
        Args:
            source_ids: 源序列 IDs
            source_mask: 源序列掩码  
            edit_positions: 每个样本的编辑位置列表
            edit_labels: 每个编辑是否正确的标签
        """
        # 编码
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=source_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        
        # 对每个编辑位置进行分类
        batch_size = source_ids.shape[0]
        all_logits = []
        all_probs = []
        
        for b in range(batch_size):
            positions = edit_positions[b]
            if len(positions) > 0:
                edit_hidden = hidden_states[b, positions]  # [num_edits, hidden_size]
                logits = self.edit_classifier(edit_hidden)  # [num_edits, 2]
                probs = torch.softmax(logits, dim=-1)[:, 1]  # 接受概率
            else:
                logits = torch.tensor([], device=source_ids.device)
                probs = torch.tensor([], device=source_ids.device)
            
            all_logits.append(logits)
            all_probs.append(probs)
        
        return {
            'edit_logits': all_logits,
            'edit_probs': all_probs,
            'edit_accepted': [p > self.accept_threshold for p in all_probs]
        }
