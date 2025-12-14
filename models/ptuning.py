"""
P-Tuning 模块
实现论文中的 P-Tuning 技术，通过可学习的 prompt embeddings 提升模型性能

原理：
- 在输入序列前添加可学习的虚拟 token（prompt）
- 使用 LSTM + MLP 编码 prompt embeddings，增强表达能力
- 不修改预训练模型参数，只学习 prompt 参数
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PTuningEncoder(nn.Module):
    """
    P-Tuning 编码器
    
    将可学习的 prompt embeddings 通过 LSTM 和 MLP 编码
    生成最终的 prompt 表示
    """
    
    def __init__(
        self,
        prompt_length: int,
        hidden_size: int,
        lstm_hidden_size: Optional[int] = None,
        mlp_hidden_size: Optional[int] = None,
        num_lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        use_lstm: bool = True,
        use_mlp: bool = True,
    ):
        """
        Args:
            prompt_length: Prompt 长度（虚拟 token 数量）
            hidden_size: 模型隐藏层大小（与 BERT hidden_size 一致）
            lstm_hidden_size: LSTM 隐藏层大小，默认与 hidden_size 相同
            mlp_hidden_size: MLP 隐藏层大小，默认与 hidden_size 相同
            num_lstm_layers: LSTM 层数
            lstm_dropout: LSTM dropout 概率
            use_lstm: 是否使用 LSTM 编码
            use_mlp: 是否使用 MLP 编码
        """
        super().__init__()
        
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size or hidden_size
        self.mlp_hidden_size = mlp_hidden_size or hidden_size
        self.use_lstm = use_lstm
        self.use_mlp = use_mlp
        
        # Prompt Embeddings
        # 使用 2 * prompt_length 来支持双向 LSTM 的初始化
        self.prompt_embeddings = nn.Embedding(prompt_length, hidden_size)
        
        # LSTM 编码器
        if use_lstm:
            self.prompt_lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=self.lstm_hidden_size // 2,  # 双向 LSTM
                num_layers=num_lstm_layers,
                dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True
            )
        else:
            self.prompt_lstm = None
        
        # MLP 编码器
        if use_mlp:
            input_dim = self.lstm_hidden_size if use_lstm else hidden_size
            self.prompt_mlp = nn.Sequential(
                nn.Linear(input_dim, self.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_size, hidden_size)
            )
        else:
            self.prompt_mlp = None
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 使用正态分布初始化 prompt embeddings
        nn.init.normal_(self.prompt_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        生成 prompt embeddings
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            prompt_embeds: [batch_size, prompt_length, hidden_size]
        """
        # 生成 prompt indices
        prompt_indices = torch.arange(self.prompt_length, device=device)
        
        # 获取 prompt embeddings [prompt_length, hidden_size]
        prompt_embeds = self.prompt_embeddings(prompt_indices)
        
        # 扩展到 batch [batch_size, prompt_length, hidden_size]
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # LSTM 编码
        if self.use_lstm and self.prompt_lstm is not None:
            prompt_embeds, _ = self.prompt_lstm(prompt_embeds)
        
        # MLP 编码
        if self.use_mlp and self.prompt_mlp is not None:
            prompt_embeds = self.prompt_mlp(prompt_embeds)
        
        # Layer Normalization
        prompt_embeds = self.layer_norm(prompt_embeds)
        
        return prompt_embeds


class PTuningWrapper(nn.Module):
    """
    P-Tuning 包装器
    
    将 prompt embeddings 与输入 embeddings 拼接，
    然后通过 BERT 编码器处理
    
    原始 ReLM 的 P-Tuning 结构：
    class PTuningWrapper(nn.Module):
        def __init__(self, model, prompt_length):
            self.prompt_embeddings = nn.Embedding(2*prompt_length, hidden_size)
            self.prompt_lstm = nn.LSTM(...)
            self.prompt_linear = nn.Sequential(...)
    """
    
    def __init__(
        self,
        encoder,
        prompt_length: int = 10,
        hidden_size: int = 768,
        use_lstm: bool = True,
        use_mlp: bool = True,
        prompt_position: str = "prefix",  # "prefix" 或 "both"
    ):
        """
        Args:
            encoder: BERT 编码器
            prompt_length: Prompt 长度
            hidden_size: 隐藏层大小
            use_lstm: 是否使用 LSTM
            use_mlp: 是否使用 MLP
            prompt_position: prompt 位置，"prefix" 表示只在前面，"both" 表示前后都有
        """
        super().__init__()
        
        self.encoder = encoder
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.prompt_position = prompt_position
        
        # Prompt 编码器
        self.prompt_encoder = PTuningEncoder(
            prompt_length=prompt_length,
            hidden_size=hidden_size,
            use_lstm=use_lstm,
            use_mlp=use_mlp,
        )
        
        # 如果使用 "both"，需要额外的后缀 prompt 编码器
        if prompt_position == "both":
            self.suffix_prompt_encoder = PTuningEncoder(
                prompt_length=prompt_length,
                hidden_size=hidden_size,
                use_lstm=use_lstm,
                use_mlp=use_mlp,
            )
        else:
            self.suffix_prompt_encoder = None
    
    def get_prompt_length(self) -> int:
        """获取 prompt 总长度"""
        if self.prompt_position == "both":
            return self.prompt_length * 2
        return self.prompt_length
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        前向传播
        
        在输入序列前（和后）添加 prompt，然后通过编码器处理
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            token_type_ids: token 类型 IDs [batch, seq_len]
            return_dict: 是否返回字典
            
        Returns:
            编码器输出，但 hidden_states 已去除 prompt 部分
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device
        
        # 获取输入的 word embeddings
        input_embeds = self.encoder.embeddings.word_embeddings(input_ids)
        
        # 生成 prompt embeddings
        prompt_embeds = self.prompt_encoder(batch_size, device)
        
        # 拼接 prompt 和输入
        if self.prompt_position == "both" and self.suffix_prompt_encoder is not None:
            suffix_prompt_embeds = self.suffix_prompt_encoder(batch_size, device)
            combined_embeds = torch.cat([prompt_embeds, input_embeds, suffix_prompt_embeds], dim=1)
            total_prompt_len = self.prompt_length * 2
        else:
            combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
            total_prompt_len = self.prompt_length
        
        # 扩展 attention_mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.prompt_length, device=device)
            if self.prompt_position == "both":
                suffix_mask = torch.ones(batch_size, self.prompt_length, device=device)
                combined_mask = torch.cat([prompt_mask, attention_mask, suffix_mask], dim=1)
            else:
                combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # 扩展 token_type_ids
        if token_type_ids is not None:
            prompt_type_ids = torch.zeros(batch_size, self.prompt_length, dtype=torch.long, device=device)
            if self.prompt_position == "both":
                suffix_type_ids = torch.zeros(batch_size, self.prompt_length, dtype=torch.long, device=device)
                combined_type_ids = torch.cat([prompt_type_ids, token_type_ids, suffix_type_ids], dim=1)
            else:
                combined_type_ids = torch.cat([prompt_type_ids, token_type_ids], dim=1)
        else:
            combined_type_ids = None
        
        # 通过编码器（使用 inputs_embeds 而不是 input_ids）
        outputs = self.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            token_type_ids=combined_type_ids,
            return_dict=return_dict,
            **kwargs
        )
        
        # 移除 prompt 部分的 hidden states，保持与原始输入对齐
        if return_dict:
            # 只保留原始输入对应的 hidden states
            if self.prompt_position == "both":
                outputs.last_hidden_state = outputs.last_hidden_state[:, self.prompt_length:-self.prompt_length, :]
            else:
                outputs.last_hidden_state = outputs.last_hidden_state[:, self.prompt_length:, :]
        else:
            # tuple 格式
            hidden_states = outputs[0]
            if self.prompt_position == "both":
                hidden_states = hidden_states[:, self.prompt_length:-self.prompt_length, :]
            else:
                hidden_states = hidden_states[:, self.prompt_length:, :]
            outputs = (hidden_states,) + outputs[1:]
        
        return outputs


class TaskSpecificPTuning(nn.Module):
    """
    任务特定的 P-Tuning
    
    为 Planner 和 Infiller 提供各自独立的 prompt，
    同时共享底层的 BERT 编码器
    
    这种设计可以：
    1. 隔离 Planner 和 Infiller 的梯度冲突
    2. 让每个任务学习自己的 prompt 表示
    3. 保持编码器共享，减少参数量
    """
    
    def __init__(
        self,
        encoder,
        prompt_length: int = 10,
        hidden_size: int = 768,
        use_lstm: bool = True,
        use_mlp: bool = True,
    ):
        """
        Args:
            encoder: 共享的 BERT 编码器
            prompt_length: 每个任务的 prompt 长度
            hidden_size: 隐藏层大小
            use_lstm: 是否使用 LSTM
            use_mlp: 是否使用 MLP
        """
        super().__init__()
        
        self.encoder = encoder
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        
        # Planner 专用 Prompt 编码器
        self.planner_prompt_encoder = PTuningEncoder(
            prompt_length=prompt_length,
            hidden_size=hidden_size,
            use_lstm=use_lstm,
            use_mlp=use_mlp,
        )
        
        # Infiller 专用 Prompt 编码器
        self.infiller_prompt_encoder = PTuningEncoder(
            prompt_length=prompt_length,
            hidden_size=hidden_size,
            use_lstm=use_lstm,
            use_mlp=use_mlp,
        )
    
    def forward_with_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task: str = "planner",  # "planner" 或 "infiller"
        return_dict: bool = True,
        **kwargs
    ):
        """
        带 prompt 的前向传播
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            token_type_ids: token 类型 IDs [batch, seq_len]
            task: 任务类型，"planner" 或 "infiller"
            return_dict: 是否返回字典
            
        Returns:
            编码器输出，hidden_states 已去除 prompt 部分
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 获取输入的 word embeddings
        input_embeds = self.encoder.embeddings.word_embeddings(input_ids)
        
        # 根据任务选择对应的 prompt 编码器
        if task == "planner":
            prompt_embeds = self.planner_prompt_encoder(batch_size, device)
        else:  # infiller
            prompt_embeds = self.infiller_prompt_encoder(batch_size, device)
        
        # 拼接 prompt 和输入（prompt 在前）
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # 扩展 attention_mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.prompt_length, device=device)
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # 扩展 token_type_ids
        if token_type_ids is not None:
            prompt_type_ids = torch.zeros(batch_size, self.prompt_length, dtype=torch.long, device=device)
            combined_type_ids = torch.cat([prompt_type_ids, token_type_ids], dim=1)
        else:
            combined_type_ids = None
        
        # 通过编码器
        outputs = self.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            token_type_ids=combined_type_ids,
            return_dict=return_dict,
            **kwargs
        )
        
        # 移除 prompt 部分的 hidden states
        if return_dict:
            outputs.last_hidden_state = outputs.last_hidden_state[:, self.prompt_length:, :]
        else:
            hidden_states = outputs[0]
            hidden_states = hidden_states[:, self.prompt_length:, :]
            outputs = (hidden_states,) + outputs[1:]
        
        return outputs
    
    def forward_planner(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Planner 前向传播的便捷方法"""
        return self.forward_with_prompt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="planner",
            **kwargs
        )
    
    def forward_infiller(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Infiller 前向传播的便捷方法"""
        return self.forward_with_prompt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="infiller",
            **kwargs
        )
