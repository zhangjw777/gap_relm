"""
Gap-ReLM 主模型
整合 Encoder、Planner、Template Builder、Infiller、Verifier
支持 P-Tuning 消融实验
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from transformers import BertModel, BertConfig, AutoModel

from .planner import EditPlanner, PlannerOutput, create_f2_weights
from .infiller import ReLMInfiller, InfillerOutput
from .template_builder import TemplateBuilder, InferenceTemplateBuilder
from .verifier import Verifier
from .ptuning import PTuningEncoder, TaskSpecificPTuning


@dataclass
class GapReLMOutput:
    """Gap-ReLM 模型输出"""
    # Planner 输出
    planner_output: Optional[PlannerOutput] = None
    
    # Infiller 输出
    infiller_output: Optional[InfillerOutput] = None
    
    # 最终预测
    predictions: Optional[torch.Tensor] = None
    
    # Verifier 输出
    verifier_accepted: Optional[torch.Tensor] = None
    
    # 总损失
    total_loss: Optional[torch.Tensor] = None
    planner_loss: Optional[torch.Tensor] = None
    infiller_loss: Optional[torch.Tensor] = None
    
    # 隐藏状态 (用于调试)
    encoder_hidden: Optional[torch.Tensor] = None


class GapReLMModel(nn.Module):
    """
    Gap-ReLM 主模型
    
    架构:
    1. 共享 Encoder (MacBERT)
    2. P-Tuning (可选，Planner/Infiller 各自独立 prompt)
    3. Edit Planner (Op Head + Insert Head)
    4. Template Builder
    5. ReLM Infiller
    6. (可选) Verifier
    7. (可选) 迭代精炼
    """
    
    def __init__(
        self,
        config,
        pretrained_model_name: str = "hfl/chinese-macbert-base",
    ):
        """
        Args:
            config: GapReLMConfig 配置对象
            pretrained_model_name: 预训练模型名称
        """
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        self.ablation_config = config.ablation
        self.f2_config = config.f2_optimization
        
        # 加载预训练编码器
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # P-Tuning (可选)
        self.ptuning = None
        self.planner_ptuning = None
        self.infiller_ptuning = None
        
        if self.ablation_config.enable_ptuning:
            self._init_ptuning(hidden_size)
        
        # Edit Planner
        self.planner = EditPlanner(
            hidden_size=hidden_size,
            num_op_labels=self.model_config.num_op_labels,
            max_insert_num=self.model_config.max_insert_num,
            dropout_prob=self.model_config.classifier_dropout,
            enable_insert=self.ablation_config.enable_insert,
            enable_delete=self.ablation_config.enable_delete,
        )
        
        # ReLM Infiller
        self.infiller = ReLMInfiller(
            encoder=self.encoder if self.model_config.share_encoder else None,
            vocab_size=self.model_config.vocab_size,
            hidden_size=hidden_size,
            share_encoder=self.model_config.share_encoder,
            enable_aux_mlm=self.ablation_config.enable_aux_mlm,
        )
        
        # 如果不共享编码器，需要创建独立的 infiller 编码器
        if not self.model_config.share_encoder:
            self.infiller_encoder = AutoModel.from_pretrained(pretrained_model_name)
            self.infiller.set_encoder(self.infiller_encoder)
        
        # Verifier (可选)
        self.verifier = None
        if self.ablation_config.enable_verifier:
            self.verifier = Verifier(
                encoder=self.encoder,
                hidden_size=hidden_size,
                dropout_prob=self.model_config.classifier_dropout,
                accept_threshold=self.ablation_config.verifier_threshold,
            )
        
        # F2 优化权重
        self.op_weights = None
        self.insert_weights = None
        if self.f2_config.enable_f2_optimization:
            self._init_f2_weights()
    
    def _init_ptuning(self, hidden_size: int):
        """初始化 P-Tuning 模块"""
        prompt_length = self.ablation_config.ptuning_prompt_length
        use_lstm = self.ablation_config.ptuning_use_lstm
        use_mlp = self.ablation_config.ptuning_use_mlp
        
        if self.ablation_config.ptuning_shared:
            # 共享 P-Tuning：Planner 和 Infiller 使用相同的 prompt
            self.ptuning = TaskSpecificPTuning(
                encoder=self.encoder,
                prompt_length=prompt_length,
                hidden_size=hidden_size,
                use_lstm=use_lstm,
                use_mlp=use_mlp,
            )
        else:
            # 独立 P-Tuning：Planner 和 Infiller 各自有独立的 prompt 编码器
            self.planner_ptuning = PTuningEncoder(
                prompt_length=prompt_length,
                hidden_size=hidden_size,
                use_lstm=use_lstm,
                use_mlp=use_mlp,
            )
            self.infiller_ptuning = PTuningEncoder(
                prompt_length=prompt_length,
                hidden_size=hidden_size,
                use_lstm=use_lstm,
                use_mlp=use_mlp,
            )
    
    def _init_f2_weights(self):
        """初始化 F2 优化权重"""
        self.op_weights, self.insert_weights = create_f2_weights(
            num_op_labels=self.model_config.num_op_labels,
            max_insert_num=self.model_config.max_insert_num,
            op_delete_weight=self.f2_config.op_delete_weight,
            op_replace_weight=self.f2_config.op_replace_weight,
            op_keep_weight=self.f2_config.op_keep_weight,
            insert_positive_weight=self.f2_config.insert_positive_weight,
            insert_zero_weight=self.f2_config.insert_zero_weight,
        )
    
    def _move_weights_to_device(self, device):
        """将权重移动到指定设备"""
        if self.op_weights is not None:
            self.op_weights = self.op_weights.to(device)
        if self.insert_weights is not None:
            self.insert_weights = self.insert_weights.to(device)
    
    def _encode_with_ptuning(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str = "planner"
    ) -> torch.Tensor:
        """
        带 P-Tuning 的编码
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            task: 任务类型 "planner" 或 "infiller"
            
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if not self.ablation_config.enable_ptuning:
            # 不使用 P-Tuning，直接编码
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return encoder_output.last_hidden_state
        
        # 获取输入的 word embeddings
        input_embeds = self.encoder.embeddings.word_embeddings(input_ids)
        
        # 根据配置选择 prompt 编码器
        if self.ablation_config.ptuning_shared and self.ptuning is not None:
            # 共享模式：使用 TaskSpecificPTuning
            if task == "planner":
                prompt_embeds = self.ptuning.planner_prompt_encoder(batch_size, device)
            else:
                prompt_embeds = self.ptuning.infiller_prompt_encoder(batch_size, device)
        else:
            # 独立模式：各自的 prompt 编码器
            if task == "planner" and self.planner_ptuning is not None:
                prompt_embeds = self.planner_ptuning(batch_size, device)
            elif task == "infiller" and self.infiller_ptuning is not None:
                prompt_embeds = self.infiller_ptuning(batch_size, device)
            else:
                # 没有对应的 ptuning，直接编码
                encoder_output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                return encoder_output.last_hidden_state
        
        prompt_length = prompt_embeds.shape[1]
        
        # 拼接 prompt 和输入（prompt 在前）
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # 扩展 attention_mask
        prompt_mask = torch.ones(batch_size, prompt_length, device=device)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # 通过编码器
        encoder_output = self.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            return_dict=True
        )
        
        # 移除 prompt 部分的 hidden states，保持与原始输入对齐
        hidden_states = encoder_output.last_hidden_state[:, prompt_length:, :]
        
        return hidden_states
    
    def _infiller_predict_with_ptuning(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_token_id: int = 103,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带 P-Tuning 的 Infiller 预测
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            mask_token_id: [MASK] token ID
            
        Returns:
            (predictions, confidence_scores)
        """
        with torch.no_grad():
            if self.ablation_config.enable_ptuning:
                # 使用 P-Tuning 编码
                hidden_states = self._encode_with_ptuning(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task="infiller"
                )
                # 通过 LM head
                logits = self.infiller.lm_head(hidden_states)
                
                # 获取预测
                probs = torch.softmax(logits, dim=-1)
                predictions = probs.argmax(dim=-1)
                confidence = probs.max(dim=-1)[0]
                
                # 只在 [MASK] 位置替换
                is_mask = input_ids == mask_token_id
                result = input_ids.clone()
                result[is_mask] = predictions[is_mask]
                
                return result, confidence
            else:
                # 不使用 P-Tuning，直接调用 infiller.predict
                return self.infiller.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=mask_token_id
                )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_labels: Optional[torch.Tensor] = None,
        insert_labels: Optional[torch.Tensor] = None,
        template_input_ids: Optional[torch.Tensor] = None,
        template_attention_mask: Optional[torch.Tensor] = None,
        infill_labels: Optional[torch.Tensor] = None,
        aux_mlm_labels: Optional[torch.Tensor] = None,
        training_stage: str = "joint",
        **kwargs
    ) -> GapReLMOutput:
        """
        前向传播
        
        Args:
            input_ids: 源序列 IDs [batch, seq_len]
            attention_mask: 源序列掩码 [batch, seq_len]
            op_labels: 操作标签 [batch, seq_len]
            insert_labels: 插入标签 [batch, seq_len]
            template_input_ids: 模板 IDs [batch, template_len]
            template_attention_mask: 模板掩码 [batch, template_len]
            infill_labels: Infiller 标签 [batch, template_len]
            aux_mlm_labels: 辅助 MLM 标签 [batch, seq_len]
            training_stage: 训练阶段 ("planner", "infiller", "joint")
            
        Returns:
            GapReLMOutput
        """
        device = input_ids.device
        self._move_weights_to_device(device)
        
        # 编码源序列（带 P-Tuning）
        encoder_hidden = self._encode_with_ptuning(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="planner"
        )
        
        planner_output = None
        infiller_output = None
        planner_loss = None
        infiller_loss = None
        total_loss = None
        
        # Planner 前向
        if training_stage in ["planner", "joint"]:
            planner_output = self.planner(
                hidden_states=encoder_hidden,
                attention_mask=attention_mask,
                op_labels=op_labels,
                insert_labels=insert_labels,
                op_weights=self.op_weights,
                insert_weights=self.insert_weights,
            )
            planner_loss = planner_output.total_loss
        
        # Infiller 前向
        if training_stage in ["infiller", "joint"] and template_input_ids is not None:
            # 注意：aux_mlm_labels 是基于源序列生成的，形状是 [batch, seq_len]
            # 但 Infiller 的输入是模板序列，形状可能是 [batch, template_len]
            # 两者长度可能不同，因此 aux_mlm_loss 应该在源序列上单独计算，
            # 而不是传给 Infiller（Infiller 只计算 infill_loss）
            
            # 如果启用 P-Tuning，需要先编码模板
            if self.ablation_config.enable_ptuning:
                infiller_hidden = self._encode_with_ptuning(
                    input_ids=template_input_ids,
                    attention_mask=template_attention_mask,
                    task="infiller"
                )
                infiller_output = self.infiller(
                    inputs_embeds=infiller_hidden,
                    attention_mask=template_attention_mask,
                    labels=infill_labels,
                    aux_mlm_labels=None,  # 不在模板上计算 aux_mlm_loss
                    aux_mlm_weight=self.config.training.mu_aux,
                )
            else:
                infiller_output = self.infiller(
                    input_ids=template_input_ids,
                    attention_mask=template_attention_mask,
                    labels=infill_labels,
                    aux_mlm_labels=None,  # 不在模板上计算 aux_mlm_loss
                    aux_mlm_weight=self.config.training.mu_aux,
                )
            infiller_loss = infiller_output.total_loss
            
            # 单独在源序列上计算辅助 MLM 损失
            if aux_mlm_labels is not None and self.ablation_config.enable_aux_mlm:
                # 使用已编码的 encoder_hidden (源序列)
                aux_logits = self.infiller.lm_head(encoder_hidden)  # [batch, seq_len, vocab]
                aux_mlm_loss = torch.nn.functional.cross_entropy(
                    aux_logits.view(-1, self.model_config.vocab_size),
                    aux_mlm_labels.view(-1),
                    ignore_index=-100
                )
                # 将 aux_mlm_loss 加到 infiller_loss
                if infiller_loss is not None:
                    infiller_loss = infiller_loss + self.config.training.mu_aux * aux_mlm_loss
                else:
                    infiller_loss = self.config.training.mu_aux * aux_mlm_loss
        
        # 计算总损失
        if planner_loss is not None and infiller_loss is not None:
            total_loss = planner_loss + self.config.training.lambda_infill * infiller_loss
        elif planner_loss is not None:
            total_loss = planner_loss
        elif infiller_loss is not None:
            total_loss = infiller_loss
        
        return GapReLMOutput(
            planner_output=planner_output,
            infiller_output=infiller_output,
            total_loss=total_loss,
            planner_loss=planner_loss,
            infiller_loss=infiller_loss,
            encoder_hidden=encoder_hidden,
        )
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
        use_iterative_refinement: bool = False,
        refinement_rounds: int = 2,
        refinement_mask_ratio: float = 0.15,
        use_verifier: bool = False,
    ) -> Dict[str, Any]:
        """
        推理预测
        
        Args:
            input_ids: 源序列 IDs
            attention_mask: 源序列掩码
            tokenizer: tokenizer
            use_iterative_refinement: 是否使用迭代精炼
            refinement_rounds: 精炼轮数
            refinement_mask_ratio: 精炼 mask 比例
            use_verifier: 是否使用 verifier
            
        Returns:
            预测结果字典
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # 编码（带 P-Tuning）
        encoder_hidden = self._encode_with_ptuning(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="planner"
        )
        
        # Planner 预测
        op_preds, insert_preds = self.planner.predict(
            encoder_hidden,
            attention_mask,
            delete_threshold=self.f2_config.delete_threshold if self.f2_config.enable_threshold_calibration else 0,
            insert_threshold=self.f2_config.insert_threshold if self.f2_config.enable_threshold_calibration else 0,
        )
        
        # 构建模板
        template_builder = InferenceTemplateBuilder(
            tokenizer=tokenizer,
            max_seq_length=self.model_config.max_seq_length,
            max_insert_per_sentence=self.f2_config.max_insert_per_sentence,
            max_insert_ratio=self.f2_config.max_insert_ratio,
        )
        
        template_result = template_builder.build_template(
            input_ids, attention_mask, op_preds, insert_preds
        )
        
        # Infiller 预测（带 P-Tuning）
        predictions, confidence = self._infiller_predict_with_ptuning(
            template_result.template_ids,
            template_result.template_mask,
            mask_token_id=tokenizer.mask_token_id
        )
        
        # 迭代精炼
        if use_iterative_refinement and self.ablation_config.enable_iterative_refinement:
            for r in range(refinement_rounds):
                # 创建精炼模板
                new_template, mask_positions = template_builder.create_refinement_template(
                    predictions, confidence, refinement_mask_ratio
                )
                
                # 重新预测（带 P-Tuning）
                predictions, confidence = self._infiller_predict_with_ptuning(
                    new_template,
                    template_result.template_mask,
                    mask_token_id=tokenizer.mask_token_id
                )
        
        # Verifier
        verifier_accepted = None
        if use_verifier and self.verifier is not None:
            verifier_accepted, _ = self.verifier.verify(
                input_ids, attention_mask,
                predictions, template_result.template_mask
            )
        
        # 解码预测结果
        decoded_predictions = []
        for b in range(batch_size):
            pred_ids = predictions[b]
            # 移除特殊 token
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            decoded_predictions.append(pred_text)
        
        return {
            'predictions': predictions,
            'decoded_predictions': decoded_predictions,
            'op_predictions': op_preds,
            'insert_predictions': insert_preds,
            'template_ids': template_result.template_ids,
            'confidence': confidence,
            'verifier_accepted': verifier_accepted,
        }
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # 保存配置
        self.config.save(os.path.join(save_path, "config.json"))
    
    @classmethod
    def from_pretrained(cls, load_path: str, config=None):
        """加载模型"""
        import os
        from config import GapReLMConfig
        
        # 加载配置
        if config is None:
            config = GapReLMConfig.load(os.path.join(load_path, "config.json"))
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        state_dict = torch.load(
            os.path.join(load_path, "pytorch_model.bin"),
            map_location='cpu'
        )
        model.load_state_dict(state_dict)
        
        return model
    
    def get_trainable_parameters(self, stage: str = "joint") -> List[nn.Parameter]:
        """
        获取指定训练阶段的可训练参数
        
        Args:
            stage: "planner", "infiller", "joint"
        """
        params = []
        
        if stage in ["planner", "joint"]:
            params.extend(self.planner.parameters())
            params.extend(self.encoder.parameters())
            # P-Tuning 参数（Planner 专用）
            if self.ablation_config.enable_ptuning:
                if self.planner_ptuning is not None:
                    params.extend(self.planner_ptuning.parameters())
                if self.ptuning is not None:
                    params.extend(self.ptuning.planner_prompt_encoder.parameters())
        
        if stage in ["infiller", "joint"]:
            params.extend(self.infiller.parameters())
            if not self.model_config.share_encoder:
                params.extend(self.infiller_encoder.parameters())
            # P-Tuning 参数（Infiller 专用）
            if self.ablation_config.enable_ptuning:
                if self.infiller_ptuning is not None:
                    params.extend(self.infiller_ptuning.parameters())
                if self.ptuning is not None:
                    params.extend(self.ptuning.infiller_prompt_encoder.parameters())
        
        if stage == "verifier" and self.verifier is not None:
            params.extend(self.verifier.parameters())
        
        return params
    
    def freeze_encoder(self, freeze: bool = True):
        """冻结/解冻编码器"""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        
        if not self.model_config.share_encoder and hasattr(self, 'infiller_encoder'):
            for param in self.infiller_encoder.parameters():
                param.requires_grad = not freeze
    
    def freeze_planner(self, freeze: bool = True):
        """冻结/解冻 Planner"""
        for param in self.planner.parameters():
            param.requires_grad = not freeze
    
    def freeze_infiller(self, freeze: bool = True):
        """冻结/解冻 Infiller"""
        for param in self.infiller.parameters():
            param.requires_grad = not freeze
    
    def freeze_ptuning(self, freeze: bool = True):
        """冻结/解冻 P-Tuning 参数"""
        if self.planner_ptuning is not None:
            for param in self.planner_ptuning.parameters():
                param.requires_grad = not freeze
        if self.infiller_ptuning is not None:
            for param in self.infiller_ptuning.parameters():
                param.requires_grad = not freeze
        if self.ptuning is not None:
            for param in self.ptuning.parameters():
                param.requires_grad = not freeze
