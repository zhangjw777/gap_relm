"""
Template Builder 模块
根据 Planner 预测构建变长模板

支持两种模式:
1. 稀疏 MASK 模式 (sparse): [CLS] template_with_sparse_masks [SEP]
2. 全 MASK 模式 (full): [CLS] [P] source [SEP] [P] [MASK]*N [SEP]
"""

import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TemplateResult:
    """模板构建结果"""
    template_ids: torch.Tensor       # [batch, max_template_len]
    template_mask: torch.Tensor      # [batch, max_template_len]
    mask_positions: List[List[int]]  # 每个样本的 [MASK] 位置
    source_to_template: List[List[int]]  # 源位置到模板位置的映射
    target_length: Optional[List[int]] = None  # 全 MASK 模式下的 target 长度


@dataclass
class FullMaskTemplateResult:
    """全 MASK 模式的模板构建结果
    
    输入格式: [CLS] source [SEP] [MASK]*N [SEP]
    注意: P-Tuning 的 prompt 在模型中动态添加，这里不包含
    """
    input_ids: torch.Tensor          # [batch, max_seq_len]，包含 source + [MASK]*N
    attention_mask: torch.Tensor     # [batch, max_seq_len]
    target_start_pos: torch.Tensor   # [batch]，target 区域起始位置（第一个 [MASK]）
    target_length: torch.Tensor      # [batch]，target 区域长度 N
    source_length: torch.Tensor      # [batch]，source 长度（不含特殊 token）


class TemplateBuilder:
    """
    模板构建器
    
    根据 Planner 的 op 和 insert 预测，构建用于 Infiller 的变长模板
    """
    
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 512,
        max_insert_per_sentence: int = 6,
        max_insert_ratio: float = 0.1,
        full_mask_mode: bool = False,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_seq_length: 最大模板长度
            max_insert_per_sentence: 每句最大插入数
            max_insert_ratio: 最大插入比例
            full_mask_mode: 是否使用全 MASK 模式
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_insert_per_sentence = max_insert_per_sentence
        self.max_insert_ratio = max_insert_ratio
        self.full_mask_mode = full_mask_mode
        
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
    
    def build_template(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_predictions: torch.Tensor,
        insert_predictions: torch.Tensor,
    ) -> TemplateResult:
        """
        根据预测构建模板
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            op_predictions: 操作预测 [batch, seq_len]
            insert_predictions: 插入预测 [batch, seq_len]
            
        Returns:
            TemplateResult
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 逐样本构建模板
        all_template_ids = []
        all_mask_positions = []
        all_source_to_template = []
        max_template_len = 0
        
        for b in range(batch_size):
            template_ids, mask_positions, source_to_template = self._build_single_template(
                input_ids[b],
                attention_mask[b],
                op_predictions[b],
                insert_predictions[b]
            )
            all_template_ids.append(template_ids)
            all_mask_positions.append(mask_positions)
            all_source_to_template.append(source_to_template)
            max_template_len = max(max_template_len, len(template_ids))
        
        # 对齐到最大长度
        max_template_len = min(max_template_len, self.max_seq_length)
        
        padded_template_ids = []
        padded_template_mask = []
        
        for template_ids in all_template_ids:
            # 截断
            if len(template_ids) > max_template_len:
                template_ids = template_ids[:max_template_len]
            
            # 填充
            pad_len = max_template_len - len(template_ids)
            padded_ids = template_ids + [self.pad_token_id] * pad_len
            padded_mask = [1] * len(template_ids) + [0] * pad_len
            
            padded_template_ids.append(padded_ids)
            padded_template_mask.append(padded_mask)
        
        return TemplateResult(
            template_ids=torch.tensor(padded_template_ids, dtype=torch.long, device=device),
            template_mask=torch.tensor(padded_template_mask, dtype=torch.long, device=device),
            mask_positions=all_mask_positions,
            source_to_template=all_source_to_template
        )
    
    def _build_single_template(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_preds: torch.Tensor,
        insert_preds: torch.Tensor,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        构建单个样本的模板
        
        Returns:
            (template_ids, mask_positions, source_to_template)
        """
        template_ids = []
        mask_positions = []
        source_to_template = []
        
        seq_len = attention_mask.sum().item()
        src_len = seq_len - 2  # 去掉 [CLS] 和 [SEP]
        
        # 风险约束：计算最大允许的插入数
        max_allowed_inserts = min(
            self.max_insert_per_sentence,
            int(src_len * self.max_insert_ratio) + 1
        )
        total_inserts = 0
        
        # [CLS]
        template_ids.append(self.cls_token_id)
        
        # 处理源序列中的每个字符
        for i in range(1, seq_len - 1):  # 跳过 [CLS] 和 [SEP]
            token_id = input_ids[i].item()
            op = op_preds[i].item()
            insert_num = insert_preds[i].item()
            
            source_to_template.append(len(template_ids))
            
            # 处理当前 token
            if op == 0:  # KEEP
                template_ids.append(token_id)
            elif op == 1:  # DELETE
                # 删除：不添加任何 token
                pass
            elif op == 2:  # REPLACE
                mask_positions.append(len(template_ids))
                template_ids.append(self.mask_token_id)
            
            # 处理插入 (仅对非删除的 token)
            if op != 1 and insert_num > 0:
                # 应用风险约束
                actual_insert = min(insert_num, max_allowed_inserts - total_inserts)
                actual_insert = max(0, actual_insert)
                
                for _ in range(actual_insert):
                    mask_positions.append(len(template_ids))
                    template_ids.append(self.mask_token_id)
                    total_inserts += 1
        
        # [SEP]
        template_ids.append(self.sep_token_id)
        
        return template_ids, mask_positions, source_to_template
    
    def build_gold_template(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_labels: torch.Tensor,
        insert_labels: torch.Tensor,
    ) -> TemplateResult:
        """
        使用 Gold 标签构建模板 (用于训练 Infiller)
        
        与 build_template 相同，但使用真实标签而非预测
        """
        return self.build_template(
            input_ids, attention_mask, op_labels, insert_labels
        )
    
    def decode_template(self, template_ids: torch.Tensor) -> str:
        """将模板 IDs 解码为字符串"""
        return self.tokenizer.decode(template_ids, skip_special_tokens=False)
    
    # ========== 全 MASK 模式方法 ==========
    
    def compute_target_length(
        self,
        op_labels: torch.Tensor,
        insert_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据 Planner 的预测计算 target 长度 N
        
        计算规则:
        - KEEP (op=0): 贡献 1
        - DELETE (op=1): 贡献 0
        - REPLACE (op=2): 贡献 1
        - INSERT: 在该位置后贡献 insert_labels[i] 个字符
        
        Args:
            op_labels: [batch, seq_len]，操作标签
            insert_labels: [batch, seq_len]，插入标签
            attention_mask: [batch, seq_len]，注意力掩码
            
        Returns:
            target_lengths: [batch]，每个样本的 target 长度
        """
        batch_size = op_labels.shape[0]
        target_lengths = []
        
        for b in range(batch_size):
            seq_len = attention_mask[b].sum().item()
            length = 0
            
            for i in range(1, int(seq_len) - 1):  # 跳过 [CLS] 和 [SEP]
                op = op_labels[b, i].item()
                insert_num = insert_labels[b, i].item()
                
                # 跳过无效标签 (-100)
                if op == -100:
                    continue
                
                # 计算贡献
                if op == 0:  # KEEP
                    length += 1
                elif op == 1:  # DELETE
                    length += 0  # 删除不贡献
                elif op == 2:  # REPLACE
                    length += 1
                
                # 插入贡献（仅对非删除位置）
                if op != 1 and insert_num > 0:
                    # 应用风险约束
                    src_len = int(seq_len) - 2
                    max_allowed = min(
                        self.max_insert_per_sentence,
                        int(src_len * self.max_insert_ratio) + 1
                    )
                    actual_insert = min(insert_num, max_allowed)
                    length += actual_insert
            
            target_lengths.append(length)
        
        return torch.tensor(target_lengths, dtype=torch.long, device=op_labels.device)
    
    def build_full_mask_template(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> FullMaskTemplateResult:
        """
        构建全 MASK 模式的模板
        
        输出格式: [CLS] source [SEP] [MASK]*N [SEP] [PAD]...
        
        注意: P-Tuning 的 prompt 在模型层面动态添加，这里只构建基础模板
        
        Args:
            input_ids: [batch, seq_len]，源序列（已包含 [CLS] 和 [SEP]）
            attention_mask: [batch, seq_len]，源序列的注意力掩码
            target_lengths: [batch]，每个样本的 target 长度
            
        Returns:
            FullMaskTemplateResult
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 计算每个样本的实际长度
        source_lengths = []
        for b in range(batch_size):
            src_len = attention_mask[b].sum().item() - 2  # 去掉 [CLS] 和 [SEP]
            source_lengths.append(int(src_len))
        
        # 计算最大输出长度: [CLS] + src + [SEP] + N + [SEP]
        max_output_len = 0
        for b in range(batch_size):
            output_len = 1 + source_lengths[b] + 1 + target_lengths[b].item() + 1
            max_output_len = max(max_output_len, output_len)
        
        # 限制在 max_seq_length 内
        max_output_len = min(max_output_len, self.max_seq_length)
        
        # 构建输出
        output_ids = torch.full(
            (batch_size, max_output_len), 
            self.pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        output_mask = torch.zeros(
            (batch_size, max_output_len), 
            dtype=torch.long, 
            device=device
        )
        target_start_positions = []
        actual_target_lengths = []
        
        for b in range(batch_size):
            src_len = source_lengths[b]
            tgt_len = int(target_lengths[b].item())
            
            # 计算可用空间
            # [CLS] + src + [SEP] + target + [SEP] <= max_output_len
            available_for_target = max_output_len - 1 - src_len - 1 - 1
            actual_tgt_len = min(tgt_len, max(0, available_for_target))
            
            pos = 0
            
            # [CLS]
            output_ids[b, pos] = self.cls_token_id
            output_mask[b, pos] = 1
            pos += 1
            
            # source tokens (不包含原来的 [CLS] 和 [SEP])
            for i in range(src_len):
                if pos >= max_output_len - 2:  # 保留空间给 [SEP] 和至少一个 target
                    break
                output_ids[b, pos] = input_ids[b, i + 1].item()  # +1 跳过原始 [CLS]
                output_mask[b, pos] = 1
                pos += 1
            
            # [SEP] (source 结束)
            output_ids[b, pos] = self.sep_token_id
            output_mask[b, pos] = 1
            pos += 1
            
            # 记录 target 起始位置
            target_start_positions.append(pos)
            
            # [MASK] * N (target 区域)
            mask_count = 0
            for _ in range(actual_tgt_len):
                if pos >= max_output_len - 1:  # 保留空间给最后的 [SEP]
                    break
                output_ids[b, pos] = self.mask_token_id
                output_mask[b, pos] = 1
                pos += 1
                mask_count += 1
            
            actual_target_lengths.append(mask_count)
            
            # [SEP] (target 结束)
            if pos < max_output_len:
                output_ids[b, pos] = self.sep_token_id
                output_mask[b, pos] = 1
        
        return FullMaskTemplateResult(
            input_ids=output_ids,
            attention_mask=output_mask,
            target_start_pos=torch.tensor(target_start_positions, dtype=torch.long, device=device),
            target_length=torch.tensor(actual_target_lengths, dtype=torch.long, device=device),
            source_length=torch.tensor(source_lengths, dtype=torch.long, device=device),
        )
    
    def build_full_mask_template_from_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_predictions: torch.Tensor,
        insert_predictions: torch.Tensor,
    ) -> FullMaskTemplateResult:
        """
        从 Planner 预测构建全 MASK 模板（推理时使用）
        
        Args:
            input_ids: [batch, seq_len]，源序列
            attention_mask: [batch, seq_len]，源序列掩码
            op_predictions: [batch, seq_len]，操作预测
            insert_predictions: [batch, seq_len]，插入预测
            
        Returns:
            FullMaskTemplateResult
        """
        # 计算 target 长度
        target_lengths = self.compute_target_length(
            op_predictions, insert_predictions, attention_mask
        )
        
        # 构建模板
        return self.build_full_mask_template(
            input_ids, attention_mask, target_lengths
        )


class InferenceTemplateBuilder(TemplateBuilder):
    """
    推理时的模板构建器
    
    增加了迭代精炼相关的功能，支持稀疏 MASK 和全 MASK 两种模式
    """
    
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 512,
        max_insert_per_sentence: int = 6,
        max_insert_ratio: float = 0.1,
        full_mask_mode: bool = False,
    ):
        super().__init__(
            tokenizer, max_seq_length, max_insert_per_sentence, max_insert_ratio, full_mask_mode
        )
    
    def create_refinement_template(
        self,
        current_output: torch.Tensor,
        confidence_scores: torch.Tensor,
        mask_ratio: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建迭代精炼的模板
        
        将置信度最低的 token 重新 mask
        
        Args:
            current_output: 当前输出 token IDs [batch, seq_len]
            confidence_scores: 置信度分数 [batch, seq_len]
            mask_ratio: 重新 mask 的比例
            
        Returns:
            (new_template_ids, mask_positions)
        """
        batch_size, seq_len = current_output.shape
        device = current_output.device
        
        new_template_ids = current_output.clone()
        all_mask_positions = []
        
        for b in range(batch_size):
            # 找出非特殊 token 的位置
            valid_positions = []
            for i in range(seq_len):
                token_id = current_output[b, i].item()
                if token_id not in [self.pad_token_id, self.cls_token_id, self.sep_token_id]:
                    valid_positions.append(i)
            
            if not valid_positions:
                all_mask_positions.append([])
                continue
            
            # 计算需要 mask 的数量
            num_to_mask = max(1, int(len(valid_positions) * mask_ratio))
            
            # 按置信度排序，选择最低的
            valid_confidences = [(pos, confidence_scores[b, pos].item()) for pos in valid_positions]
            valid_confidences.sort(key=lambda x: x[1])
            
            positions_to_mask = [pos for pos, _ in valid_confidences[:num_to_mask]]
            
            # 应用 mask
            for pos in positions_to_mask:
                new_template_ids[b, pos] = self.mask_token_id
            
            all_mask_positions.append(positions_to_mask)
        
        return new_template_ids, all_mask_positions
