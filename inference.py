"""
Gap-ReLM 推理模块
支持单句/批量推理、迭代精炼、Verifier
"""

import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm

from models import GapReLMModel
from models.template_builder import InferenceTemplateBuilder
from config import GapReLMConfig


@dataclass
class CorrectionResult:
    """纠错结果"""
    source: str                    # 原句
    prediction: str                # 纠错后的句子
    confidence: float              # 置信度
    edits: List[Dict[str, Any]]    # 编辑操作列表
    is_changed: bool               # 是否发生改变
    verifier_accepted: Optional[bool] = None  # Verifier 是否接受


class GapReLMPredictor:
    """
    Gap-ReLM 推理预测器
    """
    
    def __init__(
        self,
        model: GapReLMModel,
        tokenizer,
        config: Optional[GapReLMConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: 训练好的 GapReLM 模型
            tokenizer: tokenizer
            config: 配置（可选，使用模型自带配置）
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or model.config
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 推理配置
        self.inference_config = self.config.inference
        self.f2_config = self.config.f2_optimization
        self.ablation_config = self.config.ablation
        
        # 模板构建器
        self.template_builder = InferenceTemplateBuilder(
            tokenizer=tokenizer,
            max_seq_length=self.config.model.max_seq_length,
            max_insert_per_sentence=self.f2_config.max_insert_per_sentence,
            max_insert_ratio=self.f2_config.max_insert_ratio,
        )
    
    def predict(
        self,
        text: str,
        use_iterative_refinement: Optional[bool] = None,
        use_verifier: Optional[bool] = None,
    ) -> CorrectionResult:
        """
        对单个句子进行纠错
        
        Args:
            text: 输入句子
            use_iterative_refinement: 是否使用迭代精炼
            use_verifier: 是否使用 Verifier
            
        Returns:
            CorrectionResult
        """
        # 默认使用配置中的设置
        if use_iterative_refinement is None:
            use_iterative_refinement = self.inference_config.use_iterative_refinement
        if use_verifier is None:
            use_verifier = self.inference_config.use_verifier
        
        # 编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.model.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # 获取预测
            result = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                use_iterative_refinement=use_iterative_refinement,
                refinement_rounds=self.ablation_config.refinement_rounds,
                refinement_mask_ratio=self.ablation_config.refinement_mask_ratio,
                use_verifier=use_verifier,
            )
        
        # 解析结果
        prediction = result['decoded_predictions'][0]
        confidence = result['confidence'][0].mean().item()
        
        # 提取编辑操作
        edits = self._extract_edits(
            text,
            prediction,
            result['op_predictions'][0],
            result['insert_predictions'][0]
        )
        
        # Verifier 结果
        verifier_accepted = None
        if use_verifier and result['verifier_accepted'] is not None:
            verifier_accepted = result['verifier_accepted'][0].item()
        
        return CorrectionResult(
            source=text,
            prediction=prediction,
            confidence=confidence,
            edits=edits,
            is_changed=(text != prediction),
            verifier_accepted=verifier_accepted
        )
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_iterative_refinement: Optional[bool] = None,
        use_verifier: Optional[bool] = None,
        show_progress: bool = True,
    ) -> List[CorrectionResult]:
        """
        批量纠错
        
        Args:
            texts: 输入句子列表
            batch_size: 批大小
            use_iterative_refinement: 是否使用迭代精炼
            use_verifier: 是否使用 Verifier
            show_progress: 是否显示进度条
            
        Returns:
            CorrectionResult 列表
        """
        results = []
        
        # 分批处理
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_results = self._predict_batch_internal(
                batch_texts,
                use_iterative_refinement,
                use_verifier
            )
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(
        self,
        texts: List[str],
        use_iterative_refinement: Optional[bool],
        use_verifier: Optional[bool],
    ) -> List[CorrectionResult]:
        """内部批量预测"""
        if use_iterative_refinement is None:
            use_iterative_refinement = self.inference_config.use_iterative_refinement
        if use_verifier is None:
            use_verifier = self.inference_config.use_verifier
        
        # 编码
        encoding = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.config.model.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            result = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                use_iterative_refinement=use_iterative_refinement,
                refinement_rounds=self.ablation_config.refinement_rounds,
                refinement_mask_ratio=self.ablation_config.refinement_mask_ratio,
                use_verifier=use_verifier,
            )
        
        # 构建结果
        results = []
        for i, text in enumerate(texts):
            prediction = result['decoded_predictions'][i]
            confidence = result['confidence'][i].mean().item()
            
            edits = self._extract_edits(
                text,
                prediction,
                result['op_predictions'][i],
                result['insert_predictions'][i]
            )
            
            verifier_accepted = None
            if use_verifier and result['verifier_accepted'] is not None:
                verifier_accepted = result['verifier_accepted'][i].item()
            
            results.append(CorrectionResult(
                source=text,
                prediction=prediction,
                confidence=confidence,
                edits=edits,
                is_changed=(text != prediction),
                verifier_accepted=verifier_accepted
            ))
        
        return results
    
    def _extract_edits(
        self,
        source: str,
        prediction: str,
        op_preds: torch.Tensor,
        insert_preds: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        提取编辑操作
        
        Returns:
            编辑操作列表，每个元素包含:
            - position: 位置
            - type: 操作类型 (KEEP/DELETE/REPLACE/INSERT)
            - source_char: 原字符
            - target_char: 目标字符
        """
        edits = []
        op_names = ['KEEP', 'DELETE', 'REPLACE']
        
        src_chars = list(source)
        
        for i, char in enumerate(src_chars):
            if i + 1 < len(op_preds):  # +1 因为有 [CLS]
                op = op_preds[i + 1].item()
                insert_num = insert_preds[i + 1].item()
                
                if op != 0:  # 不是 KEEP
                    edits.append({
                        'position': i,
                        'type': op_names[op],
                        'source_char': char,
                        'target_char': None  # 需要从 prediction 中获取
                    })
                
                if insert_num > 0:
                    edits.append({
                        'position': i,
                        'type': 'INSERT',
                        'insert_count': insert_num,
                        'source_char': None,
                        'target_char': None
                    })
        
        return edits
    
    def iterative_refine(
        self,
        text: str,
        num_rounds: int = 2,
        mask_ratio: float = 0.15,
    ) -> CorrectionResult:
        """
        显式调用迭代精炼
        
        Args:
            text: 输入文本
            num_rounds: 精炼轮数
            mask_ratio: 每轮 mask 比例
            
        Returns:
            CorrectionResult
        """
        # 首次预测
        result = self.predict(text, use_iterative_refinement=False)
        current_prediction = result.prediction
        
        # 迭代精炼
        for r in range(num_rounds):
            # 编码当前预测
            encoding = self.tokenizer(
                current_prediction,
                add_special_tokens=True,
                max_length=self.config.model.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                # 获取置信度
                outputs = self.model.infiller(input_ids, attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidence = probs.max(dim=-1)[0]
                
                # 找出低置信度位置
                new_template, mask_positions = self.template_builder.create_refinement_template(
                    input_ids,
                    confidence,
                    mask_ratio
                )
                
                # 重新预测
                predictions, _ = self.model.infiller.predict(
                    new_template,
                    attention_mask,
                    mask_token_id=self.tokenizer.mask_token_id
                )
                
                current_prediction = self.tokenizer.decode(
                    predictions[0],
                    skip_special_tokens=True
                )
        
        return CorrectionResult(
            source=text,
            prediction=current_prediction,
            confidence=result.confidence,  # 使用首次预测的置信度
            edits=result.edits,
            is_changed=(text != current_prediction)
        )


class GapReLMPipeline:
    """
    便捷的推理 Pipeline
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: 模型路径
            device: 设备 ("cuda", "cpu", None=auto)
        """
        from transformers import AutoTokenizer
        from config import GapReLMConfig
        
        # 加载配置
        self.config = GapReLMConfig.load(f"{model_path}/config.json")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型
        self.model = GapReLMModel.from_pretrained(model_path, self.config)
        
        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 创建预测器
        self.predictor = GapReLMPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device
        )
    
    def __call__(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[CorrectionResult, List[CorrectionResult]]:
        """
        调用纠错
        
        Args:
            texts: 单个句子或句子列表
            **kwargs: 传递给 predict/predict_batch 的参数
            
        Returns:
            单个或多个 CorrectionResult
        """
        if isinstance(texts, str):
            return self.predictor.predict(texts, **kwargs)
        else:
            return self.predictor.predict_batch(texts, **kwargs)
    
    def correct(self, text: str) -> str:
        """
        简单接口：返回纠错后的文本
        """
        result = self.predictor.predict(text)
        return result.prediction
    
    def correct_batch(self, texts: List[str]) -> List[str]:
        """
        简单接口：批量返回纠错后的文本
        """
        results = self.predictor.predict_batch(texts)
        return [r.prediction for r in results]


def load_predictor(model_path: str, device: Optional[str] = None) -> GapReLMPredictor:
    """
    加载预测器的便捷函数
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        GapReLMPredictor
    """
    pipeline = GapReLMPipeline(model_path, device)
    return pipeline.predictor
