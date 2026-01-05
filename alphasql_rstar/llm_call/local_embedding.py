"""
本地嵌入模型模块，使用transformers加载本地嵌入模型
"""
import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from loguru import logger

# 全局变量存储模型和tokenizer，避免重复加载
_global_embedding_model = None
_global_embedding_tokenizer = None
_global_embedding_model_path = None

def load_local_embedding_model(model_path: str, device: str = "cuda:0"):
    """
    加载本地嵌入模型
    
    Args:
        model_path: 模型路径
        device: 设备，默认 "cuda:0"
    """
    global _global_embedding_model, _global_embedding_tokenizer, _global_embedding_model_path
    
    if _global_embedding_model_path == model_path and _global_embedding_model is not None:
        return _global_embedding_model, _global_embedding_tokenizer
    
    logger.info(f"Loading local embedding model from {model_path} on {device}...")
    
    # 设置设备
    if device.startswith("cuda") and torch.cuda.is_available():
        device_obj = torch.device(device)
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device_obj = torch.device("cpu")
        torch_dtype = torch.float32
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 直接加载到指定设备，不使用device_map
    # 使用低内存模式加载
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,  # 不使用device_map
        low_cpu_mem_usage=True  # 低内存模式
    )
    
    # 移动到指定设备
    model = model.to(device_obj)
    model.eval()
    
    # 清理缓存
    if device_obj.type == 'cuda':
        torch.cuda.empty_cache()
    
    _global_embedding_model = model
    _global_embedding_tokenizer = tokenizer
    _global_embedding_model_path = model_path
    
    logger.info(f"Embedding model loaded successfully on {device}")
    return model, tokenizer

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts_local(
    texts: List[str],
    model_path: str,
    device: str = "cuda:0",
    normalize: bool = True
) -> List[List[float]]:
    """
    使用本地模型对文本列表进行嵌入
    
    Args:
        texts: 文本列表
        model_path: 模型路径
        device: 设备
        normalize: 是否归一化向量
    
    Returns:
        嵌入向量列表
    """
    if not texts:
        return []
    
    model, tokenizer = load_local_embedding_model(model_path, device)
    
    # 设置设备
    if device.startswith("cuda") and torch.cuda.is_available():
        device_obj = torch.device(device)
    else:
        device_obj = torch.device("cpu")
    
    embeddings = []
    
    with torch.no_grad():
        # 批量处理 - 可以根据GPU内存调整
        batch_size = 16  # 如果使用独立GPU，可以增大batch size提高效率
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移动到设备
            encoded_input = {k: v.to(device_obj) for k, v in encoded_input.items()}
            
            # 获取模型输出
            model_output = model(**encoded_input)
            
            # 使用mean pooling
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 归一化
            if normalize:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            # 转换为numpy并添加到列表
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            
            # 清理GPU缓存，避免内存累积
            if device_obj.type == 'cuda':
                del batch_embeddings
                torch.cuda.empty_cache()
    
    return embeddings

