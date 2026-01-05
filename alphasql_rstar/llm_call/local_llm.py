"""
本地模型推理模块，直接使用transformers加载模型进行推理
"""
import os
import torch
import threading
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from alphasql_equation.llm_call.cost_recoder import CostRecorder

# 全局变量存储模型和tokenizer，避免重复加载
_global_model = None
_global_tokenizer = None
_global_model_path = None
_model_lock = threading.Lock()  # 添加线程锁，确保模型只加载一次

def load_local_model(model_path: str, device: str = "cuda:0"):
    """
    加载本地模型（线程安全）
    
    Args:
        model_path: 模型路径
        device: 设备，默认 "cuda:0"
    """
    global _global_model, _global_tokenizer, _global_model_path, _model_lock
    
    # 使用线程锁确保只加载一次
    with _model_lock:
        # 双重检查，避免重复加载
        if _global_model_path == model_path and _global_model is not None:
            return _global_model, _global_tokenizer
        
        print(f"Loading local model from {model_path} on {device}...")
        
        # 设置设备
        if device.startswith("cuda") and torch.cuda.is_available():
            # 提取GPU编号
            gpu_id = int(device.split(":")[1]) if ":" in device else 0
            device_map = f"cuda:{gpu_id}"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型 - 移除 low_cpu_mem_usage 以避免 meta tensor 问题
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True
            )
        except Exception as e:
            # 如果 device_map 失败，尝试手动移动到设备
            print(f"Warning: device_map failed, trying manual device placement: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            model = model.to(device_map)
        
        model.eval()
        
        _global_model = model
        _global_tokenizer = tokenizer
        _global_model_path = model_path
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer

def call_local_llm(
    prompt: str,
    model_path: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    stop: List[str] = None,
    device: str = "cuda:0",
    cost_recorder: Optional[CostRecorder] = None
) -> List[str]:
    """
    使用本地模型进行推理
    
    Args:
        prompt: 输入提示
        model_path: 模型路径
        temperature: 温度参数
        top_p: top_p采样参数
        max_tokens: 最大生成token数
        stop: 停止词列表
        device: 设备
        cost_recorder: 成本记录器（本地模型通常不需要）
    
    Returns:
        生成的文本列表
    """
    model, tokenizer = load_local_model(model_path, device)
    
    # 构建输入
    messages = [{"role": "user", "content": prompt}]
    
    # 使用tokenizer的apply_chat_template（如果支持）
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # 简单的格式
        text = f"User: {prompt}\nAssistant: "
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    # 如果使用device_map，模型已经在正确的设备上，需要将输入移到模型所在的设备
    if hasattr(model, 'device'):
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    elif device.startswith("cuda") and torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0 or top_p < 1.0,
    }
    
    # 添加采样参数
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    if top_p < 1.0:
        generation_kwargs["top_p"] = top_p
    
    # 设置pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 处理停止词
    if stop:
        for stop_word in stop:
            if stop_word in generated_text:
                generated_text = generated_text.split(stop_word)[0]
    
    # 更新cost recorder（估算token数）
    if cost_recorder is not None:
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        cost_recorder.update_cost(input_tokens, output_tokens)
    
    return [generated_text]

