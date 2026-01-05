from openai import OpenAI
import dotenv
import os
from typing import List, Optional
from alphasql_equation.llm_call.cost_recoder import CostRecorder
import time

dotenv.load_dotenv(override=True)

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL")

DEFAULT_COST_RECORDER = CostRecorder(model="Qwen/Qwen2.5-7B-Instruct")

MAX_RETRYING_TIMES = 5

# MAX_TIMEOUT = 60

N_CALLING_STRATEGY_SINGLE = "single"
N_CALLING_STRATEGY_MULTIPLE = "multiple"

def call_openai(prompt: str,
                model: str,
                temperature: float = 0.0,
                top_p: float = 1.0,
                n: int = 1,
                max_tokens: int = 512,
                stop: List[str] = None,
                base_url: str = None,
                api_key: str = None,
                n_strategy: str = N_CALLING_STRATEGY_SINGLE,
                cost_recorder: Optional[CostRecorder] = DEFAULT_COST_RECORDER) -> str:
    # 如果未提供 base_url 或 api_key，从环境变量读取
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY
    
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)
    retrying = 0
    while retrying < MAX_RETRYING_TIMES:
        try:
            if n == 1 or (n > 1 and n_strategy == N_CALLING_STRATEGY_SINGLE):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    top_p=top_p,
                    stop=stop,
                    # timeout=MAX_TIMEOUT,
                )
                if cost_recorder is not None:
                    cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                contents = [choice.message.content for choice in response.choices]
                break
            elif n > 1 and n_strategy == N_CALLING_STRATEGY_MULTIPLE:
                contents = []
                for _ in range(n):
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1,
                        top_p=top_p,
                        stop=stop,
                        # timeout=MAX_TIMEOUT,
                    )
                    if cost_recorder is not None:
                        cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    contents.append(response.choices[0].message.content)
                break
            else:
                raise ValueError(f"Invalid n_strategy: {n_strategy} for n: {n}")
        except Exception as e:
            print("-" * 100)
            print(f"Error calling OpenAI: {e}")
            print(f"Start retrying {retrying + 1} times")
            print("-" * 100)
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                raise e
            # sleep for 10 seconds
            time.sleep(10)
    # print(contents)
    return contents

