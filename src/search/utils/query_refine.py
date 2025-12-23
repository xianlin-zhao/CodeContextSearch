import time
import json
import random
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = None

def get_client():
    """获取OpenAI客户端，如果未初始化则初始化"""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 环境变量未设置。"
                "请创建 .env 文件并设置 OPENAI_API_KEY 和 OPENAI_BASE_URL，"
                "或者设置环境变量。"
            )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    return client

refine_query_prompt = """
You are an experienced software developer specializing in code retrieval and understanding.

<raw_query>
{RAW_QUERY}
</raw_query>

Task:
The provided <raw_query> combines a functional requirement with its arguments.
Your task is to **rewrite** this query into a natural language description that is **ACCURATE, DETAILED, and INFORMATIVE**.
The goal is to produce a refined query that captures the complete coding intent by seamlessly integrating the functional requirement with **ALL** its arguments and specific values.

Instructions:
1. Analysis:
- Analyze the <raw_query> to understand the context.
- Identify all technical keywords, library names, algorithm names, configuration flags, and specific data values.
- **JUDGMENT (CRITICAL):** **Preserve EVERYTHING.** Do NOT filter out specific data elements, file paths, numbers, or string literals. In a code retrieval context, specific values (e.g., `'/tmp/logs'`, `epoch=50`) often correspond to default parameter values or hardcoded constants in the target function. These are high-value search terms and must be retained.

2. Rewriting (Semantic Synthesis):
- Rewrite the query into a clear, grammatically correct sentence or paragraph.
- **Integrate Specifics Naturally:** Weave the "Functionality" and the specific "Arguments" together.
- Instead of saying "a specified directory", say "the '/tmp/logs' directory".
- Instead of saying "a specific threshold", say "a threshold of 0.5".
- Ensure the final query clearly describes what the code does using the **exact constraints and values** provided.

Please answer in the following format:

[start_of_analysis]
<analysis_of_intent_and_confirmation_of_specific_values>
[end_of_analysis]

[start_of_rewritten_query]
<refined_query_with_specific_values_preserved>
[end_of_rewritten_query]

Notes:
- The output must be **SPECIFIC**: Retain literal values from the arguments (paths, IDs, constants).
- The output must be **ACCURATE**: Do not hallucinate features not implied by the raw query.
- The output should be **NATURAL**: Do not just list the arguments; make them part of the narrative flow of the sentence.

Example 1:
Raw: "train model arguments: epochs=50, optimizer='adam', learning_rate=0.001, early_stopping=True, log_dir='/tmp/logs'"
Rewritten: "Train a model for 50 epochs using the Adam optimizer with a learning rate of 0.001, incorporating early stopping and saving logs specifically to the '/tmp/logs' directory."

Example 2:
Raw: "parse log arguments: file_path='/var/log/syslog', pattern='ERROR', output_format='json', max_retries=3"
Rewritten: "Parse the log file located at '/var/log/syslog' to extract entries matching the 'ERROR' pattern, performing up to 3 retries, and serialize the output to JSON format."
"""

def call_with_retry(fn, retries=5, base_delay=0.5, max_delay=8.0):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "RateLimit" not in msg and "upstream" not in msg:
                raise
            delay = min(max_delay, base_delay * (2 ** i)) * (1 + random.random() * 0.25)
            time.sleep(delay)
    return fn()

def refine_query(raw_query: str, modelname: str) -> str:
    prompt = refine_query_prompt.format(RAW_QUERY=raw_query)
    try:
        response = call_with_retry(lambda: get_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelname,
            temperature=0.3,
            top_p=0.95,
        ))
        content = response.choices[0].message.content or ""
        
        start_tag = "[start_of_rewritten_query]"
        end_tag = "[end_of_rewritten_query]"
        
        start_index = content.find(start_tag)
        end_index = content.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            refined_query = content[start_index + len(start_tag):end_index].strip()
            return refined_query
        else:
            return raw_query

    except Exception as e:
        print(f"Error refining query: {e}")
        return raw_query
