import json
import re
from openai import OpenAI
from tqdm import tqdm

API_KEY = "---"
BASE_URL = "https://api.siliconflow.cn/v1"
INPUT_FILE = "math500_qwen_ans_raw.jsonl"
OUTPUT_FILE = "math500_nodes.jsonl"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm_segment(text):
    prompt = f"""
    You are a mathematical reasoning expert. Please rewrite the following problem-solving process into a standard sequence of atomic steps.
    
    Requirements:
    1. Decompose the reasoning into logically independent, atomic steps.
    2. Keep the logical flow natural. Use transition phrases (like 'So', 'Therefore', 'Next') to start a new step, but never end a step with a colon or an introductory phrase. A step must always contain the mathematical content that follows the transition.
    3. Each step MUST start with the tag "[STEP]".
    5. Do NOT use JSON, Markdown, or any other wrapper. Just output the text with the [STEP] tags.
    5. Do NOT add, delete or edit any raw text, the only thing you can do is adding "[STEP]"
    
    6. Example of a GOOD step: "[STEP] So, 49 can be written as:  49 = 7 × 7."
    7. Example of a BAD step: "[STEP] So, 49 can be written as: [STEP] 49 = 7 × 7."
    
    Text to rewrite:
    {text}
    """
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip() # type: ignore
        steps = re.split(r'\[STEP\]', content)
        steps = [s.strip() for s in steps if s.strip()]
        return steps
    
    except Exception as e:
        print(f"Error: {e}")
        return [text]

def main():
    print(f"reading: {INPUT_FILE} ...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        
        for line in tqdm(lines, desc="segmenting..."):
            if not line.strip(): continue
            
            raw_data = json.loads(line)
            problem_id = raw_data.get('id')
            model_output = raw_data.get('model_output', '')
            
            steps = call_llm_segment(model_output)
            
            for idx, step_content in enumerate(steps):
                node = {
                    "problem_id": problem_id,
                    "step_index": idx,
                    "content": step_content
                }
                f_out.write(json.dumps(node, ensure_ascii=False) + '\n')
                
    print(f"successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()