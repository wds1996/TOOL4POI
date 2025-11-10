import transformers
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tool.tools import filter_poi_by_categories, filter_poi_by_regions, get_poi_infos, get_Interactive_POIs, finish, sort_by_pid2candidate
import re
from tqdm import tqdm


def acc_at_k(preds, target, k):
    if k > len(preds):
        k = len(preds)
    return 1.0 if target in preds[:k] else 0.0

def jug_mrr(preds, target):
    for i, p in enumerate(preds):
        if p == target:
            return 1.0 / (i + 1)
    return 0.0


def parse_tool_call(llm_reply: str):
    pattern = r"<tool_call>(.*?)</tool_call>"
    match = re.search(pattern, llm_reply, re.DOTALL)
    
    if not match:
        return None, None
    
    try:
        tool_call = json.loads(match.group(1))
        func_name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        return func_name, args
    except json.JSONDecodeError:
        return None, None


def call_qwen(messages, model, tokenizer, tools=None):

    inputs = tokenizer.apply_chat_template(messages, tools= tools, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=4096, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_logits=True, do_sample=False, temperature=None, top_p=None, top_k=None)
    response_ids = outputs['sequences'][0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response

def recommend(query_str, candidate_info=None):
    
    prompt = f"""You are a recommendation assistant. Based on the user's visit history and the current time, predict the 10 POIs the user is most likely to visit.
    Just need to output the result without explanation.
    Wrap the final result in <result></result> tags, e.g., <result>[pid1, pid2, ...]</result>.
    """
    input_str = query_str
    
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': input_str}
    ]
    output = call_qwen(messages, model, tokenizer)
    
    return output


if __name__ == "__main__":

    dataflod = "NYC"
    # model_id = "/models/Qwen2.5-7B-Instruct"
    model_id = "/models/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map={"": 3}, attn_implementation="flash_attention_2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with open('tool/tool_getinfos.json', 'r', encoding='utf-8') as f:
        getinfo = json.load(f)

    with open('tool/tools_retrieve.json', 'r', encoding='utf-8') as f:
        retrieve = json.load(f)

    with open(f'data/{dataflod}/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    Acc1 = 0.0
    Acc3 = 0.0
    Acc5 = 0.0
    Mrr = 0.0
    

    for query in tqdm(data, desc="Processing queries"):
        query_str = query["input"]
        target = query["target"]

        candidates = None
        if candidates:
            candidate_info = get_poi_infos(dataflod, candidates)
        else:
            candidate_info = None
        next_time = query["next_time"]
        query_str = f"{query_str}\nThe current time is: {next_time}\n"
        respond = recommend(query_str, candidate_info)

        predicted_pids = candidates if candidates else []
        result = re.search(r"<result>(.*?)</result>", respond)
        if result:
            try:
                predicted_pids = eval(result.group(1))

            except Exception as e:
                print("error:", e)
        else:
            print(respond)

        Acc1 += 1 if target in predicted_pids[:1] else 0
        Acc3 += acc_at_k(predicted_pids, target, 3)
        Acc5 += acc_at_k(predicted_pids, target, 5)
        Mrr += jug_mrr(predicted_pids, target)

    total_queries = len(data)
    print(f"Total queries processed: {total_queries}")
    print(f"Total queries: {total_queries}")
    print(f"Accuracy@1: {Acc1 / total_queries:.4f}")
    print(f"Accuracy@3: {Acc3 / total_queries:.4f}")
    print(f"Accuracy@5: {Acc5 / total_queries:.4f}")
    print(f"Mean Reciprocal Rank (Mrr): {Mrr / total_queries:.4f}")
