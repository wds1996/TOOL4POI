from openai import OpenAI
import transformers
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tool.tools import filter_poi_by_categories, filter_poi_by_regions, get_poi_infos, init_candidates, finish, sort_by_pid2candidate, ranking_infos
import re
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Any
import queue


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

def get_function_by_name(func_name):
    if func_name == "filter_poi_by_categories":
        return filter_poi_by_categories
    if func_name == "filter_poi_by_regions":
        return filter_poi_by_regions
    if func_name == "sort_by_pid2candidate":
        return sort_by_pid2candidate
    if func_name == "init_candidates":
        return init_candidates
    if func_name == "finish":
        return finish
    if func_name == "get_poi_infos":
        return get_poi_infos


class ClientManager:
    def __init__(self, api_key="NaN", base_url="http://localhost:8000/v1", max_clients=10):
        self.api_key = api_key
        self.base_url = base_url
        self.clients = queue.Queue()
        for _ in range(max_clients):
            client = OpenAI(api_key=api_key, base_url=base_url)
            self.clients.put(client)
    
    def get_client(self):
        return self.clients.get()
    
    def return_client(self, client):
        self.clients.put(client)

def call_qwen_threadsafe(client_manager, model_id, messages, tools=None):
    client = client_manager.get_client()
    try:
        chat_response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools,
            max_tokens=8192,
            temperature=0,
            # temperature=0.6,
            # top_p=0.8,
            # seed=2025,
        )
        return chat_response
    finally:
        client_manager.return_client(client)


def analyze_user(dataflod, query, getinfo, client_manager, model_id):
    prompt = f"""You are a user preference assistant in a recommendation system.  
Based on the user's historical visit sequence, analyze their visiting preferences and patterns.
Please analyze the user's preferences from the following two dimensions: *region* and *category*. Provide the corresponding preference keywords.
The output format should be:  
"regions": [region1, region2, ...], (Include all regions the user is likely to visit.)  
"categories": [category1, category2, ...], (Include all POI categories the user is likely to visit.)  

You should only output the final result in the specified format. Do not include any explanation.
"""    
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': query}
    ]
    response = call_qwen_threadsafe(client_manager, model_id, messages, getinfo)
    
    messages.append(response.choices[0].message.model_dump())

    if tool_calls := messages[-1].get("tool_calls", None):
        for tool_call in tool_calls:
            call_id: str = tool_call["id"]
            if fn_call := tool_call.get("function"):
                fn_name: str = fn_call["name"]
                fn_args: dict = json.loads(fn_call["arguments"])
                fn_res: str = json.dumps(get_function_by_name(fn_name)(dataflod, **fn_args))
                messages.append({
                    "role": "tool",
                    "content": fn_res,
                    "tool_call_id": call_id,
                })
    user_preference = call_qwen_threadsafe(client_manager, model_id, messages).choices[0].message.content
    return user_preference

def retrieve_candidates(dataflod, poi_list, user_preference, retrieve, client_manager, model_id):
    prompt = f"""You are a retrieval assistant in a recommendation system. 
Based on the user's historical visit records and preference information, you may call one or more tools to retrieve a candidate set of POIs that the user is likely to visit.
Only return the tool call and the final result — no explanation is needed.
"""
    input_str = f"""user's preferences: {user_preference}
You should initialize the candidate set based on the user's last visited POI {poi_list[-1]}.
"""
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': input_str}
    ]
    chat_response = call_qwen_threadsafe(client_manager, model_id, messages, retrieve)
    
    candidates = []
    while True:
        messages.append(chat_response.choices[0].message.model_dump())
        if tool_calls := messages[-1].get("tool_calls", None):
            for tool_call in tool_calls:
                call_id: str = tool_call["id"]
                if fn_call := tool_call.get("function"):
                    fn_name: str = fn_call["name"]
                    fn_args: dict = json.loads(fn_call["arguments"])
                    while isinstance(fn_args, str):
                        fn_args = json.loads(fn_args)
                    if len(candidates) !=0 and "candidate" in fn_args:
                        fn_args["candidate"] = candidates
                    
                    if fn_name != "finish":
                        candidates, lens = get_function_by_name(fn_name)(dataflod, **fn_args)
                        fn_res = f"{candidates}"
                        if lens <= 10:
                            return candidates
                    else:
                        fn_args["result"] = candidates
                        candidates = get_function_by_name(fn_name)(**fn_args)
                        if len(candidates) > 20:
                            candidates = candidates[:20]    
                        return candidates
                    messages.append({
                        "role": "tool",
                        "content": fn_res,
                        "tool_call_id": call_id,
                    })
                else:
                    print("No valid function call found in the response.")
                    return candidates       

            chat_response = call_qwen_threadsafe(client_manager, model_id, messages, retrieve) 
        else:
            print("No tool calls found in the response.")
            return candidates

def rerank_candidates(input_str, next_time, poi_list, client_manager, model_id, dataflod, candidates, getinfo):

    prompt = f"""You are a ranking assistant in a recommendation system.  
You need to analyze the user's recent check-in records to infer their check-in preferences — for example, which categories or regions the user frequently visits at the given time.  
Then, based on the current time, the historical sequence, and the candidate set, rerank all candidate POIs according to the likelihood that the user will visit.  
       
Wrap the final result in <re_rank></re_rank> tags, e.g., <re_rank>[pid1, pid2, ...]</re_rank>.
"""
    # candidates_info = get_poi_infos(dataflod, candidates)
    candidates_info = ranking_infos(dataflod, poi_list[-1], candidates)
    input_str = f"""{input_str} 
The user is likely to visit the candidates POIs: {candidates} with details information: {candidates_info}.
Please rerank the candidates according to the likelihood that the user will visit them at {next_time}.
**All candidate POIs must be included in the final ranking.**
"""
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': input_str}
    ]
    
    response = call_qwen_threadsafe(client_manager, model_id, messages)
    # response = call_qwen_threadsafe(client_manager, model_id, messages, getinfo)
    messages.append(response.choices[0].message.model_dump())

    if tool_calls := messages[-1].get("tool_calls", None):
        for tool_call in tool_calls:
            call_id: str = tool_call["id"]
            if fn_call := tool_call.get("function"):
                fn_name: str = fn_call["name"]
                fn_args: dict = json.loads(fn_call["arguments"])
                fn_res: str = json.dumps(get_function_by_name(fn_name)(dataflod, **fn_args))
                messages.append({
                    "role": "tool",
                    "content": fn_res,
                    "tool_call_id": call_id,
                })
        result = call_qwen_threadsafe(client_manager, model_id, messages).choices[0].message.content
    else:
        result = response.choices[0].message.content

    return result


def process_single_query(history, recent, dataflod, getinfo, retrieve, client_manager, model_id):
    """处理单个查询的函数"""
    history_str = history["input"]
    recent_str = recent["input"]
    poi_match = re.search(r"records:\s*(\[[^\]]+\])", history_str)
    if poi_match:
        poi_list_str = poi_match.group(1)
        poi_list = eval(poi_list_str)
    next_time = recent["next_time"]
    target = recent["target"]
    error = []
    try:
        # 模块1：用户分析
        user_preference = analyze_user(dataflod, history_str, getinfo, client_manager, model_id)
    except Exception as e:
        print(f"用户分析失败: {e}")
        error.append(str(e))

    if user_preference:
        try:
            # 模块2：候选检索
            candidates = retrieve_candidates(dataflod, poi_list, user_preference, retrieve, client_manager, model_id)
            # 计算指标
            wo_reranke_acc1 = 1 if target in candidates[:1] else 0
            wo_reranke_acc5 = acc_at_k(candidates, target, 5)
            wo_reranke_acc10 = acc_at_k(candidates, target, 10)
            wo_reranke_mrr = jug_mrr(candidates, target)
            wo_rerank = {
                'wo_reranke_acc1': wo_reranke_acc1,
                'wo_reranke_acc5': wo_reranke_acc5,
                'wo_reranke_acc10': wo_reranke_acc10,
                'wo_reranke_mrr': wo_reranke_mrr
            }
        except Exception as e:
            print(f"候选检索失败: {e}")
            error.append(str(e))
            candidates = []
            wo_rerank = {
                'wo_reranke_acc1': 0,
                'wo_reranke_acc5': 0,
                'wo_reranke_acc10': 0,
                'wo_reranke_mrr': 0
            }
    else:
        print("用户偏好分析失败，无法进行候选检索")
        candidates = []
        wo_rerank = {
            'wo_reranke_acc1': 0,
            'wo_reranke_acc5': 0,
            'wo_reranke_acc10': 0,
            'wo_reranke_mrr': 0
        }

    try:
        # 模块3：排序
        # recent_set = set(poi_list)
        # candidates = set(candidates).union(recent_set)  # 合并候选和最近访问的POI
        # candidates = list(candidates)  
        # input_str = f"{recent_str}\n Candidates: {candidates}."
        
        # -------------------------------------------------------
        if len(candidates) == 0:
            recent_set = set(poi_list)
            candidates = list(recent_set) + candidates
        # input_str = f"{recent_str}\n Candidates: {candidates}."
        # -------------------------------------------------------
    
        respond = rerank_candidates(recent_str, next_time, poi_list, client_manager, model_id, dataflod, candidates, getinfo)

        # 解析最终推荐结果
        predicted_pids = candidates if candidates else []
        result = re.search(r"<re_rank>(.*?)</re_rank>", respond)
        if result:
            try:
                predicted_pids = eval(result.group(1))
                predicted_pids = [int(pid) for pid in predicted_pids]
            except Exception as e:
                print(f"解析失败: {e}")
        
        # 计算指标
        acc1 = 1 if target in predicted_pids[:1] else 0
        acc5 = acc_at_k(predicted_pids, target, 5)
        acc10 = acc_at_k(predicted_pids, target, 10)
        mrr = jug_mrr(predicted_pids, target)
        
        return {
            'target': target,
            'predicted': predicted_pids,
            'candidates': candidates,
            'wo_rerank': wo_rerank,
            'acc1': acc1,
            'acc5': acc5,
            'acc10': acc10,
            'mrr': mrr,
            'target_in_candidates': target in candidates if candidates else False,
            'target_in_final': target in predicted_pids if predicted_pids else False
        }
    
    except Exception as e:
        print(f"排序失败: {e}")
        error.append(str(e))
        return {
            'target': target,
            'predicted': [],
            'candidates': [],
            'wo_rerank': wo_rerank,
            'acc1': 0,
            'acc5': 0,
            'acc10': 0,
            'mrr': 0,
            'target_in_candidates': False,
            'target_in_final': False,
            'error': str(e)
        }

def main():

    dataflod = "NYC" # TKY, NYC, CA
    model_id = "/models/Qwen2.5-14B-Instruct"
    model = "QW25-14B"
    # model_id = "/models/Qwen3-14B"
    

    max_threads = 16  # 可以根据服务器性能调整
    client_manager = ClientManager(
        api_key="NaN",
        base_url="http://localhost:8000/v1",
        max_clients=max_threads
    )

    # 加载工具文档
    with open('tool/tool_getinfos.json', 'r', encoding='utf-8') as f:
        getinfo = json.load(f)

    with open('tool/tools_retrieve.json', 'r', encoding='utf-8') as f:
        retrieve = json.load(f)
    
    # 加载历史数据
    with open(f'data/{dataflod}/history100.json', 'r', encoding='utf-8') as f:
        history = json.load(f)
    # 加载最近查询数据
    with open(f'data/{dataflod}/recent20.json', 'r', encoding='utf-8') as f:
        recent = json.load(f)
    
    bad_data = []
    good_data = []
    # 选择要处理的数据范围
    for i, qury in enumerate(history):
        poi_match = re.search(r"records:\s*(\[[^\]]+\])", qury["input"])
        if poi_match:
            poi_list_str = poi_match.group(1)
            poi_list = eval(poi_list_str)
        target = qury["target"]
        if target not in poi_list:
            bad_data.append(recent[i])
        else:
            good_data.append(recent[i])
    print(f"数据集 {dataflod} 中共有 {len(recent)} 条查询，其中有效查询: {len(good_data)}, 无效查询: {len(bad_data)}")

    queries_to_process = recent[:]
    # queries_to_process = good_data
    
    # 初始化累计指标
    total_queries = len(queries_to_process)
    wo_rerank_Acc1 = 0.0  
    wo_rerank_Acc5 = 0.0
    wo_rerank_Acc10 = 0.0
    wo_rerank_Mrr = 0.0
    in_rerank = 0.0
    
    Acc1 = 0.0
    Acc5 = 0.0
    Acc10 = 0.0
    Mrr = 0.0
    in_final = 0.0
    
    # bad_data 单独指标初始化
    bad_total = len(bad_data)

    bad_wo_rerank_Acc1 = 0.0
    bad_wo_rerank_Acc5 = 0.0
    bad_wo_rerank_Acc10 = 0.0
    bad_wo_rerank_Mrr = 0.0

    bad_Acc1 = 0.0
    bad_Acc5 = 0.0
    bad_Acc10 = 0.0
    bad_Mrr = 0.0
    bad_in_rerank = 0.0
    bad_in_final = 0.0

    # good_data 单独指标初始化
    good_total = len(good_data)

    good_wo_rerank_Acc1 = 0.0
    good_wo_rerank_Acc5 = 0.0
    good_wo_rerank_Acc10 = 0.0
    good_wo_rerank_Mrr = 0.0

    good_Acc1 = 0.0
    good_Acc5 = 0.0
    good_Acc10 = 0.0
    good_Mrr = 0.0
    good_in_rerank = 0.0
    good_in_final = 0.0

    # 存储所有结果
    dic_candidates = {}
    dic_predicted = {}
    results = []
    
    print(f"开始处理 {total_queries} 个查询，使用 {max_threads} 个线程...")
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 提交所有任务
        future_to_query = {
            executor.submit(
                process_single_query,
                history[i],
                query,
                dataflod,
                getinfo,
                retrieve,
                client_manager,
                model_id
            ): i for i, query in enumerate(queries_to_process)
        }
        
        with tqdm(total=total_queries, desc="Processing queries") as pbar:
            # 收集结果
            for future in as_completed(future_to_query):
                query_idx = future_to_query[future]
                try:
                    result = future.result()
                    results.append((query_idx, result))
                    
                    # 更新累计指标
                    wo_rerank_Acc1 += result['wo_rerank']['wo_reranke_acc1']  
                    wo_rerank_Acc5 += result['wo_rerank']['wo_reranke_acc5']
                    wo_rerank_Acc10 += result['wo_rerank']['wo_reranke_acc10']
                    wo_rerank_Mrr += result['wo_rerank']['wo_reranke_mrr']
                    if queries_to_process[query_idx] in bad_data:
                        bad_wo_rerank_Acc1 += result['wo_rerank']['wo_reranke_acc1']
                        bad_wo_rerank_Acc5 += result['wo_rerank']['wo_reranke_acc5']
                        bad_wo_rerank_Acc10 += result['wo_rerank']['wo_reranke_acc10']
                        bad_wo_rerank_Mrr += result['wo_rerank']['wo_reranke_mrr']
                    
                    if queries_to_process[query_idx] in good_data:
                        good_wo_rerank_Acc1 += result['wo_rerank']['wo_reranke_acc1']
                        good_wo_rerank_Acc5 += result['wo_rerank']['wo_reranke_acc5']
                        good_wo_rerank_Acc10 += result['wo_rerank']['wo_reranke_acc10']
                        good_wo_rerank_Mrr += result['wo_rerank']['wo_reranke_mrr']


                    Acc1 += result['acc1']
                    Acc5 += result['acc5'] 
                    Acc10 += result['acc10']
                    Mrr += result['mrr']
                    
                    if result['target_in_candidates']:
                        in_rerank += 1
                    if result['target_in_final']:
                        in_final += 1

                    # bad_data 单独记录指标
                    if queries_to_process[query_idx] in bad_data:
                        bad_Acc1 += result['acc1']
                        bad_Acc5 += result['acc5']
                        bad_Acc10 += result['acc10']
                        bad_Mrr += result['mrr']
                        if result['target_in_candidates']:
                            bad_in_rerank += 1
                        if result['target_in_final']:
                            bad_in_final += 1
                    if queries_to_process[query_idx] in good_data:
                        good_Acc1 += result['acc1']
                        good_Acc5 += result['acc5']
                        good_Acc10 += result['acc10']
                        good_Mrr += result['mrr']
                        if result['target_in_candidates']:
                            good_in_rerank += 1
                        if result['target_in_final']:
                            good_in_final += 1

                    # 打印单个结果
                    p = "  target out of history" if queries_to_process[query_idx] in bad_data else "  target in history"
                    if 'error' not in result:
                        print(f"\n查询 {query_idx}: 目标POI: {result['target']} {p}")
                        print(f"候选检索结果: {result['candidates']} (候选集中是否包括目标POI: {result['target_in_candidates']})")
                        print(f"最终推荐结果: {result['predicted']}(目标POI是否在最终结果中: {result['target_in_final']})")
                        dic_candidates[query_idx] = f"index: {query_idx}, target: {result['target']}, candidates: {result['candidates']}"
                        dic_predicted[query_idx] = f"index: {query_idx}, target: {result['target']}, predicted: {result['predicted']}"
                    else:
                        print(f"\n查询 {query_idx} 处理失败:\n" + "\n".join(result['error']))
                        dic_candidates[query_idx] = f"index: {query_idx}, target: {result['target']}, candidates: []"
                        dic_predicted[query_idx] = f"index: {query_idx}, target: {result['target']}, predicted: []"
                        
                except Exception as exc:
                    print(f'查询 {query_idx} 生成异常: {exc}')
                    results.append((query_idx, {
                        'acc1': 0, 'acc3': 0, 'acc5': 0, 'mrr': 0,
                        'error': str(exc)
                    }))
                
                pbar.update(1)
    
    # 按查询索引排序结果
    results.sort(key=lambda x: x[0])
    
    # 打印最终统计结果
    print(f"\n{'='*50}")
    print(f"处理完成！总查询数: {total_queries}")

    print(f"wo_rerank Accuracy@1: {wo_rerank_Acc1 / total_queries:.4f}")
    print(f"wo_rerank Accuracy@5: {wo_rerank_Acc5 / total_queries:.4f}")
    print(f"wo_rerank Accuracy@10: {wo_rerank_Acc10 / total_queries:.4f}")
    print(f"wo_rerank Mean Reciprocal Rank (MRR): {wo_rerank_Mrr / total_queries:.4f}")
    print(f"在候选集中目标POI的比例: {in_rerank / total_queries:.4f}")

    print(f"Accuracy@1: {Acc1 / total_queries:.4f}")
    print(f"Accuracy@5: {Acc5 / total_queries:.4f}")
    print(f"Accuracy@10: {Acc10 / total_queries:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {Mrr / total_queries:.4f}")
    print(f"在最终推荐结果中目标POI的比例: {in_final / total_queries:.4f}")
    
    # bad_data 额外指标
    if bad_total > 0:
        print(f"\n{'='*50}")
        print(f"bad_data 查询统计 (共 {bad_total} 条):")
        print(f"Accuracy@1: {bad_Acc1 / bad_total:.4f}")
        print(f"Accuracy@5: {bad_Acc5 / bad_total:.4f}")
        print(f"Accuracy@10: {bad_Acc10 / bad_total:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {bad_Mrr / bad_total:.4f}")
        print(f"候选集中包含目标POI比例: {bad_in_rerank / bad_total:.4f}")
        print(f"最终推荐结果中包含目标POI比例: {bad_in_final / bad_total:.4f}")
    
    # good_data 额外指标
    if good_total > 0:
        print(f"\n{'='*50}")
        print(f"good_data 查询统计 (共 {good_total} 条):")
        print(f"Accuracy@1: {good_Acc1 / good_total:.4f}")
        print(f"Accuracy@5: {good_Acc5 / good_total:.4f}")
        print(f"Accuracy@10: {good_Acc10 / good_total:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {good_Mrr / good_total:.4f}")
        print(f"候选集中包含目标POI比例: {good_in_rerank / good_total:.4f}")
        print(f"最终推荐结果中包含目标POI比例: {good_in_final / good_total:.4f}")

    # 保存结果
    with open(f'results/{dataflod}/{model}/{dataflod}_tool_candidates.json', 'w', encoding='utf-8') as f:
        json.dump(dic_candidates, f, ensure_ascii=False, indent=4)
    with open(f'results/{dataflod}/{model}/{dataflod}_tool_predicted.json', 'w', encoding='utf-8') as f:
        json.dump(dic_predicted, f, ensure_ascii=False, indent=4)
    
    # 保存指标
    metrics = {
        'total_queries': total_queries,
        'wo_rerank_Acc1': wo_rerank_Acc1 / total_queries,
        'wo_rerank_Acc5': wo_rerank_Acc5 / total_queries,
        'wo_rerank_Acc10': wo_rerank_Acc10 / total_queries,
        'wo_rerank_Mrr': wo_rerank_Mrr / total_queries,
        'in_rerank_ratio': in_rerank / total_queries,
        'Acc1': Acc1 / total_queries,
        'Acc5': Acc5 / total_queries,
        'Acc10': Acc10 / total_queries,
        'Mrr': Mrr / total_queries,
        'in_final_ratio': in_final / total_queries
    }
    with open(f'results/{dataflod}/{model}/{dataflod}_tool_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到 results/{dataflod}/{model}/{dataflod}_candidates.json 和 results/{dataflod}/{model}/{dataflod}_predicted.json")

    # 保存 bad_data 指标
    if bad_total > 0:
        bad_metrics = {
            'total_queries': bad_total,
            'wo_rerank_Acc1': bad_wo_rerank_Acc1 / bad_total,
            'wo_rerank_Acc5': bad_wo_rerank_Acc5 / bad_total,
            'wo_rerank_Acc10': bad_wo_rerank_Acc10 / bad_total,
            'wo_rerank_Mrr': bad_wo_rerank_Mrr / bad_total,
            'in_rerank_ratio': bad_in_rerank / bad_total,
            'Acc1': bad_Acc1 / bad_total,
            'Acc5': bad_Acc5 / bad_total,
            'Acc10': bad_Acc10 / bad_total,
            'Mrr': bad_Mrr / bad_total,
            'in_rerank_ratio': bad_in_rerank / bad_total,
            'in_final_ratio': bad_in_final / bad_total
        }
        with open(f'results/{dataflod}/{model}/{dataflod}_tool_OOH_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(bad_metrics, f, ensure_ascii=False, indent=4)
        print(f"bad_data 指标已保存到 results/{dataflod}/{model}/{dataflod}_tool_OOH_metrics.json")
    
    # 保存 good_data 指标
    if good_total > 0:
        good_metrics = {
            'total_queries': good_total,
            'wo_rerank_Acc1': good_wo_rerank_Acc1 / good_total,
            'wo_rerank_Acc5': good_wo_rerank_Acc5 / good_total,
            'wo_rerank_Acc10': good_wo_rerank_Acc10 / good_total,
            'wo_rerank_Mrr': good_wo_rerank_Mrr / good_total,
            'in_rerank_ratio': good_in_rerank / good_total,
            'Acc1': good_Acc1 / good_total,
            'Acc5': good_Acc5 / good_total,
            'Acc10': good_Acc10 / good_total,
            'Mrr': good_Mrr / good_total,
            'in_rerank_ratio': good_in_rerank / good_total,
            'in_final_ratio': good_in_final / good_total
        }
        with open(f'results/{dataflod}/{model}/{dataflod}_tool_IH_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(good_metrics, f, ensure_ascii=False, indent=4)
        print(f"good_data 指标已保存到 results/{dataflod}/{model}/{dataflod}_tool_IH_metrics.json")

if __name__ == "__main__":
    main()
