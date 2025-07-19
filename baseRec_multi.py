from openai import OpenAI
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue


def acc_at_k(preds, target, k):
    """Accuracy@k: 预测前k个中是否命中真实值"""
    if k > len(preds):
        k = len(preds)
    return 1.0 if target in preds[:k] else 0.0

def jug_mrr(preds, target):
    """Mrr: Reciprocal Rank（对单个样本）"""
    for i, p in enumerate(preds):
        if p == target:
            return 1.0 / (i + 1)
    return 0.0

# 线程安全的客户端管理器
class ClientManager:
    def __init__(self, api_key="NaN", base_url="http://localhost:8000/v1", max_clients=10):
        self.api_key = api_key
        self.base_url = base_url
        self.clients = queue.Queue()
        
        # 预创建客户端连接池
        for _ in range(max_clients):
            client = OpenAI(api_key=api_key, base_url=base_url)
            self.clients.put(client)
    
    def get_client(self):
        return self.clients.get()
    
    def return_client(self, client):
        self.clients.put(client)


# 线程安全的千问模型调用函数
def call_qwen_threadsafe(client_manager, model_id, messages):
    client = client_manager.get_client()
    try:
        chat_response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=8192,
            temperature=0,
        )
        return chat_response
    finally:
        client_manager.return_client(client)



def predict(input_str, client_manager, model_id):
  
    prompt = f"""You are a recommendation assistant. 
You need to analyze the user's historical sequence to infer their visiting patterns.
Then, based on both the historical sequence, predict the top *10* POIs the user is most likely to visit at the current time.

Wrap the final result in <result></result> tags, e.g., <result>[pid1, pid2, ...]</result>.
"""
    
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': input_str}
    ]

     # 调用模型
    response = call_qwen_threadsafe(client_manager, model_id, messages)
    messages.append(response.choices[0].message.model_dump())

    result = response.choices[0].message.content

    return result



# 单个查询处理函数
def process_single_query(query_data, client_manager, model_id):
    """处理单个查询的函数"""
    query_str = query_data["input"]
    target = query_data["target"]
    next_time = query_data["next_time"]
    error = []
    
    try:
        input_str = f"""{query_str} Please predict the top *10* POIs according to the likelihood that the user will visit them at {next_time}."""
        respond = predict(input_str, client_manager, model_id)

        # 解析最终推荐结果
        result = re.search(r"<result>(.*?)</result>", respond)
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
            'acc1': acc1,
            'acc5': acc5,
            'acc10': acc10,
            'mrr': mrr,
            'target_in_final': target in predicted_pids if predicted_pids else False
        }
    
    except Exception as e:
        print(f"排序失败: {e}")
        error.append(str(e))
        return {
            'target': target,
            'predicted': [],
            'acc1': 0,
            'acc5': 0,
            'acc10': 0,
            'mrr': 0,
            'target_in_final': False,
            'error': str(e)
        }
    

def main():
    # 指定数据集
    dataflod = "CA" # TKY, NYC, CA
    model_id = "/models/Qwen2.5-14B-Instruct"
    # model_id = "/models/Qwen3-14B"
    
    # 创建客户端管理器
    max_threads = 16  # 可以根据服务器性能调整
    client_manager = ClientManager(
        api_key="NaN",
        base_url="http://localhost:8000/v1",
        max_clients=max_threads
    )
    
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
        poi_match = re.search(r"check-in records:\s*(\[[^\]]+\])", qury["input"])
        if poi_match:
            poi_list_str = poi_match.group(1)
            poi_list = eval(poi_list_str)
        target = qury["target"]
        if target not in poi_list:
            bad_data.append(recent[i])
        else:
            good_data.append(recent[i])
    print(f"数据集 {dataflod} 中共有 {len(history)} 条查询，其中有效查询: {len(good_data)}, 无效查询: {len(bad_data)}")


    
    queries_to_process = recent[:]
    # queries_to_process = bad_data
    
    # 初始化累计指标
    total_queries = len(queries_to_process)
    Acc1 = 0.0
    Acc5 = 0.0
    Acc10 = 0.0
    Mrr = 0.0

    # good_data 单独指标初始化
    good_total = len(good_data)
    good_Acc1 = 0.0
    good_Acc5 = 0.0
    good_Acc10 = 0.0
    good_Mrr = 0.0
    
    results = []
    
    print(f"开始处理 {total_queries} 个查询，使用 {max_threads} 个线程...")
    
    # 使用线程池执行器
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 提交所有任务
        future_to_query = {
            executor.submit(
                process_single_query,
                query,
                client_manager,
                model_id
            ): i for i, query in enumerate(queries_to_process)
        }
        
        # 使用tqdm显示进度
        with tqdm(total=total_queries, desc="Processing queries") as pbar:
            # 收集结果
            for future in as_completed(future_to_query):
                query_idx = future_to_query[future]
                try:
                    result = future.result()
                    results.append((query_idx, result))
                    
                    # 更新累计指标
                    Acc1 += result['acc1']
                    Acc5 += result['acc5'] 
                    Acc10 += result['acc10']
                    Mrr += result['mrr']
                    
                    # good_data 单独指标更新
                    if queries_to_process[query_idx] in good_data:
                        good_Acc1 += result['acc1']
                        good_Acc5 += result['acc5']
                        good_Acc10 += result['acc10']
                        good_Mrr += result['mrr']

                    # 打印单个结果（可选）
                    if 'error' not in result:
                        print(f"\n查询 {query_idx}: 目标POI: {result['target']}")
                        print(f"最终推荐结果: {result['predicted']}(目标POI是否在最终结果中: {result['target_in_final']})")
                        pass
                    else:
                        print(f"\n查询 {query_idx} 处理失败:\n" + "\n".join(result['error']))
                        
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
    print(f"Accuracy@1: {Acc1 / total_queries:.4f}")
    print(f"Accuracy@5: {Acc5 / total_queries:.4f}")
    print(f"Accuracy@10: {Acc10 / total_queries:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {Mrr / total_queries:.4f}")

    # 保存指标
    metrics = {
        'total_queries': total_queries,
        'Acc1': Acc1 / total_queries,
        'Acc5': Acc5 / total_queries,
        'Acc10': Acc10 / total_queries,
        'Mrr': Mrr / total_queries,
    }
    with open(f'results/{dataflod}_metrics_base.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"指标已保存到 results/{dataflod}_base_metrics.json")

     # good_data 额外指标
    if good_total > 0:
        print(f"\n{'='*50}")
        print(f"good_data 查询统计 (共 {good_total} 条):")
        print(f"Accuracy@1: {good_Acc1 / good_total:.4f}")
        print(f"Accuracy@5: {good_Acc5 / good_total:.4f}")
        print(f"Accuracy@10: {good_Acc10 / good_total:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {good_Mrr / good_total:.4f}")
    # 保存 good_data 指标
    if good_total > 0:
        good_metrics = {
            'total_queries': good_total,
            'Acc1': good_Acc1 / good_total,
            'Acc5': good_Acc5 / good_total,
            'Acc10': good_Acc10 / good_total,
            'Mrr': good_Mrr / good_total
        }
        with open(f'results/{dataflod}_base_IH_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(good_metrics, f, ensure_ascii=False, indent=4)
        print(f"good_data 指标已保存到 results/{dataflod}_base_IH_metrics.json")


if __name__ == "__main__":
    main()