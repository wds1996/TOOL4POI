import csv
import json
import os
from typing import List
from geopy.distance import geodesic
import pandas as pd
import ast


def filter_poi_by_categories(dataflod: str, categories: List[str], candidate: List[int] = []):
    poi_file = os.path.join('data', dataflod, 'poi_info.csv')
    filtered_pois = []
    with open(poi_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['category'] in categories:
                poi = row['pid']
                filtered_pois.append(int(poi))
    if len(candidate) != 0:
        # 将候选列表转换为整数
        candidate = [int(poi) for poi in candidate]
        filtered_pois = [poi for poi in filtered_pois if poi in candidate]
    
    return filtered_pois, len(filtered_pois)

def filter_poi_by_regions(dataflod: str, regions: List[str], candidate: List[int] = []):
    poi_file = os.path.join('data', dataflod, 'poi_info.csv')
    filtered_pois = []
    with open(poi_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['region']) in regions or row['region'] in regions:
                poi = row['pid']
                filtered_pois.append(int(poi))
    if len(candidate) != 0:
        # 将候选列表转换为整数
        candidate = [int(poi) for poi in candidate]
        filtered_pois = [poi for poi in filtered_pois if poi in candidate]

    return filtered_pois, len(filtered_pois)

def sort_by_pid2candidate(dataflod, pid, candidate=[]):
    # 读取 POI 信息文件
    poi_file = os.path.join('data', dataflod, 'poi_info.csv')
    poi_df = pd.read_csv(poi_file)

    # 确保 POI ID 统一为字符串
    pid = str(pid)
    poi_df['pid'] = poi_df['pid'].astype(str)

    # 获取目标 POI 的经纬度
    if pid not in poi_df['pid'].values:
        raise ValueError(f"目标 PID '{pid}' 不在 POI 表中。")

    target_row = poi_df[poi_df['pid'] == pid].iloc[0]
    target_coord = (target_row['latitude'], target_row['longitude'])

    # 确定候选 POI 列表
    if len(candidate) == 0:
        candidate_df = poi_df[poi_df['pid']]
    else:
        candidate = [str(p) for p in candidate]
        candidate_df = poi_df[poi_df['pid'].isin(candidate)]

    # 计算每个候选 POI 与目标 POI 的距离
    pid_distance_list = []
    for _, row in candidate_df.iterrows():
        coord = (row['latitude'], row['longitude'])
        distance = geodesic(target_coord, coord).kilometers
        pid_distance_list.append((row['pid'], distance))

    # 按距离升序排序
    sorted_distances = sorted(pid_distance_list, key=lambda x: x[1])
    sorted_pids = [int(pid) for pid, _ in sorted_distances]

    return sorted_pids, len(sorted_pids)


def get_poi_infos(dataflod, poi_list):
    """
    传入datafold和poi_list，返回每个POI对应的信息（字典列表）。
    """
    poi_info_path = os.path.join('data', dataflod, 'poi_info.csv')
    poi_infos = {}
    with open(poi_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['pid'])
            if pid in poi_list:
                # 处理open_time为字典
                open_time = row['visit_time_and_count']
                try:
                    open_time_dict = eval(open_time)
                except Exception:
                    open_time_dict = open_time
                # poi_infos[pid] = {
                #     'pid': pid,
                #     'category': row['category'],
                #     'region': row['region'],
                #     # 'visit_time_and_count': open_time_dict
                # }  poi 1 (belong to Bridge, located in region 1)
                poi_infos[pid] = {pid : f"category: {row['category']}, region: {row['region']}"}

    poi_infos_json = [poi_infos[pid] for pid in poi_list]
    # 保证顺序与poi_list一致
    return poi_infos_json


def init_candidates(dataflod, POI_id):
    # 文件路径
    graph_path = f"data/{dataflod}/poi_transition_graph.csv"
    potential_list = []
    # 读取图邻居信息
    graph_df = pd.read_csv(graph_path)
    graph_dict = {}
    for _, row in graph_df.iterrows():
        pid = int(row['pid'])
        try:
            neighbors = ast.literal_eval(row['potential_poi'])
        except Exception:
            neighbors = []
        graph_dict[pid] = neighbors

    # 收集所有邻居 pid
    potential_list = graph_dict.get(POI_id, [])
    potential_list = list(set(potential_list))

    return potential_list, len(potential_list)

def get_Interactive_POIs(dataflod, POI_id, candidate=[]):
    # 文件路径
    graph_path = f"data/{dataflod}/poi_transition_graph.csv"

    # 读取图邻居信息
    graph_df = pd.read_csv(graph_path)
    graph_dict = {}
    for _, row in graph_df.iterrows():
        pid = int(row['pid'])
        try:
            neighbors = ast.literal_eval(row['potential_poi'])
        except Exception:
            neighbors = []
        graph_dict[pid] = neighbors

    # 收集所有邻居 pid
    potential_list = graph_dict.get(POI_id, [])

    # 如果提供了候选列表，则过滤
    if len(candidate) != 0:
        potential_list = [int(poi) for poi in potential_list]
        potential_list = [pid for pid in potential_list if pid in candidate]
    else:
        # 如果没有候选列表，则返回所有邻居
        potential_list = list(set(potential_list))

    return potential_list, len(potential_list)

def ranking():
    return "Sort the candidate set"

def finish(result):
    return result


def ranking_infos(dataflod, last_poi, poi_list):
    """
    传入dataflod和poi_list，返回每个POI对应的信息（字典列表）。
    每个信息包括类别、区域、以及与last_poi的地理距离（单位：公里）。
    """
    poi_info_path = os.path.join('data', dataflod, 'poi_info.csv')
    poi_infos = {}
    poi_coordinates = {}

    # 第一次遍历：记录所有 POI 的经纬度
    with open(poi_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['pid'])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            poi_coordinates[pid] = (lat, lon)

    # 获取 last_poi 的坐标
    if last_poi not in poi_coordinates:
        raise ValueError(f"last_poi {last_poi} not found in POI list.")
    last_coord = poi_coordinates[last_poi]

    # 第二次遍历：构造 poi_list 中每个 POI 的信息
    with open(poi_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['pid'])
            if pid in poi_list:
                coord = (float(row['latitude']), float(row['longitude']))
                distance = geodesic(last_coord, coord).kilometers

                # 解析 open_time 字段
                open_time = row['visit_time_and_count']
                try:
                    open_time_dict = eval(open_time)
                except Exception:
                    open_time_dict = open_time

                poi_infos[pid] = {
                    pid: f"category: {row['category']}, region: {row['region']}, distance_from_last_poi(km): {distance:.2f}"
                }

    # 保证顺序与 poi_list 一致
    poi_infos_json = [poi_infos[pid] for pid in poi_list]
    return poi_infos_json



# pois,lens = filter_poi_by_categories("NYC", ["Arts & Crafts Store", "Gym / Fitness Center"])
# print(pois)

# pois,lens = get_Interactive_POIs("NYC", 0)
# print(pois)

# pois = sort_by_pid2candidate("NYC", 0, candidate=[0,1,2,3,4,5])
# print(pois)

# poi_infos = get_poi_infos("NYC", [0, 1, 2])
# print(poi_infos)

# pois = ranking_infos("NYC", 0, [1, 2, 3])
# print(pois)