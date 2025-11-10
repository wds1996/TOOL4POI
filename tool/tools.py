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
    poi_file = os.path.join('data', dataflod, 'poi_info.csv')
    poi_df = pd.read_csv(poi_file)

    pid = str(pid)
    poi_df['pid'] = poi_df['pid'].astype(str)

    if pid not in poi_df['pid'].values:
        raise ValueError(f"目标 PID '{pid}' 不在 POI 表中。")

    target_row = poi_df[poi_df['pid'] == pid].iloc[0]
    target_coord = (target_row['latitude'], target_row['longitude'])

    if len(candidate) == 0:
        candidate_df = poi_df[poi_df['pid']]
    else:
        candidate = [str(p) for p in candidate]
        candidate_df = poi_df[poi_df['pid'].isin(candidate)]

    pid_distance_list = []
    for _, row in candidate_df.iterrows():
        coord = (row['latitude'], row['longitude'])
        distance = geodesic(target_coord, coord).kilometers
        pid_distance_list.append((row['pid'], distance))

    sorted_distances = sorted(pid_distance_list, key=lambda x: x[1])
    sorted_pids = [int(pid) for pid, _ in sorted_distances]

    return sorted_pids, len(sorted_pids)


def get_poi_infos(dataflod, poi_list):

    poi_info_path = os.path.join('data', dataflod, 'poi_info.csv')
    poi_infos = {}
    with open(poi_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['pid'])
            if pid in poi_list:
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
    return poi_infos_json


def init_candidates(dataflod, POI_id):
    graph_path = f"data/{dataflod}/poi_transition_graph.csv"
    potential_list = []
    graph_df = pd.read_csv(graph_path)
    graph_dict = {}
    for _, row in graph_df.iterrows():
        pid = int(row['pid'])
        try:
            neighbors = ast.literal_eval(row['potential_poi'])
        except Exception:
            neighbors = []
        graph_dict[pid] = neighbors

    potential_list = graph_dict.get(POI_id, [])
    potential_list = list(set(potential_list))

    return potential_list, len(potential_list)

def get_Interactive_POIs(dataflod, POI_id, candidate=[]):

    graph_path = f"data/{dataflod}/poi_transition_graph.csv"

    graph_df = pd.read_csv(graph_path)
    graph_dict = {}
    for _, row in graph_df.iterrows():
        pid = int(row['pid'])
        try:
            neighbors = ast.literal_eval(row['potential_poi'])
        except Exception:
            neighbors = []
        graph_dict[pid] = neighbors

    potential_list = graph_dict.get(POI_id, [])

    if len(candidate) != 0:
        potential_list = [int(poi) for poi in potential_list]
        potential_list = [pid for pid in potential_list if pid in candidate]
    else:
        potential_list = list(set(potential_list))

    return potential_list, len(potential_list)

def ranking():
    return "Sort the candidate set"

def finish(result):
    return result


def ranking_infos(dataflod, last_poi, poi_list):

    poi_info_path = os.path.join('data', dataflod, 'poi_info.csv')
    poi_infos = {}
    poi_coordinates = {}

    with open(poi_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['pid'])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            poi_coordinates[pid] = (lat, lon)

    if last_poi not in poi_coordinates:
        raise ValueError(f"last_poi {last_poi} not found in POI list.")
    last_coord = poi_coordinates[last_poi]

    with open(poi_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['pid'])
            if pid in poi_list:
                coord = (float(row['latitude']), float(row['longitude']))
                distance = geodesic(last_coord, coord).kilometers

                open_time = row['visit_time_and_count']
                try:
                    open_time_dict = eval(open_time)
                except Exception:
                    open_time_dict = open_time

                poi_infos[pid] = {
                    pid: f"category: {row['category']}, region: {row['region']}, distance_from_last_poi(km): {distance:.2f}"
                }

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
