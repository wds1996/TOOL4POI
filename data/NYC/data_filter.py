import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import csv
from datetime import datetime, timezone, timedelta
from openlocationcode import openlocationcode as olc


# 处理空间位置
def get_pluscode(latitude, longitude):
    # 获取Plus Code
    plus_code = olc.encode(latitude, longitude)
    return plus_code[0:6]


# 处理时间
def format_time(row):
    # 假设tz_offset是以分钟为单位的
    tz_offset = int(row['tz_offset'])
    # 解析原始UTC时间
    original_time = datetime.strptime(row['time'], "%a %b %d %H:%M:%S %z %Y")
    # 计算偏移小时数（从分钟转为小时）
    offset_hours = tz_offset / 60
    # 创建新的时区对象
    from_zone = timezone.utc
    to_zone = timezone(timedelta(hours=offset_hours))
    # 确保原始时间是UTC
    original_time = original_time.replace(tzinfo=from_zone)
    # 转换到目标时区
    adjusted_time = original_time.astimezone(to_zone)
    return adjusted_time.strftime("%Y-%m-%d %H:%M")

# 保持映射关系
def save_mapping(mapping, file_path):
    # 保存uid映射文件
    with open(file_path, "w", newline="") as uidfile:
        writer = csv.writer(uidfile)
        writer.writerow(["original_uid", "new_uid"])
        for original_uid, new_uid in mapping.items():
            writer.writerow([original_uid, new_uid])


if __name__ == "__main__":
    dataflod = "CA"
    min_user_interactions = 10  # 用户交互的最小次数
    min_poi_interactions = 10  # POI交互的最小次数

    # 文件路径
    input_file = f"data/{dataflod}.txt"  # 替换为您的txt文件路径
    # 读取txt文件到DataFrame
    df = pd.read_csv(input_file, delimiter="\t", header=None, names=['uid', 'pid', '_', 'category', 'latitude', 'longitude', 'tz_offset', 'time'])

    # 第一次过滤：过滤掉POI交互次数少于阈值的数据
    poi_counts = df['pid'].value_counts()
    valid_pids = poi_counts[poi_counts >= min_poi_interactions].index
    df_filtered_poi = df[df['pid'].isin(valid_pids)]

    # 第二次过滤：过滤掉用户交互序列少于阈值的用户
    user_counts = df_filtered_poi['uid'].value_counts()
    valid_uids = user_counts[user_counts >= min_user_interactions].index
    df_filtered = df_filtered_poi[df_filtered_poi['uid'].isin(valid_uids)].copy()

    df_filtered["region"] = df_filtered.apply(lambda row: get_pluscode(row['latitude'], row['longitude']), axis=1)

    # 初始化映射字典和计数器
    uid_map = {}
    pid_map = {}
    region_map = {}
    uid_counter = 0
    pid_counter = 0
    region_counter = 0

    # 创建新的uid和pid映射
    df_filtered['new_uid'] = df_filtered['uid'].apply(lambda x: uid_map.setdefault(x, len(uid_map)))
    df_filtered['new_pid'] = df_filtered['pid'].apply(lambda x: pid_map.setdefault(x, len(pid_map)))
    df_filtered['new_region'] = df_filtered['region'].apply(lambda x: region_map.setdefault(x, len(region_map))) 
    df_filtered['formatted_time'] = df_filtered.apply(format_time, axis=1)

    # 保存处理后的数据到csv文件
    df_output = df_filtered[['new_uid', 'new_pid', 'category', 'new_region', 'latitude', 'longitude', 'formatted_time']]
    output_file = f"data/{dataflod}/{dataflod}.csv"
    df_output.to_csv(output_file, index=False, header=['uid', 'pid', 'category', 'region', 'latitude', 'longitude', 'time'])

    # 保存uid映射文件
    uid_map_file = f"data/{dataflod}/uidmap.csv"
    save_mapping(uid_map, uid_map_file)

    # 保存pid映射文件
    pid_map_file = f"data/{dataflod}/pidmap.csv"
    save_mapping(pid_map, pid_map_file)

    # 保存region映射文件
    region_map_file = f"data/{dataflod}/regionmap.csv"
    save_mapping(region_map, region_map_file)

    print("处理完成！")