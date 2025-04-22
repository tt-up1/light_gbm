from datetime import datetime

import holidays
import numpy as np

# 基础速度设定（单位：吨/min）
land_base_speed = 1  # 单设备速度

# 定义货物种类及影响因子
CARGO_TYPES = ['A', 'B', 'C', 'D']
CARGO_IMPACT = {'A': 0.9, 'B': 0.8, 'C': 0.7, 'D': 0.6}

# 货物量范围（吨）
CARGO_MIN = 10
CARGO_MAX = 1000

# 中国节假日
china_holidays = holidays.China(years=[2024, 2025])
land_holiday_impact = {"工作日": 1.0, "周末": 0.9, "小长假": 0.8, "大长假": 0.7}

# 定义路况状态和对应的数值/概率
road_conditions = ["优", "良", "中", "差", "极差"]
road_probabilities = [0.6, 0.2, 0.1, 0.07, 0.03]
road_impact = {"优": 1.0, "良": 0.8, "中": 0.6, "差": 0.4, "极差": 0.2}
# 定义车辆状态及对应数值
vehicle_conditions = ["优", "良", "中", "差", "极差"]
truck_probabilities = [0.3, 0.4, 0.2, 0.07, 0.03]
truck_impact = {"优": 1.0, "良": 0.8, "中": 0.6, "差": 0.4, "极差": 0.2}
# 定义有无缓冲区
buffer_impact = {"有": 0.9, "无": 0.65}

# 港口理想状态下的单设备速度（吨/min）
port_base_speed = 1

port_holiday_impact = {"工作日": 1.0, "周末": 0.9, "小长假": 0.8, "大长假": 0.7}
# 定义设备状态及对应数值/概率
equip_conditions = ["优", "良", "中", "差", "极差"]
equip_probabilities = [0.3, 0.4, 0.2, 0.07, 0.03]
equip_impact = {"优": 1.0, "良": 0.8, "中": 0.6, "差": 0.4, "极差": 0.2}


# 1. 数据准备与预处理
# 这里假设 land_data 已包含预测目标 'pre_land_time' 和其他特征
def get_time_state(t):
    t = t.split(' ')[1]
    hour = int(t.split(':')[0])
    if 7 <= hour < 9:
        t_state = "早高峰"
    elif 17 <= hour < 19:
        t_state = "晚高峰"
    elif 9 <= hour < 17:
        t_state = "白天"
    else:
        t_state = "夜间"
    return t_state

# 反转mapping
def refMapping(pddata, mappings):
    # 深拷贝一份数据，避免修改原始数据
    data = pddata.copy()

    for key, mapping_dict in mappings.items():
        if key not in data:
            continue  # 如果列不存在，跳过

        # 构建反向映射字典 {1: "白天", 2: "夜间", ...}
        reversed_mapping = {v: k for k, v in mapping_dict.items()}

         # 检查 DataFrame 中的值是否都在映射里
        unique_values = data[key].unique()
        for val in unique_values:
            if val not in reversed_mapping:
                 raise ValueError(f"列 '{key}' 中的值 {val} 没有对应的映射！{key}: {reversed_mapping}")

        # 执行反向映射
        data[key] = data[key].map(reversed_mapping)
    return data

# 判断节假日
def get_holiday_state(date):
    """
        根据日期返回节假日类型。

        参数:
            date (str): 输入的日期，格式为 'YYYY-MM-DD'。

        返回:
            str: 节假日类型，可能的值包括：
                - '大长假'
                - '小长假'
                - '周末'
                - '工作日'
        """
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")  # 转换为 datetime 类型
    if date in china_holidays:
        if "春节" in china_holidays[date] or "国庆" in china_holidays[date]:
            return "大长假"
        return "小长假"
    elif date.weekday() in [5, 6]:
        return "周末"
    else:
        return "工作日"


# 定义陆运的天气因子映射（归一化到0～1）
def get_land_weather_factor(weather):
    """
    根据陆运天气，返回一个天气影响因子（0～1），
    数值越高表示天气条件越理想，对陆运作业影响越小。
    """
    mapping = {
        "晴": 0.99,
        "阴": 0.96,
        "多云": 0.92,
        "小雨": 0.85,
        "中雨": 0.75,
        "大雨": 0.65,
        "暴雨": 0.6,
        "雨夹雪": 0.70,
        "雾": 0.7,
    }
    return mapping.get(weather, 1.0)


# 定义港口的天气因子映射（归一化到0～1）
def get_port_weather_factor(weather):
    """
    根据港口天气，返回一个天气影响因子（0～1），
    港口通常对恶劣天气更为敏感，因此数值可能与陆运略有不同。
    """
    mapping = {
        "晴": 0.99,
        "阴": 0.97,
        "多云": 0.95,
        "小雨": 0.90,
        "中雨": 0.85,
        "大雨": 0.80,
        "暴雨": 0.7,
        "雨夹雪": 0.8,
        "雾": 0.80,
    }
    return mapping.get(weather, 1.0)


# 返回陆地风力影响因子，根据风力等级来
def get_land_wind_factor(wind):
    wind = int(wind)
    if (wind > 10):
        return 0
    wind_level = {
        1: 1, 2: 0.95, 3: 0.9, 4: 0.85, 5: 0.8,
        6: 0.75, 7: 0.7, 8: 0.65, 9: 0.6, 10: 0.5
    }
    return wind_level.get(wind, 1)


# 返回港口风力影响因子，根据风力等级来
def get_port_wind_factor(wind):
    wind = int(wind)
    if (wind > 10):
        return 0
    wind_level = {
        1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.7,
        6: 0.65, 7: 0.6, 8: 0.55, 9: 0.5, 10: 0.4
    }
    return wind_level.get(wind, 1)


# 得到陆运某个时间点的时间影响因子
def get_land_time_factor(t):
    t = t.split(' ')[1]
    hour = int(t.split(':')[0])
    if 7 <= hour < 9:
        t_factor = 0.7
    elif 17 <= hour < 19:
        t_factor = 0.6
    elif 9 <= hour < 17:
        t_factor = 0.99
    else:
        t_factor = 0.8
    return t_factor


# 得到港口某个时间点的时间影响因子
def get_port_time_factor(t):
    t = t.split(' ')[1]
    hour = int(t.split(':')[0])
    if 7 < hour < 19:
        t_factor = 0.99
    else:
        t_factor = 0.7
    return t_factor


# 生成 n 辆车的状态并计算均值
def get_average_vehicle_state(n):
    states = np.random.choice(vehicle_conditions, size=n, p=truck_probabilities)
    state_values = [truck_impact[state] for state in states]
    state_value = np.mean(state_values)
    # 返回平均值作为整体综合状态
    if state_value > 0.8:
        return state_value, "优"
    elif state_value > 0.6:
        return state_value, "良"
    elif state_value > 0.4:
        return state_value, "中"
    elif state_value > 0.2:
        return state_value, "差"
    else:
        return state_value, "极差"


# 生成 n 辆设备的状态并计算均值
def get_average_equip_state(n):
    states = np.random.choice(equip_conditions, size=n, p=equip_probabilities)
    state_values = [equip_impact[state] for state in states]
    state_value = np.mean(state_values)
    # 返回平均值作为整体综合状态
    if state_value > 0.8:
        return state_value, "优"
    elif state_value > 0.6:
        return state_value, "良"
    elif state_value > 0.4:
        return state_value, "中"
    elif state_value > 0.2:
        return state_value, "差"
    else:
        return state_value, "极差"