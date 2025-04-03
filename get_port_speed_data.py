from datetime import datetime

import holidays
import numpy as np
import pandas as pd
from get_land_speed_data import land_rate_data as land_data
# 设置输出选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)

port_base_speed = 1  # 港口理想状态下的单设备速度（吨/min）
# 中国节假日
china_holidays = holidays.China(years=[2024, 2025])
holiday_impact = {"工作日": 1.0, "周末": 0.9, "小长假": 0.8, "大长假": 0.7}
# 定义设备状态及对应数值
equip_conditions = ["优", "良", "中", "差", "极差"]
equip_probabilities = [0.3, 0.4, 0.2, 0.07, 0.03]
equip_impact= {"优": 1.0, "良": 0.8, "中": 0.6, "差": 0.4, "极差": 0.2}

# 读取 Excel 文件
df = pd.read_excel("weather_data.xlsx")  # 读取默认的第一个 sheet
n=len(df)

def get_port_weather_factor(weather):
    """
    根据港口天气，返回一个天气影响因子（0～1），
    港口通常对恶劣天气更为敏感，因此数值可能与陆运略有不同。
    """
    mapping = {
        "晴": 0.99,
        "阴": 0.97,
        "多云":0.95,
        "小雨": 0.90,
        "中雨": 0.85,
        "大雨": 0.80,
        "暴雨":0.7,
        "雨夹雪":0.8,
        "雾": 0.80,
    }
    return mapping.get(weather, 1.0)

#返回港口风力影响因子，根据风力等级来
def get_port_wind_factor(wind):
    wind=int(wind)
    if (wind > 10):
        return 0
    wind_level = {
        1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.7,
        6: 0.65, 7: 0.6, 8: 0.55, 9: 0.5, 10: 0.4
    }
    return wind_level.get(wind, 1)

#得到港口某个时间点的时间影响因子
def get_port_time_factor(t):
    t = t.split(' ')[1]
    hour =int(t.split(':')[0])
    if 7<hour<19:
        t_factor=0.99
    else:
        t_factor=0.7
    return t_factor

# 判断节假日
def get_holiday_state(date):
    if date in china_holidays:
        if "春节" in china_holidays[date] or "国庆" in china_holidays[date]:
            return "大长假"
        return "小长假"
    elif date.weekday() in [5, 6]:
        return "周末"
    else:
        return "工作日"
# 生成 n 辆设备的状态并计算均值
def get_average_equip_state(n):
    states = np.random.choice(equip_conditions, size=n, p=equip_probabilities)
    state_values = [equip_impact[state] for state in states]
    state_value=np.mean(state_values)
    # 返回平均值作为整体综合状态
    if state_value >0.8:
        return state_value,"优"
    elif state_value>0.6:
        return state_value,"良"
    elif state_value>0.4:
        return state_value,"中"
    elif state_value>0.2:
        return state_value,"差"
    else:
        return state_value,"极差"

def get_port_rate(land_data):
    #时间作为索引后访问失败
    land_data.reset_index(inplace=True)
    #时间影响因子
    times=land_data["time"]
    time_factors=[get_port_time_factor(t) for t in times]
    #天气风力影响
    weathers=land_data["weather"]
    weather_factors=[get_port_weather_factor(w) for w in weathers]

    winds=land_data["wind"]
    wind_factors=[get_port_wind_factor(w)for w in winds ]

    cargo_types=land_data["cargo_type"]
    cargo_factors=land_data["cargo_factor"]
    holiday_factors=[]

    buffer=land_data["buffer"]
    buffer_factors=land_data["buffer_factor"]

    # 设备影响因子
    equip_nums = []
    equip_prefs = []
    equip_levels=[]

    port_rates=[]
    factors=[]
    for index, row in land_data.iterrows():
        # 节假日，周末影响因子
        date = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")  # 转换为 datetime 类型
        holiday_type = get_holiday_state(date)
        # 获取节假日因子
        holiday_factor = holiday_impact[holiday_type]
        holiday_factors.append(holiday_factor)

        # 随机生成设备数量（例如 1~5）
        equip_num = np.random.randint(1, 5)
        equip_nums.append(equip_num)
        # 为每个设备生成性能，并取平均作为每条记录中的性能值
        # 这批设备的综合状态
        equip_pref, equip_level = get_average_equip_state(equip_num)
        equip_prefs.append(equip_pref)
        equip_levels.append(equip_level)

    # 整理为 DataFrame
    port_data = pd.DataFrame({
        "time": times,
        "time_factor_port": time_factors,
        "cargo_type": cargo_types,
        "cargo_factor": cargo_factors,
        "equip_num": equip_nums,
        "equip_pref": equip_prefs,
        "equip_level":equip_levels,
        "weather": weathers,
        "weather_factor_port": weather_factors,
        "wind":winds,
        "wind_factor_port":wind_factors,
        "holiday_factor":holiday_factors,
        "buffer":buffer,
        "buffer_factor":buffer_factors,
    })
    # 总影响因子
    for index, row in port_data.iterrows():
        factor = row["equip_pref"] * row["weather_factor_port"] * row["wind_factor_port"] * row["cargo_factor"] * row[
            "time_factor_port"] * row["holiday_factor"] * row["buffer_factor"]
        factors.append(factor)
        # 实时速率生成
        port_rate = port_base_speed * factor * row["equip_num"]
        port_rates.append(port_rate)
    port_data["factors"]=factors
    port_data["port_rate"]=port_rates
    return port_data

port_rate_data=get_port_rate(land_data)
