import re

import numpy as np
import pandas as pd
import random
import holidays
from datetime import time, datetime

# 设置输出选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)

# 基础速度设定（单位：吨/min）
land_base_speed = 1  # 单设备速度

# 定义货物种类及影响因子
CARGO_TYPES = ['A', 'B', 'C','D']
CARGO_IMPACT = {'A': 0.9, 'B': 0.8, 'C': 0.7,'D':0.6}

# 中国节假日
china_holidays = holidays.China(years=[2024, 2025])
holiday_impact = {"工作日": 1.0, "周末": 0.9, "小长假": 0.8, "大长假": 0.7}

# 定义路况状态和对应的概率
road_conditions = ["优", "良", "中", "差", "极差"]
road_probabilities = [0.6, 0.2, 0.1, 0.07, 0.03]
road_impact = {"优": 1.0, "良": 0.8, "中": 0.6,"差":0.4,"极差":0.2}
# 定义车辆状态及对应数值
vehicle_conditions = ["优", "良", "中", "差", "极差"]
truck_probabilities = [0.3, 0.4, 0.2, 0.07, 0.03]
truck_impact= {"优": 1.0, "良": 0.8, "中": 0.6, "差": 0.4, "极差": 0.2}

buffer_impact={"有":0.9,"无":0.65}

# 读取 Excel 文件
df = pd.read_excel("weather_data.xlsx")  # 读取默认的第一个 sheet
n=len(df)
#print(df.head(5))

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

# 定义陆运的天气因子映射（归一化到0～1）
def get_land_weather_factor(weather):
    """
    根据陆运天气，返回一个天气影响因子（0～1），
    数值越高表示天气条件越理想，对陆运作业影响越小。
    """
    mapping = {
        "晴": 0.99,
        "阴": 0.96,
        "多云":0.92,
        "小雨": 0.85,
        "中雨": 0.75,
        "大雨": 0.65,
        "暴雨":0.6,
        "雨夹雪": 0.70,
        "雾": 0.7,
    }
    return mapping.get(weather, 1.0)
#返回陆地风力影响因子，根据风力等级来
def get_land_wind_factor(wind):
    wind=int(wind)
    if(wind>10):
        return 0
    wind_level={
       1:1,2:0.95,3:0.9,4:0.85,5:0.8,
       6:0.75,7:0.7,8:0.65,9:0.6,10:0.5
    }
    return wind_level.get(wind,1)

#得到陆运某个时间点的时间影响因子
def get_land_time_factor(t):
    t=t.split(' ')[1]
    hour=int(t.split(':')[0])
    if 7<=hour<9 :
        t_factor=0.7
    elif 17<=hour<19:
        t_factor=0.6
    elif 9<=hour<17:
        t_factor=0.99
    else:
        t_factor=0.8
    return t_factor

# 生成 n 辆车的状态并计算均值
def get_average_vehicle_state(n):
    states = np.random.choice(vehicle_conditions, size=n, p=truck_probabilities)
    state_values = [truck_impact[state] for state in states]
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

# -------------------- 陆运记录生成 --------------------
def get_land_rate():
    #不依赖表中数据的部分参数
    # 货物类型影响相关参数
    #提取表格中的日期,随机生成当天的时间点
    times=[]
    time_factors=[]
    holiday_factors = []
    weathers=[]
    weather_factors=[]

    winds=[]
    wind_factors=[]

    roads=[]
    road_factors=[]

    cargo_types = []
    cargo_factors = []

    truck_nums=[]
    truck_levels=[]
    truck_factors=[]

    buffers=[]
    buffer_factors=[]

    land_rates=[]
    factors=[]
    for index, row in df.iterrows():
        # 随机生成时间， 输出格式：HH:MM:SS，例如 14:23:45
        random_time = time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
        t=row["date"]
        t = t.split(' ')[0] + ' ' + str(random_time)
        times.append(t)
        # 获得时间影响因子
        time_factor=get_land_time_factor(t)
        time_factors.append(time_factor)
        #节假日，周末影响因子
        date = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")  # 转换为 datetime 类型
        holiday_type = get_holiday_state(date)
        # 获取节假日因子
        holiday_factor = holiday_impact[holiday_type]
        holiday_factors.append(holiday_factor)
        #天气影响因子生成
        hour = int(str(random_time).split(':')[0])
        if(len(row["weather"])>2):
            weather=row["weather"].split('~')
            if int(hour)<=14:
                weather=weather[0]
            elif int(hour) > 14:
                weather = weather[1]
        else:
            weather=row["weather"]
        weathers.append(weather)
        weather_factor=get_land_weather_factor(weather)
        weather_factors.append(weather_factor)

        #风力影响因子生成
        wind=row["wind"]
        wind=re.findall('[0-9]',wind)[0] if re.findall('[0-9]',wind) else 1
        winds.append(wind)
        wind_factor=get_land_wind_factor(wind)
        wind_factors.append(wind_factor)

        #路况因子生成
        # 随机生成一个路况状态
        road= np.random.choice(road_conditions, p=road_probabilities)
        roads.append(road)
        road_factor=road_impact[road]
        road_factors.append(road_factor)

        #货物影响因子
        cargo_type = np.random.choice(CARGO_TYPES)
        cargo_types.append(cargo_type)
        cargo_factor = CARGO_IMPACT[cargo_type]
        cargo_factors.append(cargo_factor)

        # 随机生成卡车数量（例如 1~10辆）
        truck_num = np.random.randint(1, 11)
        truck_nums.append(truck_num)
        # 这批车的综合状态
        truck_factor,truck_level=get_average_vehicle_state(truck_num)
        truck_factors.append(truck_factor)
        truck_levels.append(truck_level)

        #缓冲区因子
        buffer=np.random.choice(["有","无"],p=[0.6,0.4])
        buffer_factor=buffer_impact[buffer]
        buffers.append(buffer)
        buffer_factors.append(buffer_factor)

        #总影响因子
        factor= truck_factor * road_factor * weather_factor *wind_factor* cargo_factor*time_factor*holiday_factor*buffer_factor
        factors.append(factor)
        #实时速率生成
        land_rate= land_base_speed * factor*truck_num
        land_rates.append(land_rate)

    # 整理为 DataFrame
    land_data = pd.DataFrame({
        "time": times,
        "time_factor_land": time_factors,
        "holiday_factor":holiday_factors,
        "cargo_type": cargo_types,
        "cargo_factor": cargo_factors,
        "truck_num": truck_nums,
        "truck_factor":truck_factors,
        "truck_level":truck_levels,
        "road":roads,
        "road_factor": road_factors,
        "weather": weathers,
        "weather_factor_land": weather_factors,
        "wind":winds,
        "wind_factor_land":wind_factors,
        "buffer":buffers,
        "buffer_factor":buffer_factors,
        "factor":factors,
        "land_rate":land_rates
    })
    land_data.set_index("time", inplace=True)
    return land_data

land_rate_data=get_land_rate()
#print(land_rate_data["land_rate"][:9])
#print(port_rate_data["port_rate"][:9])
#print(land_rate_data.iloc[:2,:])
#print(port_rate_data.iloc[:2,:])






