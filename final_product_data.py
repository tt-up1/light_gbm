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
# 货物量范围（吨）
CARGO_MIN = 10
CARGO_MAX = 1000
# 基础速度设定（单位：吨/min）
land_base_speed = 1  # 单设备速度
port_base_speed = 1  # 港口理想状态下的单设备速度（kg/min）

# 定义货物种类及影响因子（归一化后）
CARGO_TYPES = ['A', 'B', 'C']
CARGO_IMPACT = {'A': 1.0, 'B': 0.8, 'C': 0.6}

# 中国节假日
china_holidays = holidays.China(years=[2024, 2025])
holiday_factors = {"工作日": 1.0, "周末": 0.9, "小长假": 0.8, "大长假": 0.6}
traffic_factors = {"畅通": 1.0, "轻度拥堵": 0.9, "中度拥堵": 0.8, "严重拥堵": 0.6, "事故/封路": 0.4}

# 读取 Excel 文件
df = pd.read_excel("data.xlsx")  # 读取默认的第一个 sheet
n=len(df)
#print(df.head(5))

# 判断节假日
def get_holiday_factor(date):
    if date in china_holidays:
        if "春节" in china_holidays[date] or "国庆" in china_holidays[date]:
            return "大长假"
        return "小长假"
    elif date.weekday() in [5, 6]:
        return "周末"
    else:
        return "工作日"
# 计算最终影响因子
def get_road_factor(weather_factor, time_factor,date):
    date= datetime.strptime(date, "%Y-%m-%d %H:%M:%S")  # 转换为 datetime 类型
    holiday_type = get_holiday_factor(date)
    # 获取各个因子
    holiday_factor = holiday_factors[holiday_type]
    # 计算最终影响因子
    random_adjustment = random.uniform(0.9, 1.1)  # 随机调整
    # 加权求和 + 乘法调整
    final_factor = round((0.4 * weather_factor + 0.3 * time_factor + 0.2 * holiday_factor) * random_adjustment, 2)
    return final_factor

# 定义陆运和港口的天气因子映射（归一化到0～1）
def get_land_weather_factor(weather):
    """
    根据陆运天气，返回一个天气影响因子（0～1），
    数值越高表示天气条件越理想，对陆运作业影响越小。
    """
    mapping = {
        "晴": 1.0,
        "阴": 1,
        "多云":1.0,
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
       1:1,2:0.95,3:0.9,4:0.8,5:0.65,
       6:0.5,7:0.3,8:0.15,9:0.1,10:0
    }
    return wind_level.get(wind,1)
#返回港口风力影响因子，根据风力等级来
def get_port_wind_factor(wind):
    wind=int(wind)
    if (wind > 10):
        return 0
    wind_level = {
        1: 1, 2: 0.9, 3: 0.85, 4: 0.7, 5: 0.55,
        6: 0.35, 7: 0.2, 8: 0.1, 9: 0.05, 10: 0
    }
    return wind_level.get(wind, 1)
def get_port_weather_factor(weather):
    """
    根据港口天气，返回一个天气影响因子（0～1），
    港口通常对恶劣天气更为敏感，因此数值可能与陆运略有不同。
    """
    mapping = {
        "晴": 1.0,
        "阴": 1.0,
        "多云":1.0,
        "小雨": 0.90,
        "中雨": 0.85,
        "大雨": 0.80,
        "暴雨":0.7,
        "雨夹雪":0.8,
        "雾": 0.80,
    }
    return mapping.get(weather, 1.0)
#得到陆运某个时间点的时间影响因子
def get_land_time_factor(t):
    t=t.split(' ')[1]
    hour=int(t.split(':')[0])
    if 7<=hour<9 :
        t_factor=0.6
    elif 17<=hour<19:
        t_factor=0.5
    elif 9<=hour<17:
        t_factor=1
    else:
        t_factor=0.7
    return t_factor
#得到港口某个时间点的时间影响因子
def get_port_time_factor(t):
    t = t.split(' ')[1]
    hour = t.split(':')[0]
    if 7<hour<19:
        t_factor=1
    else:
        t_factor=0.7
    return t_factor
# -------------------- 陆运记录生成 --------------------
def get_land_rate():
    #不依赖表中数据的部分参数
    # 货物类型影响相关参数
    #提取表格中的日期,随机生成当天的时间点
    times=[]
    time_factors=[]
    weathers=[]
    weather_factors=[]
    winds=[]
    wind_factors=[]
    road_factors=[]
    land_rates=[]
    cargo_types=[]
    cargo_factors=[]
    truck_nums=[]
    truck_prefs=[]
    truck_factors=[]
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
        wind=re.findall('[0-9]',wind)[0]
        winds.append(wind)
        wind_factor=get_land_wind_factor(wind)
        wind_factors.append(wind_factor)

        #路况因子生成
        road_factor=get_road_factor(weather_factor,time_factor,t)
        road_factors.append(road_factor)

        #货物影响因子
        cargo_type = np.random.choice(CARGO_TYPES)
        cargo_types.append(cargo_type)
        cargo_factor = CARGO_IMPACT[cargo_type]
        cargo_factors.append(cargo_factor)

        # 随机生成卡车数量（例如 1~10辆）
        truck_num = np.random.randint(1, 11)
        truck_nums.append(truck_num)
        # 为每辆车生成性能，并取平均truck_perf作为每条记录中的性能值
        truck_perf = np.mean(np.random.normal(0.7, 0.1,truck_num).clip(0.0, 1.0))
        truck_prefs.append(truck_perf)
        #print(truck_num,truck_perf,truck_num*truck_perf)
        truck_factors.append(truck_perf)
        #总影响因子
        factor= truck_perf * road_factor * weather_factor *wind_factor* cargo_factor*time_factor
        factors.append(factor)
        #实时速率生成
        land_rate= land_base_speed * factor*truck_num
        land_rates.append(land_rate)
    #路况原始值
    road=[]
    min_road=min(road_factors)
    max_road=max(road_factors)
    s=(max_road-min_road)/4
    for index,row in df.iterrows():
        #print(index,road_factors[index],min_road,s,max_road)
        if  min_road<=road_factors[index]<min_road+s:
            road.append("重度拥堵")
        elif min_road+s<=road_factors[index]<min_road+2*s:
            road.append("中度拥堵")
        elif min_road+2*s<=road_factors[index]<=min_road+3*s:
            road.append("轻度拥堵")
        else:
            road.append("畅通")
        #print(road)
    #print(len(road),len(factors))
    truck_level = []
    min_truck = min(truck_prefs)
    max_truck = max(truck_prefs)
    tr= (max_truck- min_truck) / 4
    for index, row in df.iterrows():
        # print(index,road_factors[index],min_road,s,max_road)
        if min_truck <= truck_prefs[index] <min_truck + tr:
            truck_level.append("整体性能严重下降")
        elif min_truck + tr <= truck_prefs[index] < min_truck + 2 * tr:
            truck_level.append("整体性能重度下降")
        elif min_truck + 2 * tr <= truck_prefs[index] <= min_truck+3*tr:
            truck_level.append("整体性能轻度下降")
        else:
            truck_level.append("整体情况良好")


    '''
    #归一化
    truck_factors = np.array(truck_factors)  # 转为 NumPy 数组
    min_val, max_val = truck_factors.min(), truck_factors.max()
    truck_factors_norm = (truck_factors - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(
        truck_factors)
   
    归一化效果不好
    factors = np.array(factors)  # 转为 NumPy 数组
    min_val, max_val = factors.min(), factors.max()
    factors_norm = (factors - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(
       factors)
       '''
    # 整理为 DataFrame
    land_data = pd.DataFrame({
        "time": times,
        "time_factor": time_factors,
        "cargo_type": cargo_types,
        "cargo_factor": cargo_factors,
        "truck_num": truck_nums,
        "truck_perf": truck_prefs,
        "truck_factors":truck_factors,
        "truck_level":truck_level,
        "road":road,
        "road_factor": road_factors,
        "weather": weathers,
        "weather_factor_land": weather_factors,
        "winds":winds,
        "wind_factors":wind_factors,
        "factor":factors,
        "land_rate":land_rates

    })
    land_data.set_index("time", inplace=True)
    return land_data
def get_port_rate(land_rate_data):
    #不依赖表中数据的部分参数
    # 货物类型影响相关参数
    #提取表格中的日期,随机生成当天的时间点
    #时间作为索引后访问失败
    land_rate_data.reset_index(inplace=True)
    times=land_rate_data["time"]
    time_factors=land_rate_data["time_factor"]
    #天气风力影响
    weathers=land_rate_data["weather"]
    weather_factors=[get_port_weather_factor(w) for w in weathers]
    winds=land_rate_data["winds"]
    wind_factors=[get_port_wind_factor(w)for w in winds ]

    cargo_types=land_rate_data["cargo_type"]
    cargo_factors=land_rate_data["cargo_factor"]

    # 设备影响因子
    equip_nums = []
    equip_prefs = []
    equip_factors = []

    port_rates=[]
    factors=[]
    for index, row in land_rate_data.iterrows():
        # 随机生成设备数量（例如 1~5）
        equip_num = np.random.randint(1, 5)
        equip_nums.append(equip_num)
        # 为每个设备生成性能，并取平均作为每条记录中的性能值
        equip_perf = np.mean(np.random.normal(0.7, 0.1,equip_num).clip(0.0, 1.0))
        equip_prefs.append(equip_perf)
        #equip_factors.append(equip_perf*equip_num),似乎平均性能就是影响因子了
        equip_factors.append(equip_perf)
    '''
    equip_factors = np.array(equip_factors)  # 转为 NumPy 数组
    min_val, max_val = equip_factors.min(), equip_factors.max()
    equip_factors_norm = (equip_factors - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(
        equip_factors)
    '''


    # 整理为 DataFrame
    port_data = pd.DataFrame({
        "time": times,
        "time_factor": time_factors,
        "cargo_type": cargo_types,
        "cargo_factor": cargo_factors,
        "equip_num": equip_nums,
        "equip_perf": equip_prefs,
        "equip_factor":equip_factors,
        "weather": weathers,
        "weather_factor_port": weather_factors,
        "wind_factor":wind_factors,
    })
    print(len(port_data))
    for index, row in port_data.iterrows():
        # 总影响因子
        factor = row["weather_factor_port"] * row["wind_factor"] * row["cargo_factor"] *row["time_factor"]*row["equip_factor"]
        factors.append(factor)
        #实时速率生成
        port_rate= port_base_speed * factor*row["equip_num"]
        port_rates.append(port_rate)
    port_data["factor"]=factors
    port_data["port_rate"]=port_rates
    #port_data.set_index("time", inplace=True)
    return port_data

def get_buffer_factor(land_rate_data,port_rate_data):
    land_rate_data["port_rate"] = port_rate_data["port_rate"]
    port_rate_data["land_rate"] = land_rate_data["land_rate"]
    #缓冲区货物量，影响因子,
    # 逐元素操作，计算buffer_factor
    # 限制buffer_factor的最小值为0
    print(type(port_rate_data["port_rate"]),type(land_rate_data["land_rate"]))
    buffer_factor=1-port_rate_data["port_rate"]/land_rate_data["land_rate"]
    buffer_factor = buffer_factor.clip(lower=0.1)
    buffer_amount =200*buffer_factor
    land_rate_data["buffer_amout"]=buffer_amount
    port_rate_data["buffer_amout"]=buffer_amount
    land_rate_data["buffer_factor"]=buffer_factor
    port_rate_data["buffer_factor"]=buffer_factor

def predic_land_time(land_rate_data,port_rate_data):
    # 随机生成陆地货物量（吨）
    cargo_amount = np.random.uniform(CARGO_MIN, CARGO_MAX, n)
    land_rate_data["cargo_num"]=cargo_amount
    #print(land_rate_data["cargo_num"][:5])
    land_time=(cargo_amount/land_rate_data["land_rate"])*land_rate_data["buffer_factor"]*port_rate_data["port_rate"]
    land_rate_data["pre_land_time"]=land_time
def predic_port_time(land_rate_data,port_rate_data):
    #港口货物量应该是陆运加缓冲
    cargo_mount=land_rate_data["cargo_num"]+port_rate_data["buffer_amout"]
    port_rate_data["cargo_num"]=cargo_mount
    port_time=(cargo_mount/port_rate_data["port_rate"])*port_rate_data["buffer_factor"]*land_rate_data["land_rate"]
    port_rate_data["pre_port_time"]=port_time
land_rate_data=get_land_rate()
#print(land_rate_data["land_rate"][:9])
port_rate_data=get_port_rate(land_rate_data)
#print(port_rate_data["port_rate"][:9])
get_buffer_factor(land_rate_data,port_rate_data)
predic_land_time(land_rate_data,port_rate_data)
predic_port_time(land_rate_data,port_rate_data)
try:
    land_rate_data.to_excel('land_origin_data.xlsx', sheet_name='Sheet1', index=False)
    port_rate_data.to_excel('port_origin_data.xlsx', sheet_name='Sheet1', index=False)

except TypeError as e:
    print(f"遇到类型错误: {e}")
    print("请检查输入的数据是否与预期相符。")
#print(land_rate_data.iloc[:2,:])
#print(port_rate_data.iloc[:2,:])






