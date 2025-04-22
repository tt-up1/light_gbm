# 提取表格中的日期,随机生成当天的时间点
import random
import re
from datetime import time,datetime

import numpy as np
import pandas as pd


from until import land_base_speed, port_base_speed, get_average_equip_state, buffer_impact, get_average_vehicle_state, \
    CARGO_IMPACT, CARGO_TYPES, road_impact, road_conditions, road_probabilities, get_port_wind_factor, \
    get_land_wind_factor, get_port_weather_factor, get_land_weather_factor, land_holiday_impact, port_holiday_impact, \
    get_holiday_state, get_port_time_factor, get_land_time_factor, CARGO_MIN, CARGO_MAX

# 设置输出选项
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 10000)
# pd.set_option('display.max_colwidth', None)

# df = pd.read_excel("weather_data.xlsx")  # 读取默认的第一个 sheet
# n = len(df)
#
# print(df.head(5))

class LandData:
    times = []
    time_factors = []
    holiday_factors = []
    weathers = []
    weather_factors = []

    winds = []
    wind_factors = []

    roads = []
    road_factors = []

    cargo_types = []
    cargo_factors = []

    truck_nums = []
    truck_levels = []
    truck_factors = []

    buffers = []
    buffer_factors = []

    land_rates = []
    factors = []

    def __init__(self, df):
        """
        初始化对象的属性
        :param row: pandas, 天气数据行
        """
        for index, row in df.iterrows():
            # 随机生成时间， 输出格式：HH:MM:SS，例如 14:23:45
            random_time = time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
            t = row["date"]
            t = t.split(' ')[0] + ' ' + str(random_time)
            self.times.append(t)
            # 获得时间影响因子
            time_factor = get_land_time_factor(t)
            self.time_factors.append(time_factor)
            # 节假日，周末影响因子
          #  date = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")  # 转换为 datetime 类型
            holiday_type = get_holiday_state(t)
            # 获取节假日因子
            holiday_factor = land_holiday_impact[holiday_type]
            self.holiday_factors.append(holiday_factor)
            # 天气影响因子生成
            hour = int(str(random_time).split(':')[0])
            if (len(row["weather"]) > 2):
                weather = row["weather"].split('~')
                if int(hour) <= 14:
                    weather = weather[0]
                elif int(hour) > 14:
                    weather = weather[1]
            else:
                weather = row["weather"]
            self.weathers.append(weather)
            weather_factor = get_land_weather_factor(weather)
            self.weather_factors.append(weather_factor)

            # 风力影响因子生成
            wind = row["wind"]
            wind = re.findall('[0-9]', wind)[0] if re.findall('[0-9]', wind) else 1
            self.winds.append(wind)
            wind_factor = get_land_wind_factor(wind)
            self.wind_factors.append(wind_factor)

            # 路况因子生成
            # 随机生成一个路况状态
            road = np.random.choice(road_conditions, p=road_probabilities)
            self.roads.append(road)
            road_factor = road_impact[road]
            self.road_factors.append(road_factor)

            # 货物影响因子
            cargo_type = np.random.choice(CARGO_TYPES)
            self.cargo_types.append(cargo_type)
            cargo_factor = CARGO_IMPACT[cargo_type]
            self.cargo_factors.append(cargo_factor)

            # 随机生成卡车数量（例如 1~10辆）
            truck_num = np.random.randint(1, 11)
            self.truck_nums.append(truck_num)
            # 这批车的综合状态
            truck_factor, truck_level = get_average_vehicle_state(truck_num)
            self.truck_factors.append(truck_factor)
            self.truck_levels.append(truck_level)

            # 缓冲区因子
            buffer = np.random.choice(["有", "无"], p=[0.6, 0.4])
            buffer_factor = buffer_impact[buffer]
            self.buffers.append(buffer)
            self.buffer_factors.append(buffer_factor)

            # 总影响因子
            factor = truck_factor * road_factor * weather_factor * wind_factor * cargo_factor * time_factor * holiday_factor * buffer_factor
            self.factors.append(factor)
            # 实时速率生成
            land_rate = land_base_speed * factor * truck_num
            self.land_rates.append(land_rate)

    def to_dataframe(self):
        # 整理为 DataFrame
        land_data = pd.DataFrame({
            "time": self.times,
            "time_factor_land": self.time_factors,
            "holiday_factor": self.holiday_factors,
            "cargo_type": self.cargo_types,
            "cargo_factor": self.cargo_factors,
            "truck_num": self.truck_nums,
            "truck_factor": self.truck_factors,
            "truck_level": self.truck_levels,
            "road": self.roads,
            "road_factor": self.road_factors,
            "weather": self.weathers,
            "weather_factor_land": self.weather_factors,
            "wind": self.winds,
            "wind_factor_land": self.wind_factors,
            "buffer": self.buffers,
            "buffer_factor": self.buffer_factors,
            "factor": self.factors,
            "land_rate": self.land_rates
        })
        land_data.set_index("time", inplace=True)
        return land_data


class TrainData:
    # 提取表格中的日期,随机生成当天的时间点
    times = []
    # 陆运时间影响因子
    land_time_factors = []
    # 海运时间影响因子
    port_time_factors = []

    # 陆运节假日因子
    land_holiday_factors = []
    # 海运节假日因子
    port_holiday_factors = []

    # 天气影响因子
    weathers = []
    land_weather_factors = []
    port_weather_factors = []

    # 风力影响因子
    winds = []
    land_wind_factors = []
    port_wind_factors = []

    # 路况因子
    roads = []
    road_factors = []

    # 货物影响因子
    cargo_types = []
    cargo_factors = []

    # 这批车的综合状态
    truck_nums = []
    truck_levels = []
    truck_factors = []

    # 缓冲区因子
    buffers = []
    buffer_factors = []

    # 设备影响因子
    equip_nums = []
    equip_prefs = []
    equip_levels = []

    # 陆地速率
    land_rates = []
    # 船舶速率
    port_rates = []

    # 总影响因子
    factors = []

    # 陆地总影响因子
    land_factors = []
    # 海运总影响因子
    port_factors = []

    # 船舶总影响因子
    factors = []

    def __init__(self, df):
        """
        初始化对象的属性
        :param row: [], 天气数据行
        """
        for index, row in df.iterrows():
            # 随机生成时间， 输出格式：HH:MM:SS，例如 14:23:45
            random_time = time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
            t = row["date"]
            t = t.split(' ')[0] + ' ' + str(random_time)
            self.times.append(t)

            # 获得时间影响因子
            # 获得陆运时间影响因子
            land_time_factor = get_land_time_factor(t)
            self.land_time_factors.append(land_time_factor)
            # 获得海运时间影响因子
            port_time_factor = get_port_time_factor(t)
            self.port_time_factors.append(port_time_factor)

            # 节假日，周末影响因子
            #date = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")  # 转换为 datetime 类型
            holiday_type = get_holiday_state(t)

            # 获取陆运节假日因子
            land_holiday_factor = land_holiday_impact[holiday_type]
            self.land_holiday_factors.append(land_holiday_factor)
            # 获取海运节假日因子
            port_holiday_factor = port_holiday_impact[holiday_type]
            self.port_holiday_factors.append(port_holiday_factor)

            # 天气影响因子生成
            hour = int(str(random_time).split(':')[0])
            if (len(row["weather"]) > 2):
                weather = row["weather"].split('~')
                if int(hour) <= 14:
                    weather = weather[0]
                elif int(hour) > 14:
                    weather = weather[1]
            else:
                weather = row["weather"]
            self.weathers.append(weather)

            # 陆运天气因子
            land_weather_factor = get_land_weather_factor(weather)
            self.land_weather_factors.append(land_weather_factor)
            # 海运天气因子
            port_weather_factor = get_port_weather_factor(weather)
            self.port_weather_factors.append(port_weather_factor)

            # 风力影响因子生成
            wind = row["wind"]
            wind = re.findall('[0-9]', wind)[0] if re.findall('[0-9]', wind) else 1
            self.winds.append(wind)

            # 陆运风力因子
            land_wind_factor = get_land_wind_factor(wind)
            self.land_wind_factors.append(land_wind_factor)
            # 海运风力因子
            port_wind_factor = get_port_wind_factor(wind)
            self.port_wind_factors.append(port_wind_factor)

            # 路况因子生成
            # 随机生成一个路况状态
            road = np.random.choice(road_conditions, p=road_probabilities)
            self.roads.append(road)
            road_factor = road_impact[road]
            self.road_factors.append(road_factor)

            # 货物影响因子
            cargo_type = np.random.choice(CARGO_TYPES)
            self.cargo_types.append(cargo_type)
            cargo_factor = CARGO_IMPACT[cargo_type]
            self.cargo_factors.append(cargo_factor)

            # 随机生成卡车数量（例如 1~10辆）
            truck_num = np.random.randint(1, 11)
            self.truck_nums.append(truck_num)
            # 这批车的综合状态
            truck_factor, truck_level = get_average_vehicle_state(truck_num)
            self.truck_factors.append(truck_factor)
            self.truck_levels.append(truck_level)

            # 缓冲区因子
            buffer = np.random.choice(["有", "无"], p=[0.6, 0.4])
            buffer_factor = buffer_impact[buffer]
            self.buffers.append(buffer)
            self.buffer_factors.append(buffer_factor)

            # 随机生成设备数量（例如 1~5）
            equip_num = np.random.randint(1, 5)
            self.equip_nums.append(equip_num)
            # 为每个设备生成性能，并取平均作为每条记录中的性能值
            # 这批设备的综合状态
            equip_pref, equip_level = get_average_equip_state(equip_num)
            self.equip_prefs.append(equip_pref)
            self.equip_levels.append(equip_level)

            # 海运总影响因子
            port_factor = equip_pref * port_weather_factor * port_wind_factor * cargo_factor * \
                          port_time_factor * port_holiday_factor * buffer_factor
            self.port_factors.append(port_factor)

            # 海运实时速率生成
            port_rate = port_base_speed * port_factor * equip_num
            self.port_rates.append(port_rate)

            # 陆运影响因子
            land_factor = truck_factor * road_factor * land_weather_factor * land_wind_factor * cargo_factor * land_time_factor * land_holiday_factor * buffer_factor
            self.land_factors.append(land_factor)
            # 陆运实时速率生成
            land_rate = land_base_speed * land_factor * truck_num
            self.land_rates.append(land_rate)

    def to_dataframe(self):
        # 整理为 DataFrame
        land_data = pd.DataFrame({
            "time": self.times,
            "time_factor_land": self.land_time_factors,
            "holiday_factor": self.land_holiday_factors,
            "cargo_type": self.cargo_types,
            "cargo_factor": self.cargo_factors,
            "truck_num": self.truck_nums,
            "truck_factor": self.truck_factors,
            "truck_level": self.truck_levels,
            "road": self.roads,
            "road_factor": self.road_factors,
            "weather": self.weathers,
            "weather_factor_land": self.land_weather_factors,
            "wind": self.winds,
            "wind_factor_land": self.land_wind_factors,
            "buffer": self.buffers,
            "buffer_factor": self.buffer_factors,
            "factor": self.land_factors,
            "land_rate": self.land_rates,
            "port_rate": self.port_rates
        })
        n = len(land_data)
        # 随机生成陆地货物量（吨）
        cargo_amount = np.random.uniform(CARGO_MIN, CARGO_MAX, n)
        land_data["cargo_num"] = cargo_amount
        # 港口速率影响标准化 0.1之间
        land_time = (cargo_amount / land_data["land_rate"]) * land_data["port_rate"]
        land_data["pre_land_time"] = land_time

        # land_data.set_index("time", inplace=True)
        return land_data