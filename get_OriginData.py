# 提取表格中的日期,随机生成当天的时间点
import datetime
import random
import re
from datetime import time

import numpy as np
import pandas as pd


from until import land_base_speed, port_base_speed, get_average_equip_state, buffer_impact, get_average_vehicle_state, \
    CARGO_IMPACT, CARGO_TYPES, road_impact, road_conditions, road_probabilities, get_port_wind_factor, \
    get_land_wind_factor, get_port_weather_factor, get_land_weather_factor, land_holiday_impact, port_holiday_impact, \
    get_holiday_state, get_port_time_factor, get_land_time_factor, CARGO_MIN, CARGO_MAX


class TrainData:
    # 提取表格中的日期,随机生成当天的时间点
    times = []
    # 陆运时间影响因子
    land_time_factors = []
    # 装船时间影响因子
    port_time_factors = []

    # 陆运节假日因子
    land_holiday_factors = []
    # 装船节假日因子
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
    truck_states = []

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

    # 陆地总影响因子
    land_factors = []
    # 海运总影响因子
    port_factors = []

    #货物量
    cargo_num=[ ]

    batch_ids = []  # New: batch_id 标识一批货物
    sub_task_types = []  # New: 子任务类型 直装/提前到港
    p_values = []  # New:  直装比例


    def __init__(self, df):
        """
        初始化对象的不依赖于表格数据的属性
        :param row: [], 天气数据行
        """
        for index, row in df.iterrows():
            batch_id = f"batch_{index:04d}"  # 利用每一天天气记录id生成唯一 batch_id
            total_cargo = np.random.uniform(CARGO_MIN, CARGO_MAX)
            p = np.random.uniform(0.0, 1.0) #随机比例
            cargo_direct = p * total_cargo #直装量
            cargo_advance = (1 - p) * total_cargo  #提前量

            # 随机生成当天的时间，得到当天的陆运和装船时间 输出格式：HH:MM:SS，例如 14:23:45
            #random_time = time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
            random_time=datetime.time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
            t = row["date"]
            t = t.split(' ')[0] + ' ' + str(random_time)
           # self.times.append(t)
            self._generate_record(row,t, batch_id, 'direct', cargo_direct, p)

            #得到前一天的随机时间，生成提前到港记录
            prev_day = (datetime.datetime.strptime(row["date"].split(' ')[0], "%Y-%m-%d") - datetime.timedelta(
                days=1)).strftime("%Y-%m-%d")
            random_prev_time = datetime.time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
            prev_time_str = prev_day + ' ' + random_prev_time.strftime("%H:%M:%S")
            self._generate_record(row, prev_time_str, batch_id, 'advance', cargo_advance, p)

    def _generate_record(self, row, t, batch_id, sub_task_type, cargo_amount, p):
            self.times.append(t)
            self.batch_ids.append(batch_id)
            self.sub_task_types.append(sub_task_type)
            self.p_values.append(p)  # Append p for both

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
            # 获取装船假日因子
            port_holiday_factor = port_holiday_impact[holiday_type]
            self.port_holiday_factors.append(port_holiday_factor)

            # 天气影响因子生成
            hour = int(t.split(' ')[1].split(':')[0])

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
            # 装船天气因子
            port_weather_factor = get_port_weather_factor(weather)
            self.port_weather_factors.append(port_weather_factor)

            # 风力影响因子生成
            # wind = row["wind"]
            # wind = re.findall('[0-9]', wind)[0] if re.findall('[0-9]', wind) else 1
            wind_str = str(row.get("wind", "1"))
            wind_digits = re.findall(r'\d+', wind_str)
            wind = int(wind_digits[0]) if wind_digits else 1
            self.winds.append(wind)

            # 陆运风力因子
            land_wind_factor = get_land_wind_factor(wind)
            self.land_wind_factors.append(land_wind_factor)
            # 装船风力因子
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
            truck_state, truck_level = get_average_vehicle_state(truck_num)
            self.truck_states.append(truck_state)
            self.truck_levels.append(truck_level)

            # 缓冲区因子 提前到港和直装不同
            if sub_task_type == 'advance':  #保持数据一致性 弱化影响
                buffer =np.random.choice(["有", "无"], p=[0.9, 0.1])
                buffer_factor = buffer_impact[buffer]if buffer == "有" else 0.95
            else:  # direct
                buffer = np.random.choice(["有", "无"], p=[0.6, 0.4])
                buffer_factor = buffer_impact[buffer]
            self.buffers.append(buffer)
            self.buffer_factors.append(buffer_factor)

            # 随机生成设备数量（例如 1~5）
            equip_num = np.random.randint(1, 6)
            self.equip_nums.append(equip_num)
            # 为每个设备生成性能，并取平均作为每条记录中的性能值
            # 这批设备的综合状态
            equip_pref, equip_level = get_average_equip_state(equip_num)
            self.equip_prefs.append(equip_pref)
            self.equip_levels.append(equip_level)

            # 装船总影响因子（只对direct乘，或弱化）
            if sub_task_type == 'direct':
                port_factor = equip_pref * (
                            equip_num / 5) * port_weather_factor * port_wind_factor * cargo_factor * port_time_factor * port_holiday_factor * buffer_factor
            else:
                port_factor = equip_pref * (
                            equip_num / 5) * port_weather_factor * port_wind_factor * cargo_factor * port_time_factor * port_holiday_factor  # 不乘buffer_factor
            # 装船总影响因子
            #port_factor = equip_pref *(equip_num/5 )*port_weather_factor * port_wind_factor * cargo_factor * \
                      #    port_time_factor * port_holiday_factor * buffer_factor
            self.port_factors.append(port_factor)

            # 海运实时速率生成
            port_rate = port_base_speed * port_factor
            self.port_rates.append(port_rate)

            # 陆运影响因子
            land_factor = (truck_num/10)*truck_state* road_factor * land_weather_factor * land_wind_factor * cargo_factor * land_time_factor * land_holiday_factor * buffer_factor
            self.land_factors.append(land_factor)
            # 陆运实时速率生成
            land_rate = land_base_speed * land_factor
            self.land_rates.append(land_rate)

            #随机生成货物量
            #cargo_amount = np.random.uniform(CARGO_MIN, CARGO_MAX,1)[0]
            self.cargo_num.append(cargo_amount)

    def to_dataframe_land(self):
        # 整理为 DataFrame
        land_data = pd.DataFrame({
            "batch_id": self.batch_ids,
            "sub_task_type": self.sub_task_types,
            "p": self.p_values,
            "time": self.times,
            "time_factor_land": self.land_time_factors,
            "holiday_factor": self.land_holiday_factors,
            "cargo_type": self.cargo_types,
            "cargo_factor": self.cargo_factors,
            "truck_num": self.truck_nums,
            "truck_state": self.truck_states,
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
            "port_rate": self.port_rates,
            "cargo_num":self.cargo_num
        })
        # 港口速率影响标准化 0.1之间
        #land_time = (land_data["cargo_num"] / land_data["land_rate"]) * land_data["port_rate"]
        #land_data["pre_land_time"] = land_time

        # 基础计算：所有行都用 cargo_num / land_rate
        land_data['pre_land_time'] = land_data['cargo_num'] / land_data['land_rate']

        # 只对 'direct' 乘 port_rate， 直装时完成时间才会受港口速率影响
        mask_direct = land_data['sub_task_type'] == 'direct'
        land_data.loc[mask_direct, 'pre_land_time'] *= land_data.loc[mask_direct, 'port_rate']
        # land_data.set_index("time", inplace=True)
        return land_data
    def to_dataframe_port(self):
        # 整理为 DataFrame，港口记录 每个批次只有一次装船记录
        port_data_list = []
        unique_batches = list(set(self.batch_ids))
        for b in unique_batches:
            # Find index of 'direct' for the batch (use day's data)
            direct_idx = [i for i, (bid, st) in enumerate(zip(self.batch_ids, self.sub_task_types)) if
                          bid == b and st == 'direct']
            idx = direct_idx[0] if direct_idx else self.batch_ids.index(b)  # Fallback
            port_data_list.append({
                "batch_id": b,
                "time": self.times[idx],
                "time_factor_port": self.port_time_factors[idx],
                "holiday_factor": self.port_holiday_factors[idx],
                "cargo_type": self.cargo_types[idx],
                "cargo_factor": self.cargo_factors[idx],
                "equip_num": self.equip_nums[idx],
                "equip_state": self.equip_prefs[idx],
                "equip_level": self.equip_levels[idx],
                "weather": self.weathers[idx],
                "weather_factor_port": self.port_weather_factors[idx],
                "wind": self.winds[idx],
                "wind_factor_port": self.port_wind_factors[idx],
                "buffer": self.buffers[idx],
                "buffer_factor": self.buffer_factors[idx],
                "factor": self.port_factors[idx],
                "land_rate": self.land_rates[idx],
                "port_rate": self.port_rates[idx],
                "cargo_num": sum([c for i, c in enumerate(self.cargo_num) if self.batch_ids[i] == b])  # Total cargo
            })
        port_data = pd.DataFrame(port_data_list)
        #港口完成时间受直装货物陆运速率影响
        port_time = (port_data["cargo_num"] / port_data["port_rate"]) * port_data["land_rate"]
        port_data["pre_port_time"] = port_time
        return port_data

def get_data():
    df = pd.read_excel("weather_data.xlsx")
    train = TrainData(df)
    return train.to_dataframe_land(), train.to_dataframe_port()

#主程序运行一次，下次在缓存中引入
if __name__ == "__main__":
    land_data, port_data = get_data()
    land_data.to_excel('land_origin_data.xlsx', sheet_name='Sheet1', index=False)
    port_data.to_excel('port_origin_data.xlsx', sheet_name='Sheet1', index=False)


