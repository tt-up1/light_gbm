from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 设置输出选项
from TrainData import TrainData
from until import get_time_state, get_holiday_state, refMapping

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 10000)
# pd.set_option('display.max_colwidth', None)
# 定义所有有序分类变量的映射
mappings = {
    'time_type': {'白天': 1, '夜间': 2, '早高峰': 3, '晚高峰': 4},
    'holiday_type': {"工作日": 1, "周末": 2, "小长假": 3, "大长假": 4},
    'cargo_type': {'D': 1, 'C': 2, 'B': 3, 'A': 4},
    'truck_state': {"优": 1, "良": 2, "中": 3, "差": 4, "极差": 5},
    'road_state': {"优": 1, "良": 2, "中": 3, "差": 4, "极差": 5},
    'weather': {"晴": 1, "阴": 2, "多云": 3, "小雨": 4, "中雨": 5,
                "大雨": 6, "大暴雨": 7, "雨夹雪": 8, "雾": 9},
    'wind': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
             6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
    'buffer': {"有": 1, "无": 2}
}
def creatTrainData():
    df = pd.read_excel("weather_data.xlsx")

    train = TrainData(df)
    ld=train.to_dataframe()

    data=pd.DataFrame()
    # #处理顺序型
    data["time_type"]=[get_time_state(t) for t in ld['time']]
    data["cargo_type"]=ld["cargo_type"]
    data["truck_state"]=ld["truck_level"]
    data["road_state"]=ld["road"]
    data["weather"]=ld["weather"]
    data["wind"]=[int(w) for w in ld["wind"]]
    data["buffer"]=ld["buffer"]
    data["holiday_type"]=[get_holiday_state(t) for t in ld['time']]
    data["pre_land_time"]=ld["pre_land_time"]
    data["port_rate"]=ld["port_rate"]
    data["land_rate"]=ld["land_rate"]
    data["truck_num"]=ld["truck_num"]
    def checkMapping(pddata, mappings):
        data = pddata.copy()
        for key, mapping_dict in mappings.items():
            if key not in pddata:
                raise ValueError(f"列 '{key}' 在数据中不存在!")

            unique_values = data[key].unique()
            for val in unique_values:
                # print(val)
                if val not in mappings[key]:
                    # print(unique_values)
                    print(f"列 '{key}' 中的值 '{val}'没有对应的映射！{key}: {mappings[key]}")
                    # raise ValueError(f"列 '{key}' 中的值 '{val}'没有对应的映射！{key}: {mappings[key]}")
            data[key] = data[key].map(mappings[key])
        return data
    # 校验是否缺映射
    # checkMapping(data,mappings)
    # 逐列转换
    for col, mapping in mappings.items():
        data[col] = data[col].map(mapping)

    # 用众数填充缺失值，然后转换为 int
    data['weather'] = data['weather'].fillna(data['weather'].mode()[0]).astype(int)
    # 用1填充缺失数据，因为生成阶段是默认晴天
    # data['weather'] = data['weather'].fillna(1).astype(int)
    scaler = MinMaxScaler(feature_range=(0.1, 4))
    data[['cargo_num', ]] = scaler.fit_transform(ld[['cargo_num']])

    # 使用分位数截断（保留98%数据）
    q_low = data['pre_land_time'].quantile(0.01)
    q_high = data['pre_land_time'].quantile(0.99)
    data = data[(data['pre_land_time'] > q_low) & (data['pre_land_time'] < q_high)]

    # 对出现次数<5的类别合并
    for col in ['road_state', 'weather',]:
        counts = data[col].value_counts()
        rare_cats = counts[counts < 5].index
        data[col] = data[col].replace(rare_cats, -1)
    data.to_excel('land_train_data_log.xlsx', sheet_name='Sheet1', index=False)
    data['pre_land_time']=np.log1p(data['pre_land_time'])
    #处理后的训练原始数据集，写入文件
    data.to_excel('land_train_data.xlsx', sheet_name='Sheet1', index=False)
def getTrainData():
    ld = pd.read_excel("land_train_data.xlsx")
    data = pd.DataFrame()
    data["time_type"] = ld["time_type"]
    data["cargo_type"] = ld["cargo_type"]
    data["truck_state"] = ld["truck_state"]
    data["road_state"] = ld["road_state"]
    data["weather"] = ld["weather"]
    data["wind"] = ld["wind"]
    data["buffer"] = ld["buffer"]
    data["holiday_type"] = ld["holiday_type"]
    data["pre_land_time"] = ld["pre_land_time"]
    data["port_rate"] = ld["port_rate"]
    data["land_rate"] = ld["land_rate"]
    data["truck_num"] = ld["truck_num"]
    data['cargo_num'] =ld["cargo_num"]
    return data

def getRealTestData(pddata,scaler):
    pddata = refMapping(pddata, mappings)
    for index, row in pddata.iterrows():
        # # 逆变换还原缩放
        original_value = scaler.inverse_transform([[row['cargo_num']]])
        cargo_num_original = original_value[index][0]
        print("缩放前:[cargo_num]", "[", cargo_num_original, "]")