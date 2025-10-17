from datetime import datetime

import numpy as np
import pandas as pan
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from until import get_time_state, get_holiday_state, refMapping

pan.set_option('display.max_columns', None)
# pd.s`et_option('display.max_rows', None)
# pd.set_option('display.width', 10000)
# pd.set_option('display.max_colwidth', None)
# 定义所有有序分类变量的映射
mappings_land = {
    'time_type': {'白天': 1, '夜间': 2, '早高峰': 3, '晚高峰': 4},
    'holiday_type': {"工作日": 1, "周末": 2, "小长假": 3, "大长假": 4},
    'cargo_type': {'D': 1, 'C': 2, 'B': 3, 'A': 4},
    'truck_level': {"优": 1, "良": 2, "中": 3, "差": 4, "极差": 5},
    'road_state': {"优": 1, "良": 2, "中": 3, "差": 4, "极差": 5},
    'weather': {"晴": 1, "阴": 2, "多云": 3, "小雨": 4, "中雨": 5,
                "雾": 6,"大雨": 7, "大暴雨": 8,"雨夹雪": 9,  },
    'wind': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
             6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
    'buffer': {"有": 1, "无": 2},
'sub_task_type': {'direct': 1, 'advance': 2} #子任务分类变量
}
mappings_port= {
    'time_type': {'白天': 1, '早高峰': 1, '晚高峰': 1,'夜间': 2},
    'holiday_type': {"工作日": 1, "周末": 2, "小长假": 3, "大长假": 4},
    'cargo_type': {'D': 1, 'C': 2, 'B': 3, 'A': 4},
    'equip_level': {"优": 1, "良": 2, "中": 3, "差": 4, "极差": 5},
    'weather': {"晴": 1, "阴": 2, "多云": 3, "小雨": 4, "中雨": 5,
               "雾":6, "大雨": 7, "雨夹雪": 8,"大暴雨": 9 },
    'wind': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
             6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
    'buffer': {"有": 1, "无": 2}
}

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

def creatTrainData_land():
    data=pan.DataFrame()
    ld=pan.read_excel("land_origin_data.xlsx")
    # #处理顺序型
    data["time_type"]=[get_time_state(t) for t in ld['time']]
    data["cargo_type"]=ld["cargo_type"]
    data["truck_level"]=ld["truck_level"]
    data["road_state"]=ld["road"]
    data["weather"]=ld["weather"]
    data["wind"]=[int(w) for w in ld["wind"]]
    data["buffer"]=ld["buffer"]
    data["holiday_type"]=[get_holiday_state(t) for t in ld['time']]
    data["pre_land_time"]=ld["pre_land_time"]
    data["port_rate"]=ld["port_rate"]
    data["land_rate"]=ld["land_rate"]
    data["truck_num"]=ld["truck_num"]
    data["sub_task_type"] = ld["sub_task_type"]  # New field
    data["p"] = ld["p"]  # New field
    data["batch_id"] = ld["batch_id"]  # New field for potential aggregation
    # 逐列转换
    for col, mapping in mappings_land.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)

    # 用众数填充缺失值，然后转换为 int
    data['weather'] = data['weather'].fillna(data['weather'].mode()[0]).astype(int)
    # 用1填充缺失数据，因为生成阶段是默认晴天
    # data['weather'] = data['weather'].fillna(1).astype(int)
    scaler = MinMaxScaler(feature_range=(0.1, 4))
    data[['cargo_num']] = scaler.fit_transform(ld[['cargo_num']])

    # 使用分位数截断（保留98%数据）
    q_low = data['pre_land_time'].quantile(0.01)
    q_high = data['pre_land_time'].quantile(0.99)
    data = data[(data['pre_land_time'] > q_low) & (data['pre_land_time'] < q_high)]
    '''
     # 对出现次数<5的类别合并
        for col in ['road_state', 'weather',]:
            counts = data[col].value_counts()
            rare_cats = counts[counts < 5].index
            data[col] = data[col].replace(rare_cats, -1)
    '''
    data.to_excel('land_train_data_log.xlsx', sheet_name='Sheet1', index=False)
    data['pre_land_time']=np.log1p(data['pre_land_time'])
    #处理后的训练原始数据集，写入文件
    data.to_excel('land_train_data.xlsx', sheet_name='Sheet1', index=False)
    return data


def creatTrainData_port():
    data=pan.DataFrame()
    pd=pan.read_excel("port_origin_data.xlsx")
    # #处理顺序型
    data["time_type"]=[get_time_state(t) for t in pd['time']]
    data["cargo_type"]=pd["cargo_type"]
    data["equip_level"]=pd["equip_level"]
    data["weather"]=pd["weather"]
    data["wind"]=[int(w) for w in pd["wind"]]
    data["buffer"]=pd["buffer"]
    data["holiday_type"]=[get_holiday_state(t) for t in pd['time']]
    data["pre_port_time"]=pd["pre_port_time"]
    data["port_rate"]=pd["port_rate"]
    data["land_rate"]=pd["land_rate"]
    data["equip_num"]=pd["equip_num"]
    data["batch_id"] =pd["batch_id"]  # New field
    # 逐列转换
    for col, mapping in mappings_port.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)

    # 用众数填充缺失值，然后转换为 int
    data['weather'] = data['weather'].fillna(data['weather'].mode()[0]).astype(int)
    # 用1填充缺失数据，因为生成阶段是默认晴天
    # data['weather'] = data['weather'].fillna(1).astype(int)
    scaler = MinMaxScaler(feature_range=(0.1, 4))
    data[['cargo_num', ]] = scaler.fit_transform(pd[['cargo_num']])

    # 使用分位数截断（保留98%数据）
    q_low = data['pre_port_time'].quantile(0.01)
    q_high = data['pre_port_time'].quantile(0.99)
    data = data[(data['pre_port_time'] > q_low) & (data['pre_port_time'] < q_high)]

    '''
    # 对出现次数<5的类别合并
        for col in [ 'weather',]:
            counts = data[col].value_counts()
            rare_cats = counts[counts < 5].index
            data[col] = data[col].replace(rare_cats, -1)
    '''
    data.to_excel('port_train_data_log.xlsx', sheet_name='Sheet1', index=False)
    data['pre_port_time']=np.log1p(data['pre_port_time'])
    #处理后的训练原始数据集，写入文件
    data.to_excel('port_train_data.xlsx', sheet_name='Sheet1', index=False)


def getRealTestData_land(pddata,scaler):
    pddata = refMapping(pddata, mappings_land)
    for index, row in pddata.iterrows():
        # # 逆变换还原缩放
        original_value = scaler.inverse_transform([[row['cargo_num']]])
        cargo_num_original = original_value[0][0]
        print("缩放前:[cargo_num]", "[", cargo_num_original, "]")

def getRealTestData_port(pddata,scaler):
    pddata = refMapping(pddata, mappings_port)
    for index, row in pddata.iterrows():
        # # 逆变换还原缩放
        original_value = scaler.inverse_transform([[row['cargo_num']]])
        cargo_num_original = original_value[0][0]
        print("缩放前:[cargo_num]", "[", cargo_num_original, "]")

if __name__ == "__main__":
     creatTrainData_land()
     creatTrainData_port()