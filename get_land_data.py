import numpy as np
from get_land_speed_data import land_rate_data as land_data
from get_port_speed_data import port_rate_data as port_data

# 货物量范围（吨）
CARGO_MIN = 10
CARGO_MAX = 1000

def predic_land_time(land_data,port_data):
    n=len(land_data)
    # 随机生成陆地货物量（吨）
    cargo_amount = np.random.uniform(CARGO_MIN, CARGO_MAX, n)
    land_data["cargo_num"]=cargo_amount
    land_data["port_rate"]=port_data["port_rate"]
    #港口速率影响标准化 0.1之间
    land_time=(cargo_amount/land_data["land_rate"])*land_data["port_rate"]
    land_data["pre_land_time"]=land_time

predic_land_time(land_data,port_data)

try:
    land_data.to_excel('land_origin_data.xlsx', sheet_name='Sheet1', index=False)
except TypeError as e:
    print(f"遇到类型错误: {e}")
    print("请检查输入的数据是否与预期相符。")