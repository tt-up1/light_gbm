import numpy as np
from get_port_speed_data import port_rate_data as port_data
from get_land_speed_data import land_rate_data as land_data

def predic_port_time(land_data,port_data):
    #港口货物量应该是陆运加缓冲
    cargo_mount=land_data["cargo_num"]+port_data["buffer_amout"]
    port_data["cargo_num"]=cargo_mount
    port_time=(cargo_mount/port_data["port_rate"])*land_data["land_rate"]
    port_data["pre_port_time"]=port_time

predic_port_time(land_data,port_data)

try:
    port_data.to_excel('port_origin_data.xlsx', sheet_name='Sheet1', index=False)
except TypeError as e:
    print(f"遇到类型错误: {e}")
    print("请检查输入的数据是否与预期相符。")

