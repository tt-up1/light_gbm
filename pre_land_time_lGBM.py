from datetime import datetime
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from get_land_data import land_data as ld  # 假设数据已经准备好
from get_land_speed_data import china_holidays

force_col_wise= True

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

def get_holiday_state(t):
    date = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")  # 转换为 datetime 类型
    if date in china_holidays:
        if "春节" in china_holidays[date] or "国庆" in china_holidays[date]:
            return "大长假"
        return "小长假"
    elif date.weekday() in [5, 6]:
        return "周末"
    else:
        return "工作日"

data=pd.DataFrame()
#处理顺序型
data["time_type"]=[get_time_state(t) for t in ld['time']]
data["cargo_type"]=ld["cargo_type"]
data["truck_state"]=ld["truck_level"]
data["road_state"]=ld["road"]
data["weather"]=ld["weather"]
data["wind"]=[int(w) for w in ld["wind"]]
data["buffer"]=ld["buffer"]
data["holiday_type"]=[get_holiday_state(t) for t in ld['time']]
# 定义所有有序分类变量的映射
mappings = {
    'time_type': {'白天':1,'夜间': 2,'早高峰': 3,'晚高峰': 4},
    'holiday_type':{"工作日": 1, "周末": 2, "小长假": 3, "大长假": 4},
    'cargo_type': {'D': 1, 'C': 2, 'B': 3,'A':4},
    'truck_state':{"优": 1, "良": 2, "中": 3, "差": 4, "极差": 5},
    'road_state':{"优": 1, "良": 2, "中": 3, "差": 4, "极差": 5},
    'weather':{"晴": 1, "阴": 2, "多云":3, "小雨": 4, "中雨": 5,
               "大雨": 6,  "暴雨":7,  "雨夹雪": 8, "雾": 9},
    'wind':{ 1:1,2:2,3:3,4:4,5:5,
       6:6,7:7,8:8,9:9,10:10},
    'buffer':{"有":1,"无":2}
}
'''
categorical_features = ['time_type', 'holiday_type', 'truck_state',
                        'road_state', 'weather', 'wind', 'buffer', 'cargo_type']
'''
# 逐列转换
for col, mapping in mappings.items():
    data[col] = data[col].map(mapping)

data["pre_land_time"]=ld["pre_land_time"]
data["port_rate"]=ld["port_rate"]
data["land_rate"]=ld["land_rate"]
# 将其他需要为数值型的字段转换为数值

scaler = MinMaxScaler(feature_range=(0.1, 4))
data[['truck_num','cargo_num', ]] = scaler.fit_transform(ld[['truck_num','cargo_num']])

# 使用分位数截断（保留98%数据）
q_low = data['pre_land_time'].quantile(0.01)
q_high = data['pre_land_time'].quantile(0.99)
data = data[(data['pre_land_time'] > q_low) & (data['pre_land_time'] < q_high)]

# 对出现次数<5的类别合并
for col in ['road_state', 'weather',]:
    counts = data[col].value_counts()
    rare_cats = counts[counts < 5].index
    data[col] = data[col].replace(rare_cats, -1)

#处理后的训练原始数据集，写入文件
data.to_excel('land_train_data.xlsx', sheet_name='Sheet1', index=False)


# 划分特征和目标
y1 = data['pre_land_time']
X1 = data.drop('pre_land_time', axis=1)

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# 再从训练集中划分 20% 验证集
X_train_part, X_valid_part, y_train_part, y_valid_part = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 2. 模型训练（基础版）
# 创建 LightGBM 数据集对象
train_dataset= lgb.Dataset(X_train, label=y_train)
valid_dataset = lgb.Dataset(X_valid_part, label=y_valid_part, reference=train_dataset)

# 对有序分类变量设置单调约束（以time_type为例）
monotone_constraints =[-1 if col in ['time_type','weather',
                         'road_state','holiday_type','truck_state',
                         'wind', 'buffer','cargo_type'] else 0 for col in X_train.columns.tolist()]


# 参数设置：回归任务，使用 rmse 评估指标
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.015,  # 降低学习率
    'num_leaves': 255,  # 限制叶子数，防止过拟合
    'max_depth': 12,  # 限制树的深度
    'min_data_in_leaf': 30,  # 增大叶子节点的最小样本数
    'lambda_l1': 2,  # 增加 L1 正则化
    'lambda_l2': 1.5,  # 增加 L2 正则化
    'feature_fraction': 0.7,  # 降低特征使用比例
    'bagging_fraction': 0.7,  # 采样 70% 数据
    'bagging_freq': 5,
    'min_split_gain': 0.01,
    'cat_smooth': 30,  # 改善分类变量处理
   # 'boosting_type': 'dart' , # 使用DART模式应对异常值
    'verbosity': -1,
    'monotone_constraints': monotone_constraints,  # 正确的列表格式
}


# 训练模型，使用早停（这里只使用训练集做验证，实际项目中可使用独立验证集）
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=2000,
    valid_sets=[valid_dataset,train_dataset],
    valid_names=["valid","train"],
    callbacks=[lgb.early_stopping(10),
            lgb.reset_parameter(learning_rate=[
             0.1 if i < 100 else 0.02 if i < 500 else 0.01
                for i in range(2000)])
               ]
)
te=pd.DataFrame({
    "time_type": 3,
    "cargo_type": 2,
    "truck_state": 2,
    "road_state": 2,
    "weather": 5,
    "wind": 9,
    "buffer": 1,
    "holiday_type": 3,
    #"pre_land_time": 359.0748706,
    "port_rate": 0.823004,
    "land_rate": 1.11448064,
    "truck_num": 2.7,
    "cargo_num": 1.982153122
},index=[0])
y_pre=model.predict(te)
print(y_pre)
#lgb.plot_tree(model,tree_index=0)
'''
# 3. 超参数优化（网格搜索）
# ======================
# 定义参数网格
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100,200],
    'max_depth': [3, 7]
}

# 使用 GridSearchCV 进行超参数优化（回归任务用负均方误差评分）
gbm = lgb.LGBMRegressor(verbose=-1)
grid = GridSearchCV(gbm, param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

print("最佳参数:", grid.best_params_)
best_model = grid.best_estimator_
'''
# ======================
# 4. 特征选择
# ======================
# 方法1：基于特征重要性
lgb.plot_importance(model)
importance = model.feature_importance()
feature_names = X1.columns
print("特征重要性:")
for name, score in zip(feature_names, importance):
    print(f"  {name}: {score}")
'''
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X1)
shap.summary_plot(shap_values, X1)
'''

'''
# 方法2：递归特征消除（RFE）— 使用 LGBMRegressor 进行特征选择
selector = RFE(estimator=lgb.LGBMRegressor())
selector.fit(X_train, y_train)
print("被选中的特征:", X1.columns[selector.support_].tolist())
'''

# 5. 模型评估
#训练集误差
from sklearn.model_selection import KFold
import lightgbm as lgb
import numpy as np

y_trpre = model.predict(X_train)
mse = mean_squared_error(y_train, y_trpre)
print("Train MSE:", mse)
print("Train RMSE:", np.sqrt(mse))

# 预测（使用最佳模型）
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)
print("Test RMSE:", np.sqrt(mse))

#如果相对 RMSE < 10%，说明误差较小，模型较好；如果 > 30%，说明误差较大。
relative_rmse = (np.sqrt(mse) / np.mean(y_test)) * 100
print(f"相对 RMSE: {relative_rmse:.2f}%")

#MAPE < 10%：非常好的模型,10% ≤ MAPE < 20%：可接受,MAPE > 20%：误差较大，可能需要优化
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")

# ======================
# 6. 预测新数据（示例代码）
# ======================
'''
new_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # ...其他特征
})
# 根据预处理步骤对 new_data 进行必要的转换
prediction = best_model.predict(new_data)
print("预测结果:", prediction)
'''

# ======================
# 7. 可视化结果
# ======================
# 绘制真实值与预测值的散点图，并添加 y=x 对角线
'''
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="预测结果")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="理想情况")
plt.xlabel("true value")
plt.ylabel("predict value")
plt.title("true and predict value")
plt.legend()
plt.show()
'''


'''
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, alpha=0.5, label="残差")
plt.axhline(0, color='red', linestyle='--', label="零误差线")
plt.xlabel("真实值")
plt.ylabel("真实值与预测值之差")
plt.title("残差图")
plt.legend()
plt.show()
'''


