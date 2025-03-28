import holidays
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from final_product_data import land_rate_data  # 假设数据已经准备好


force_col_wise= True
# ======================
# 1. 数据准备与预处理
# ======================
# 这里假设 land_rate_data 已包含预测目标 'pre_land_time' 和其他特征
data1 = land_rate_data.copy()
#print(max(data1["road_factor"]),min(data1["road_factor"]))
# 示例代码（以陆运数据为例）
data_land = data1.drop(columns=[
    'time_factor', 'cargo_factor', 'truck_factors','truck_level','road',
     #'weather_factor_land', 'wind_factors',
    'factor','weather',
    'land_rate',  'buffer_factor',
])

#实验性删除缓冲和相关速率的特征，模型过于依赖直接特征，忽视间接特征
data_land=data_land.drop(columns=['port_rate','buffer_amout'])

#print(data_land)
# 1. 处理时间字段：将 time 转换为 datetime，并提取日期时间特征
data_land['time'] = pd.to_datetime(data_land['time'])
#data_land['year']   = data_land['time'].dt.year
data_land['month']  = data_land['time'].dt.month
data_land['day']    = data_land['time'].dt.day
data_land['hour']   = data_land['time'].dt.hour
data_land['minute'] = data_land['time'].dt.minute
data_land['day_of_week'] = data_land['time'].dt.dayofweek
china_holidays = holidays.China(years=[2024, 2025])
data_land['is_holiday'] = data_land['time'].dt.date.apply(lambda x: x in china_holidays).astype(int)
data_land['is_weekend'] = (data_land['day_of_week'] >= 5).astype(int)

data_land=data_land.drop(columns=["time"])
# 构造时间周期性编码（捕捉早晚高峰,时间因子）
data_land['hour_sin'] = np.sin(2 * np.pi * data_land['hour'] / 24)
data_land['hour_cos'] = np.cos(2 * np.pi * data_land['hour'] / 24)

时间四个类
天气
风力

# 将其他需要为数值型的字段转换为数值
numeric_cols_land = [ 'truck_num','winds','wind_factors','weather_factor_land','month','day','hour','minute', 'day_of_week', 'cargo_num', 'pre_land_time']
for col in numeric_cols_land:
    if col in data_land.columns:
        data_land[col] = pd.to_numeric(data_land[col], errors='coerce')

# 2.对车辆特征处理,标准化
from sklearn.preprocessing import StandardScaler
data_land['truck_num_scaled'] = StandardScaler().fit_transform(data_land[['truck_num']])
# 构造性能-数量交互特征
data_land['truck_total_perf'] = data_land['truck_num'] * data_land['truck_perf']

车辆有序的

车辆数量


#3.货物类型有序

data_land = pd.get_dummies(data_land, columns=["cargo_type"])


#处理后的训练原始数据集，写入文件
data_land.to_excel('land_train_data.xlsx', sheet_name='Sheet1', index=False)


#print(data_land.iloc[:5,:])
#print("=== land_rate_data 的数据类型 ===")
#print(data_land.dtypes)

# 划分特征和目标
y1 = data_land['pre_land_time']
X1 = data_land.drop('pre_land_time', axis=1)


# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)


# 再从训练集中划分 20% 验证集
X_train_part, X_valid_part, y_train_part, y_valid_part = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
# ======================
# 2. 模型训练（基础版）
# ======================
# 创建 LightGBM 数据集对象
train_dataset= lgb.Dataset(X_train, label=y_train)
valid_dataset = lgb.Dataset(X_valid_part, label=y_valid_part, reference=train_dataset)

# 参数设置：回归任务，使用 rmse 评估指标
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 10,          # 减少深度，防止过拟合
    'n_estimators': 200,     # 增加树数量，配合低学习率
    'num_leaves': 20,        # 允许更灵活的分裂
    'reg_alpha': 0.1,        # L1 正则化
    'reg_lambda': 0.1  ,      # L2 正则化
    'feature_fraction': 0.7,  # 每次分裂随机选择70%特征，强制探索 truck_level
    'min_data_in_leaf': 5,
    'verbose': -1  # 不显示日志
}

# 训练模型，使用早停（这里只使用训练集做验证，实际项目中可使用独立验证集）
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=500,
    valid_sets=[train_dataset,valid_dataset],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(10)]
)

# ======================
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
importance = model.feature_importance()
feature_names = X1.columns
print("特征重要性:")
for name, score in zip(feature_names, importance):
    print(f"  {name}: {score}")

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X1)
shap.summary_plot(shap_values, X1)

# 方法2：递归特征消除（RFE）— 使用 LGBMRegressor 进行特征选择
selector = RFE(estimator=lgb.LGBMRegressor())
selector.fit(X_train, y_train)
print("被选中的特征:", X1.columns[selector.support_].tolist())

# 计算特征的方差
feature_variance = X_train.var()
print("特征方差:\n", feature_variance)

low_variance_features = feature_variance[feature_variance < 0.05].index
print("低方差特征的唯一值个数:")
print(X_train[low_variance_features].nunique())



# ======================
# 5. 模型评估
# ======================
#训练集误差

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


