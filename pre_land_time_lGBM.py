from datetime import datetime
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import preData

force_col_wise= True

# preData.creatTrainData()
data=preData.getTrainData()
'''
categorical_features = ['time_type', 'holiday_type', 'truck_state',
                        'road_state', 'weather', 'wind', 'buffer', 'cargo_type']
'''
# 划分特征和目标
y1 = data['pre_land_time']
X1 = data.drop('pre_land_time', axis=1)

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# 再从训练集中划分 20% 验证集
X_train_part, X_valid_part, y_train_part, y_valid_part = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

#查看y的分布

import matplotlib.pyplot as plt
import seaborn as sns

# # 假设y_train是你的目标变量
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)  # (行数, 列数, 当前子图索引)
sns.histplot(y_train, kde=True, bins=30)
plt.title('Distribution of Target Variable (y)')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
#plt.show()

# #查看xy的分布
# import pandas as pd
#
# # 合并X和y，计算相关系数
data = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
corr = data.corr()
# plt.subplot(1, 2, 2)
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
# plt.title("特征与目标的相关性热力图")
#plt.show()

#无序类型
categorical_features = ['weather', 'cargo_type', 'holiday_type','wind','time_type','buffer','road_state']

train_dataset= lgb.Dataset(X_train, label=y_train,categorical_feature=categorical_features)
valid_dataset = lgb.Dataset(X_valid_part, label=y_valid_part, reference=train_dataset,categorical_feature=categorical_features)
# 无序分类
monotone_constraints = [
    1 if col == 'cargo_num' else (  # cargo_num 设为单调递增
        -1 if col in ['port_rate', 'land_rate', 'truck_num'] else 0  # 其他三个特征递减
    )
    for col in X_train.columns.tolist()
]
# 2. 模型训练（基础版）
# 创建 LightGBM 数据集对象
# train_dataset= lgb.Dataset(X_train, label=y_train)
# valid_dataset = lgb.Dataset(X_valid_part, label=y_valid_part, reference=train_dataset)

# # 对有序分类变量设置单调约束（以time_type为例）
# monotone_constraints =[-1 if col in ['time_type','weather',
#                          'road_state','holiday_type','truck_state',
#                          'wind', 'buffer','cargo_type'] else 0 for col in X_train.columns.tolist()]


# 参数设置：回归任务，使用 rmse 评估指标
params = {
    'objective': 'regression',
    'metric': 'rmse',
    # 'learning_rate': 0.015,  # 降低学习率
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

eval_result = {}

# 训练模型，使用早停（这里只使用训练集做验证，实际项目中可使用独立验证集）
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=20000,
    valid_sets=[valid_dataset,train_dataset],
    valid_names=["valid","train"],
    callbacks=[lgb.early_stopping(100),
               lgb.record_evaluation(eval_result),
            lgb.reset_parameter(learning_rate=[
             0.03 if i < 300 else 0.01 if i < 1000 else 0.003
                for i in range(20000)])
               ]
)
te=pd.DataFrame({
    "time_type": 1,
    "cargo_type": 1,
    "truck_state": 1,
    "road_state": 1,
    "weather": 1,
    "wind": 2,
    "buffer": 1,
    "holiday_type":1 ,
    #"pre_land_time": 359.0748706,
   "port_rate":  0.76212576,
    "land_rate": 0.90502434,
    "truck_num": 2,
    "cargo_num": 3.164788205
},index=[0])
# 假设 model 是你的训练好的模型
y_pred_log = model.predict(te)  # 预测的是 log(1 + y)

# 还原到原始尺度
y_pre = np.expm1(y_pred_log)      # y = exp(log(1 + y)) - 1
print("模型预测值：",y_pre)
# y_model,yreal=toCreateData(te,mappings,scaler)
# print("反推值factor：",y_model)
# print("直接计算值：",yreal)
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


#loss可视化

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(eval_result['train']['rmse'], label='Train RMSE')
plt.plot(eval_result['valid']['rmse'], label='Valid RMSE')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('LightGBM Training Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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


