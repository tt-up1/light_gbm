from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

force_col_wise= True
#调用生成数据，生成后注释掉
# preData.creatTrainData()
data=pd.read_excel("port_train_data.xlsx")
# 划分特征和目标
y1 = data['pre_port_time']
X1 = data.drop(['pre_port_time', 'batch_id'], axis=1)

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# 再从训练集中划分 20% 验证集
X_train_part, X_valid_part, y_train_part, y_valid_part = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

#查看y的分布
# # 假设y_train是你的目标变量
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)  # (行数, 列数, 当前子图索引)
sns.histplot(y_train, kde=True, bins=30)
plt.title('Distribution of Target Variable (y)')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
#plt.show()

# #查看xy的分布
# # 合并X和y，计算相关系数
data = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
corr = data.corr()
plt.subplot(1, 2, 2)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("特征与目标的相关性热力图")
plt.show()

# train_dataset= lgb.Dataset(X_train, label=y_train)
#valid_dataset = lgb.Dataset(X_valid_part, label=y_valid_part, reference=train_dataset)

# 无序类型（港口特定，无 sub_task_type/p/road_state/truck_level）
categorical_features = ['weather', 'cargo_type', 'holiday_type', 'wind', 'time_type', 'buffer']

train_dataset= lgb.Dataset(X_train, label=y_train,categorical_feature=categorical_features)
valid_dataset = lgb.Dataset(X_valid_part, label=y_valid_part, reference=train_dataset,categorical_feature=categorical_features)
#无序分类
monotone_constraints = [
    1 if col == 'cargo_num' else (  # cargo_num 设为单调递增
        -1 if col in ['port_rate', 'port_rate', 'equip_num'] else 0  # 其他三个特征递减
    )
    for col in X_train.columns.tolist()
]

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
    # 'monotone_constraints': monotone_constraints,  # 正确的列表格式
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
    "equip_level": 5,
    "weather": 1,
    "wind": 9,
    "buffer": 1,
    "holiday_type":1 ,
    #"pre_port_time": 8.352778007
   "port_rate":  0.06671808,
    "land_rate": 0.10216206,
    "equip_num": 1,
    "cargo_num": 3.767317839
},index=[0])


# 假设 model 是你的训练好的模型
y_pred_log = model.predict(te)  # 预测的是 log(1 + y)
print("模型预测值：",y_pred_log)
# 还原到原始尺度
y_pre = np.expm1(y_pred_log)      # y = exp(log(1 + y)) - 1
print("模型还原预测值：",y_pre)
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

# 5. 模型评估
#训练集误差
from sklearn.model_selection import KFold

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
