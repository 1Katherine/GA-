import datetime
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
import xgboost
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from sko.GA import GA,RCGA



# 主机上运行的代码


'''
    根据名称构建模型
'''
def build_model(name):
    if name.lower() == "lgb":
        model = lgb.LGBMRegressor()
    elif name.lower() == "gdbt":
        model = GradientBoostingRegressor()
    else:
        model = RandomForestRegressor()
    return model


'''
    不重新建模，使用已经构建好的模型
'''
def build_training_model(name):
    import warnings
    if name.lower() == "lgb":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load('./files44/lgb/lgb.pkl')
    elif name.lower() == "gdbt":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load('./files44/gdbt/gdbt.pkl')
    elif name.lower() == "rf":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load('./files44/rf/rf.pkl')
    elif name.lower() == 'xgb':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load('./files44/xgb/xgb.pkl')
    elif name.lower() == 'ada':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load('./files44/ada/ada.pkl')
    else:
        raise Exception("[!] There is no option ")
    return model


'''
    黑盒模型，传入参数，计算target（根据模型预测参数的执行时间）
'''
def black_box_function(params):
    model = build_training_model(name)
    runtime = model.predict(np.matrix([params]))[0]
    return runtime


# --------------------- 生成 gan-rs 初始种群 start -------------------
initpoint_path = './wordcount-100G-GAN.csv'
initsamples_df = pd.read_csv(initpoint_path)

def ganrs_samples_all():
    # 初始样本
    initsamples = initsamples_df[vital_params_list].to_numpy()
    return initsamples

def ganrs_samples_odd():
    initsamples_odd = initsamples_df[initsamples_df.index % 2 == 0]
    initsamples = initsamples_odd[vital_params_list].to_numpy()
    return initsamples

def ganrs_samples_even():
    initsamples_even = initsamples_df[initsamples_df.index % 2 == 1]
    initsamples = initsamples_even[vital_params_list].to_numpy()
    return initsamples

def get_ganrs_samples(kind):
    if kind == 'all':
        samples = ganrs_samples_all()
    if kind == 'odd':
        samples = ganrs_samples_odd()
    if kind == 'even':
        samples = ganrs_samples_even()
    return samples
# --------------------- 生成 gan-rs 初始种群 end -------------------

if __name__ == '__main__':
    name = 'rf'
    # 重要参数
    vital_params_path = './files44/' + name + "/selected_parameters.txt"
    # 维护的参数-范围表
    conf_range_table = "Spark_conf_range_wordcount.xlsx"

    '''
        读取模型输出的重要参数
    '''
    vital_params = pd.read_csv(vital_params_path)
    # 参数范围和精度，从参数范围表里面获取

    # 参数范围表
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    # SparkConf 列存放的是配置参数的名称
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    # 转化后的字典形式：{index(值): {column(列名): value(值)}}
    # {'spark.broadcast.blockSize': {'Range': '32-64m', 'min': 32.0, 'max': 64.0, 'pre': 1.0, 'unit': 'm'}
    confDict = sparkConfRangeDf.to_dict('index')

    '''
        获取格式
    '''
    # 遍历训练数据中的参数，读取其对应的参数空间
    confLb = []  # 参数空间上界
    confUb = []  # 参数空间下界
    precisions = []  # 参数精度
    for conf in vital_params['vital_params']:
        if conf in confDict:
            confLb.append(confDict[conf]['min'])
            confUb.append(confDict[conf]['max'])
            precisions.append(confDict[conf]['pre'])
        else:
            print(conf, '-----参数没有维护: ', '-----')
    # ------------新增代码 start--------------
    # 重要参数的特征值
    parameters_features_path = './files44/' + name + "/parameters_features.txt"
    parameters_features_file = []
    parameters_features = []
    vital_params_list = []
    for line in open(parameters_features_path, encoding='gb18030'):
        parameters_features_file.append(line.strip())
    # 取出重要参数
    parameters_features_file = parameters_features_file[-len(vital_params):]
    for conf in vital_params['vital_params']:
        for para in parameters_features_file:
            if conf in para:
                # 重要参数列表
                vital_params_list.append(para.split(':')[0])
                # 重要参数的特征重要值列表
                parameters_features.append(float(para.split(':')[1]))

    # ------------新增代码 end--------------
    '''
        开始遗传算法
    '''
    startTime = datetime.datetime.now()
    # 确定其他参数
    fitFunc = black_box_function  # 适应度函数
    nDim = len(vital_params)  # 参数个数
    sample_kind = 'even'
    initsamples = get_ganrs_samples(kind=sample_kind)
    sizePop = len(initsamples)  # 种群数量
    maxIter = 10  # 迭代次数
    # probMut = 0.01  # 变异概率
    probMut = np.array(parameters_features)

    ga = GA(initsamples=initsamples,func=fitFunc,  n_dim=nDim, size_pop=sizePop, max_iter=maxIter, prob_mut=probMut, lb=confLb, ub=confUb,
            precision=precisions)
    best_x, best_y = ga.run()
    endTime = datetime.datetime.now()
    searchDuration = (endTime - startTime).seconds
    generation_best_file = open('./binary_result/generation_best.txt', 'a')
    print('\nbinary GA ' + name + ' ,sizePop=' + str(sizePop) + ' ,maxIter=' + str(maxIter), file=generation_best_file)
    print(vital_params_list, file=generation_best_file)
    print('best_x : ' + str(best_x), file=generation_best_file)
    print('best_y : ' + str(best_y), file=generation_best_file)
    generation_best = []
    print('generation_best \n', file=generation_best_file)
    for row, x in enumerate(ga.generation_best_X):
        temp = dict(zip(vital_params_list, x))
        temp['runtime'] = ga.generation_best_Y[row]
        generation_best.append(temp)
        print(str(temp), file=generation_best_file)
    df = pd.DataFrame(generation_best)
    df.to_csv('./binary_result/generation_best '+ name + ' sizePop=' + str(sizePop) + ' maxIter=' + str(maxIter) + '.csv', index=False)

    # %% Plot the binary_result
    import pandas as pd
    import matplotlib.pyplot as plt

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    ax[0].set_title('binary GA ' + name)
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig('./binary_result/pic/RCGA ' + name + ' sizePop=' + str(sizePop) + ' maxIter=' + str(maxIter) + '.png')
    plt.show()




