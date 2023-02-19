import pandas as pd
import warnings
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, EditedNearestNeighbours
from collections import Counter
from sklearn.model_selection import train_test_split
from pyecharts.charts import WordCloud,Bar, Pie,Grid,Page,TreeMap,Line
from pyecharts import options as opts
import math
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier,MLPRegressor #引入包

import os
from sklearn.tree import export_graphviz
import pybaobabdt
import pygraphviz
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity


def xiangguanxishu(X, Y):
    '''计算相关系数 '''
    XY = X * Y
    X2 = X ** 2
    Y2 = Y ** 2
    n = len(XY)
    numerator = n * XY.sum() - X.sum() * Y.sum()    # 分子
    denominator = math.sqrt(n * X2.sum() - X.sum() ** 2) * math.sqrt(n * Y2.sum() - Y.sum() ** 2)  # 分母

    if denominator == 0:
        return 'NaN'
    rhoXY = numerator / denominator
    return rhoXY


def open_file():
    data1 = pd.read_csv('set_a_data.csv')
    data2 = pd.read_csv('set_b_data.csv')
    data3 = pd.read_csv('set_b_data.csv')
    data = pd.concat([data1, data2,data3])
    data_list = data.dropna(thresh=9600,axis=1)
    print(data_list.isnull().sum())
    print(data_list.head())
    print(data_list)
    # print(data.isnull().any().items())
    for i in data_list.isnull().any().items():
        name = str(i[0])
        val  = str(i[1])
        if 'True'==val:
            data_list[name] = data_list[name].fillna(data_list[name].median())
    print(data_list.isnull().any())
    return data_list

def mintt():
    data = open_file()
    data_txt = data.drop(labels=['RecordID','lab'], axis=1)
    data_lab = data['lab']
    ttt = []
    for i in data_txt.columns:
        aa = xiangguanxishu(data_txt[i],data_lab)
        ttt.append([i,aa])
    for i in ttt:
        print(i)

    cc = ClusterCentroids()  # 经测试，三个降采样中这个评分最高
    X_resampled, y_resampled = cc.fit_resample(data_txt, data_lab.values.reshape([-1]))
    data_train_X, data_train_Y = X_resampled, y_resampled

    x_train_all, iris_x_test, y_train_all, iris_y_test = train_test_split(data_train_X, data_train_Y, random_state=7, test_size=0.2)
    iris_x_train, iris_x_yan, iris_y_train, iris_y_yan = train_test_split(x_train_all, y_train_all, random_state=11, test_size=0.2)
    print(iris_x_test)
    print(iris_x_train)
    print(iris_x_yan)

    from sklearn.preprocessing import MinMaxScaler

    min_max_scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    new_X_train = iris_x_train
    new_X_test = iris_x_test
    new_X_yan = iris_x_yan

    from sklearn.preprocessing import Normalizer
    normalizer = Normalizer(copy=True, norm='l2').fit(new_X_train)
    new_X_train = normalizer.transform(new_X_train)
    new_X_test = normalizer.transform(new_X_test)
    new_X_yan = normalizer.transform(new_X_yan)

    #随机森林模型
    from sklearn.ensemble import RandomForestClassifier

    clf1 = RandomForestClassifier(n_estimators=200)
    clf1.fit(new_X_train, iris_y_train)
    #生成随机森林可视化图
    # for idx, estimator in enumerate(clf1.estimators_):
    #     # 导出.dot文件
    #     export_graphviz(estimator,
    #                     out_file='tree{}.dot'.format(idx),
    #                     feature_names=data_txt.columns,
    #                     class_names='lba',
    #                     rounded=True,
    #                     proportion=False,
    #                     precision=2,
    #                     filled=True)
    #     # 转换为.png文件
    #     os.system('dot -Tpng tree{}.dot -o tree{}.png'.format(idx, idx))
    _y = clf1.predict(new_X_test)
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = iris_y_test.reshape([-1, ])
    y_pred = _y
    #评价指标F1,精确率和响应率
    print('评价指标F1:',f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None))
    zhenshi_s = 0
    zhenshi_c = 0
    yuce_s    = 0
    yuce_c    = 0
    dddd = []
    for i in y_pred:
        dddd.append(i)
        if i==0:
            yuce_c+=1
        else:
            yuce_s+=1
    print(dddd)
    zzzz = []
    for i in y_true:
        zzzz.append(i)
        if i==0:
            zhenshi_c+=1
        else:
            zhenshi_s+=1
    print(zzzz)
    print('精确率:',precision_score(y_true, y_pred))
    print('响应率:',recall_score(y_true, y_pred))

    import matplotlib.pyplot as plt  # 导入绘图包

    pre_y = clf1.predict_proba(new_X_yan)[:, 1]
    fpr_Nb, tpr_Nb, _ = roc_curve(iris_y_yan, pre_y)
    aucval = auc(fpr_Nb, tpr_Nb)                  # 计算auc的取值
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_Nb, tpr_Nb, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林AUC曲线")
    plt.text(0.15, 0.9, "AUC = " + str(round(aucval, 4)))
    plt.show()

    dx = []
    dy = []
    for i in data_txt.columns:
        dx.append(i)
    for i in clf1.feature_importances_:
        dy.append(i)
    print(dx)
    print(dy)
    bar1 = (
        Bar(init_opts=opts.InitOpts(width="1700px",
                                height="750px",))
            .add_xaxis(dx)
            .add_yaxis(" ", dy)
            .set_global_opts(
            title_opts=opts.TitleOpts(title="特征重要性"),
            tooltip_opts=opts.TooltipOpts(is_show=True))
    )

    (
        Page(page_title="all")
            .add(bar1)
            .render("all.html")
    )



    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50, 50), random_state=1)  # 3个隐藏层
    # # 拟合—模型训练
    # print(iris_x_train)
    # clf.fit(iris_x_train, iris_y_train)
    #
    # iris_y_predict = clf.predict(iris_x_test)
    # score = clf.score(iris_x_test, iris_y_test, sample_weight=None)
    # print('iris_y_predict=')
    # print(iris_y_predict)
    # ddd = []
    # for i in iris_y_predict:
    #     ddd.append(i)
    # print(ddd)
    # print('iris_y_test=')
    # print(iris_y_test)
    # print('Accuracy:', score)



if __name__ == '__main__':
    mintt()

