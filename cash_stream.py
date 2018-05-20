#coding:utf-8
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR                                                   # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor      # 集成算法
import xgboost
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,cross_validate
# 平稳性检验
from statsmodels.tsa.stattools import adfuller as ADF
# 白噪声检测
from statsmodels.stats.diagnostic import acorr_ljungbox
# ARIMA模型
from statsmodels.tsa.arima_model import ARIMA



# 先计算大钱的输入输出，确定大的方向，就能做到八九不离十 ,收入/支出大钱时间上（固定时间差）有没有规律
# 根据自己的最好成绩于排行榜最好成绩的差距，平均预测结果每个相差1116,总绝对差距为3348000
# train中有3638790条余额记录，test中有180000条余额记录
def show_record():
	"""根据生成的图来辅助验证预测结果是否合理"""
    filepath = "./train/test_input.txt"
    data = pd.read_csv(filepath, nrows=180000, names=['id', 'type', 'date', 'bal'])
    print(data.head())

    for name, groups in data.groupby('id'):
        print groups
        if name>3000 and name<4000:
            plt.figure(name)
            plt.plot(groups['date'], groups['bal'])
            plt.xlim(0, 90)
            plt.title(name)
            print name
            plt.savefig('./60dayrecord/%i.png'%name)

# show_record()


"""
说明：
下面的程序为目前最好成绩的版本，相比取测试集最后一天的余额的baseline好了27个点，
主要是统计每个客户发工资、转账等记录的周期性，然后看在预测时间范围内是否有发工资、转账等行为，
排除掉1,2月份发年终奖（奖金）等工资，剩下的工资记录平均值作为可能的正常工资金额，发工资时间取剩下工资记录的最后一天，
转账给他人和信用卡等特征主要使用平均值，然后根据预测发工资的时间差，确定应该减少多少金额，
最后在baseline的基础上，更具预测结果修改记录(用上面的程序生成的图辅助确认)
"""
def static_trade_record():
    """
    在test中找工资、理财等特征的时间差与规律
    :return:
    """
    filepath = "./train/test_input.txt"
    bal_data = pd.read_csv(filepath, nrows=180000, names=['id', 'type', 'date', 'bal'])
    trade_data = pd.read_csv(filepath, names=['user_id', 'type', 'card_id', 'date',
                                              'abstract', 'note', 'class', 'money'], skiprows=180000)

    # is_null = trade_data['money'].isnull()
    # trade_nan = trade_data[is_null]
    # trade_nan['money'] = trade_nan['class']
    # trade_nan['class'] = trade_nan['note']

    print(trade_data.shape)
    money = []
    sal = []
    all_input = []
    all_output = []
    all_credit = []
    all_fare_date = []
    all_credit_date = []
    all_id = []
    # 工资
    date_fare = []
    fare_temp = []
    # 转入
    date_to = []
    to_temp = []
    # 转出
    date_out = []
    out_temp = []
    # 还招行信用卡
    date_credit = []
    credit_temp = []

    for name, groups in trade_data.groupby('user_id'):

        # print((groups['abstract'].iloc[2]))
        for i in range(len(groups)):
            if np.isnan(groups.iloc[i, -1]):
                money.append(groups.iloc[i, -2])
            else:
                money.append(groups.iloc[i, -1])

            # print ("money is:", money[i])

            if (str(groups['abstract'].iloc[i]).__contains__('工资')
                    or str(groups['note'].iloc[i]).__contains__('工资福利')
                    or str(groups['class'].iloc[i]).__contains__('工资福利')):
                if money[i] >= 1000 and groups['date'].iloc[i] > 10:           # 小于1000就不认为时每月固定工资了
                    if groups['date'].iloc[i] > 45 or groups['date'].iloc[i] < 40:
                        date_fare.append(groups['date'].iloc[i])
                        fare_temp.append(money[i])
                        print "fare:", groups['date'].iloc[i], money[i]

            if (str(groups['abstract'].iloc[i]).__contains__('他人转入')
                    or str(groups['note'].iloc[i]).__contains__('他人转入')
                    or str(groups['class'].iloc[i]).__contains__('他人转入')):
                if money[i] >= 1000:           # 小于1000就不管了
                    date_to.append(groups['date'].iloc[i])
                    to_temp.append(money[i])
                    # print "_to_:", groups['date'].iloc[i], money[i]

            if (str(groups['note'].iloc[i]).__contains__('转账给他人')
                    or str(groups['class'].iloc[i]).__contains__('转账给他人')):
                if money[i] <= -1000 and len(date_fare)>0 and (
                        groups['date'].iloc[i]-date_fare[-1] < 30) and(
                            fare_temp[-1] + money[i] > 0):           # 转账要低于工资1000
                    date_out.append(groups['date'].iloc[i])
                    out_temp.append(money[i])
                    print "out_:", groups['date'].iloc[i], money[i]

            if (str(groups['note'].iloc[i]).__contains__('还招行信用卡')
                    or str(groups['class'].iloc[i]).__contains__('还招行信用卡')):
                if money[i] <= -1000 and len(date_fare) > 0 and (
                        groups['date'].iloc[i] - date_fare[-1] < 7) and (
                            fare_temp[-1] + money[i] > 1000):        # 还卡额要低于工资1000
                    date_credit.append(groups['date'].iloc[i])
                    credit_temp.append(money[i])
                    print "cred:", groups['date'].iloc[i], money[i]
        # plt.figure(name)
        # plt.title("%i"%name)
        # plt.scatter(date_fare, fare_temp)
        # plt.scatter(date_to, to_temp)
        # plt.plot(date_out, out_temp,'r.')
        # plt.savefig("./compare/%i"%name)
        all_id.append(name)
        # 对fare_temp结果去掉最大值和最小值取平均
        if len(fare_temp)>2:
            fare_temp.remove(max(fare_temp))
            fare_temp.remove(min(fare_temp))

        if len(fare_temp) != 0:
            print fare_temp[-1]
            all_input.append(fare_temp[-1])        # 如果有‘工资’关键词，则append最后一个工资
            all_fare_date.append(date_fare[-1])
        else:
            all_input.append(0)                    # 如果没有‘工资’关键词，则append列表[0]
            all_fare_date.append(0)

        # 对7天内的output结果去掉最大值和最小值取平均得到平均每天的减少
        if len(out_temp) > 2:
            out_temp.remove(max(out_temp))
            out_temp.remove(min(out_temp))

        if len(out_temp) != 0:
            if date_out[-1]-date_fare[-1] <=0:
                if (date_out[-1]-date_fare[-1])%30 == 0:
                    out_mean = np.round(out_temp[-1]/((date_out[-1]-date_fare[-1])%30+1), 2)
                else:
                    out_mean = np.round(out_temp[-1] / ((date_out[-1]-date_fare[-1])%30), 2)
            else:
                out_mean = np.round(out_temp[-1]/(date_out[-1]-date_fare[-1] + 1), 2)
            print out_mean
            all_output.append(out_mean)  # 如果有‘工资’关键词，则append最后一个工资

        else:
            all_output.append(0)  # 如果没有‘工资’关键词，则append列表[0]

        # 对7天内的credit结果去掉最大值和最小值取平均得到平均每天的减少
        if (len(credit_temp) > 2):
            credit_temp.remove(max(credit_temp))
            credit_temp.remove(min(credit_temp))

        if len(credit_temp) != 0:

            if date_credit[-1]-date_fare[-1]<=0:
                if (date_credit[-1] - date_fare[-1]) % 30 == 0:         # 避免刚好相差30天出现的问题
                    credit_mean = np.round(credit_temp[-1]/((date_credit[-1]-date_fare[-1])%30+1), 2)
                else:
                    credit_mean = np.round(credit_temp[-1] / ((date_credit[-1] - date_fare[-1])%30), 2)
            else:
                credit_mean = np.round(credit_temp[-1]/(date_credit[-1]-date_fare[-1] + 1), 2)
            print credit_mean
            all_credit.append(credit_mean)  # 如果有‘工资’关键词，则append最后一个工资
            all_credit_date.append(date_fare[-1])

        else:
            all_credit.append(0)  # 如果没有‘工资’关键词，则append列表[0]

        money = []
        # 每个客户工资部分清空
        date_fare = []
        fare_temp = []
        # 每个客户他人转入部分清空
        date_to = []
        to_temp = []
        # 每个客户转账给他人部分清空
        date_out = []
        out_temp = []
        # 还招行信用卡清空
        date_credit = []
        credit_temp = []

        print("<===========正在生成用户%i：工资、日期列表和转账记录===============>\n"%name)
    print np.shape(all_fare_date)
    print np.shape(all_id)
    print np.shape(all_input)
    print np.shape(all_output)
    print np.shape(all_fare_date)
    dataframe = {'id': all_id, 'fare': all_input, 'fare_date': all_fare_date, 'out': all_output,'credit':all_credit}
    new = pd.DataFrame(dataframe)
    print new.head()

    predict = []
    date = []

    last_data = []
    for name, groups in bal_data.groupby('id'):
        last_data.append(groups['bal'].iloc[-1])
        date.append(groups['date'].iloc[-1] + 7)
    for i in range(3000):
        predict.append(last_data[i])

    #
    # print(np.shape(predict))
    data_copy = bal_data.__deepcopy__()
    data_copy = data_copy.drop_duplicates(['id'])
    # print(data_copy.head())
    data_copy['date'] = date
    data_copy['bal'] = predict[:]

    rs = pd.merge(data_copy, new, on='id', how='left')

    end_predict = []
    for i in range(3000):
        if date[i] - rs['fare_date'].iloc[i]-30 >= 0:               # 在发工资的可能时间范围
            print i, rs['id'].iloc[i], rs['fare'].iloc[i], rs['out'].iloc[i], rs['credit'].iloc[i]
            # print np.shape(rs['out'])
            result_temp = last_data[i] + rs['fare'].iloc[i]\
                               + rs['credit'].iloc[i]*((date[i] - rs['fare_date'].iloc[i])%30)\
                               + (rs['out'].iloc[i])*((date[i] - rs['fare_date'].iloc[i])%30)
            if result_temp < 0:
                result_temp = 0
            end_predict.append(result_temp)

        else:
            end_predict.append(last_data[i])

    end_data_copy = bal_data.__deepcopy__()
    end_data_copy = end_data_copy.drop_duplicates(['id'])
    # print(data_copy.head())
    end_data_copy['date'] = date
    end_data_copy['bal'] = end_predict[:]
    print(end_data_copy.head())
    print len(end_data_copy)
    end_data_copy.to_csv('./test_output.txt', index=False, header=False)

# static_trade_record()


"""
说明：
该程序是可能会有更好结果的版本，但是苦于机器原因，训练时间比较长
"""
def last_try():
	"""
	# 对差分数据做典型回归模型和直接取最后一天数据
	# 工资、转入、转出、信用卡、理财五个特征一起进行xgboost回归，回归结果为第67天的余额与第一天的余额差
	# 预测结果为余额差与第一天的余额相加
	"""
    filepath = "./train/train.txt"
    filepath1 = "./train/test_input.txt"
    train_data = pd.read_csv(filepath, nrows=3638790, names=['id', 'type', 'date', 'bal'])
    data_test = pd.read_csv(filepath1, nrows=180000, names=['id', 'type', 'date', 'bal'])
    train_trade_data = pd.read_csv(filepath, names=['user_id', 'type', 'card_id', 'date',
                                              'abstract', 'note', 'class', 'money'], skiprows=3638790)
    test_trade_data = pd.read_csv(filepath1, names=['user_id', 'type', 'card_id', 'date',
                                              'abstract', 'note', 'class', 'money'], skiprows=180000)
    print(train_data.head())
    print(data_test.head())
    xgb = xgboost.XGBRegressor(max_depth=10, min_child_weight=5)
    date = []
    train_set = []
    train_label = []
    predict = []
    test_set = []
    test_label = []
    temp_client = []

    money = []
    sal = []
    all_input = []
    all_output = []
    all_credit = []
    all_fare_date = []
    all_credit_date = []
    all_id = []
    # 工资
    date_fare = []
    fare_temp = []
    # 转入
    date_to = []
    to_temp = []
    # 转出
    date_out = []
    out_temp = []
    # 还招行信用卡
    date_credit = []
    credit_temp = []
    # 第三方理财
    date_licai = []
    licai_temp = []



    for name, groups in train_trade_data.groupby('user_id'):
        # print((groups['abstract'].iloc[2]))
        for i in range(len(groups)):                # 每一条记录的金额
            if np.isnan(groups.iloc[i, -1]):
                money.append(groups.iloc[i, -2])
            else:
                money.append(groups.iloc[i, -1])

            # print ("money is:", money[i])

            if (str(groups['abstract'].iloc[i]).__contains__('工资')
                    or str(groups['note'].iloc[i]).__contains__('工资福利')
                    or str(groups['class'].iloc[i]).__contains__('工资福利')):
                if groups['date'].iloc[i] > 45 or groups['date'].iloc[i] < 40:
                    date_fare.append(groups['date'].iloc[i])
                    fare_temp.append(money[i])
                    # print "fare:", groups['date'].iloc[i], money[i]

            if (str(groups['abstract'].iloc[i]).__contains__('他人转入')
                    or str(groups['note'].iloc[i]).__contains__('他人转入')
                    or str(groups['class'].iloc[i]).__contains__('他人转入')):
                date_to.append(groups['date'].iloc[i])
                to_temp.append(money[i])
                # print "_to_:", groups['date'].iloc[i], money[i]

            if (str(groups['note'].iloc[i]).__contains__('转账给他人')
                    or str(groups['class'].iloc[i]).__contains__('转账给他人')):
                date_out.append(groups['date'].iloc[i])
                out_temp.append(money[i])
                # print "out_:", groups['date'].iloc[i], money[i]

            if (str(groups['note'].iloc[i]).__contains__('还招行信用卡')
                    or str(groups['class'].iloc[i]).__contains__('还招行信用卡')):
                date_credit.append(groups['date'].iloc[i])
                credit_temp.append(money[i])
                # print "cred:", groups['date'].iloc[i], money[i]

            if (str(groups['note'].iloc[i]).__contains__('第三方理财')
                    or str(groups['class'].iloc[i]).__contains__('第三方理财')):
                date_licai.append(groups['date'].iloc[i])
                licai_temp.append(money[i])
                # print "licai:", groups['date'].iloc[i], money[i]

        # 对工资处理
        data1 = pd.DataFrame({'date': date_fare, 'value_fare': fare_temp})
        data1 = data1.groupby('date').sum()
        data1['date'] = data1.index
        data1.reset_index(drop=True)
        # 对转入处理
        data2 = pd.DataFrame({'date': date_to, 'value_to': to_temp})
        data2 = data2.groupby('date').sum()
        data2['date'] = data2.index
        data2.reset_index(drop=True)
        # print data1, data2
        # 对转出处理
        data3 = pd.DataFrame({'date': date_out, 'value_out': out_temp})
        data3 = data3.groupby('date').sum()
        data3['date'] = data3.index
        data3.reset_index(drop=True)
        # 对信用卡处理
        data4 = pd.DataFrame({'date': date_credit, 'value_credit': credit_temp})
        data4 = data4.groupby('date').sum()
        data4['date'] = data4.index
        data4.reset_index(drop=True)
        # 对理财处理
        data5 = pd.DataFrame({'date': date_licai, 'value_licai': licai_temp})
        data5 = data5.groupby('date').sum()
        data5['date'] = data5.index
        data5.reset_index(drop=True)

        rs = pd.merge(data1, data2, on=['date'], how='outer')
        rs = pd.merge(rs, data3, on=['date'], how='outer')
        rs = pd.merge(rs, data4, on=['date'], how='outer')
        rs = pd.merge(rs, data5, on=['date'], how='outer')

        bal_rs = train_data.loc[train_data['id'] == name]
        # print bal_rs
        bal_rs_temp = pd.merge(rs, bal_rs, on='date', how='outer')
        end_data = bal_rs_temp.reindex(columns=['id', 'date', 'type', 'value_fare', 'value_to',
                                           'value_out', 'value_credit', 'value_licai', 'bal'])
        bal_diff = end_data['bal'].diff(1).dropna()
        end_data['bal'] = bal_diff
        end_data = end_data.fillna(0)
        # print(end_data)

        money = []
        # 每个客户工资部分清空
        date_fare = []
        fare_temp = []
        # 每个客户他人转入部分清空
        date_to = []
        to_temp = []
        # 每个客户转账给他人部分清空
        date_out = []
        out_temp = []
        # 还招行信用卡清空
        date_credit = []
        credit_temp = []
        # 第三方理财清空
        date_licai = []
        licai_temp = []

        # 训练差分样本
        for i in range(23):
            # 计算差分
            # a = groups['bal'].iloc[i:i + 60].diff(1).dropna()
            # print a
            # print end_data.iloc[:, 3:]
            train_set.append(((end_data.iloc[:, 3:]).iloc[i:i+60, :]))
            # print "train set is:\n",train_set
            train_label.append(bal_rs['bal'].iloc[i+67] - bal_rs['bal'].iloc[i])
            # print "train label is:\n", train_label

        print("<======================%i is done===================>" % name)
    print np.shape(train_set), np.shape(train_label)
    # 交叉验证

    # param_grid = {"max_depth": range(5,10,2)}
    # grid = GridSearchCV(xgb, param_grid,scoring='neg_mean_absolute_error', cv=2, verbose=1)
    # grid.fit(train_set, train_label)

    # print grid.best_params_
    # print grid.best_score_

    xgb.fit(np.array(train_set), np.reshape(train_label, (-1, 1)))
    train_predict = xgb.predict(train_set)

    #
    print "train's mae is:", metrics.mean_absolute_error(train_label, train_predict)
    print("<======================training is done===================>")
    for name, groups in data_test.groupby('user_id'):
        for i in range(len(groups)):  # 每一条记录的金额
            if np.isnan(groups.iloc[i, -1]):
                money.append(groups.iloc[i, -2])
            else:
                money.append(groups.iloc[i, -1])

            # print ("money is:", money[i])

            if (str(groups['abstract'].iloc[i]).__contains__('工资')
                or str(groups['note'].iloc[i]).__contains__('工资福利')
                or str(groups['class'].iloc[i]).__contains__('工资福利')):
                if groups['date'].iloc[i] > 45 or groups['date'].iloc[i] < 40:
                    date_fare.append(groups['date'].iloc[i])
                    fare_temp.append(money[i])
                    # print "fare:", groups['date'].iloc[i], money[i]

            if (str(groups['abstract'].iloc[i]).__contains__('他人转入')
                    or str(groups['note'].iloc[i]).__contains__('他人转入')
                    or str(groups['class'].iloc[i]).__contains__('他人转入')):
                date_to.append(groups['date'].iloc[i])
                to_temp.append(money[i])
                # print "_to_:", groups['date'].iloc[i], money[i]

            if (str(groups['note'].iloc[i]).__contains__('转账给他人')
                    or str(groups['class'].iloc[i]).__contains__('转账给他人')):
                date_out.append(groups['date'].iloc[i])
                out_temp.append(money[i])
                # print "out_:", groups['date'].iloc[i], money[i]

            if (str(groups['note'].iloc[i]).__contains__('还招行信用卡')
                    or str(groups['class'].iloc[i]).__contains__('还招行信用卡')):
                date_credit.append(groups['date'].iloc[i])
                credit_temp.append(money[i])
                # print "cred:", groups['date'].iloc[i], money[i]

            if (str(groups['note'].iloc[i]).__contains__('第三方理财')
                or str(groups['class'].iloc[i]).__contains__('第三方理财')):
                date_licai.append(groups['date'].iloc[i])
                licai_temp.append(money[i])
                # print "licai:", groups['date'].iloc[i], money[i]

        # 对工资处理
        data1 = pd.DataFrame({'date': date_fare, 'value_fare': fare_temp})
        data1 = data1.groupby('date').sum()
        data1['date'] = data1.index
        data1.reset_index(drop=True)
        # 对转入处理
        data2 = pd.DataFrame({'date': date_to, 'value_to': to_temp})
        data2 = data2.groupby('date').sum()
        data2['date'] = data2.index
        data2.reset_index(drop=True)
        # print data1, data2
        # 对转出处理
        data3 = pd.DataFrame({'date': date_out, 'value_out': out_temp})
        data3 = data3.groupby('date').sum()
        data3['date'] = data3.index
        data3.reset_index(drop=True)
        # 对信用卡处理
        data4 = pd.DataFrame({'date': date_credit, 'value_credit': credit_temp})
        data4 = data4.groupby('date').sum()
        data4['date'] = data4.index
        data4.reset_index(drop=True)
        # 对理财处理
        data5 = pd.DataFrame({'date': date_licai, 'value_licai': licai_temp})
        data5 = data5.groupby('date').sum()
        data5['date'] = data5.index
        data5.reset_index(drop=True)
        # 将5个特征拼接
        rs = pd.merge(data1, data2, on=['date'], how='outer')
        rs = pd.merge(rs, data3, on=['date'], how='outer')
        rs = pd.merge(rs, data4, on=['date'], how='outer')
        rs = pd.merge(rs, data5, on=['date'], how='outer')

        bal_rs = train_data.loc[train_data['id'] == name]
        # print bal_rs
        bal_rs_temp = pd.merge(rs, bal_rs, on='date', how='outer')
        end_data = bal_rs_temp.reindex(columns=['id', 'date', 'type', 'value_fare', 'value_to',
                                                'value_out', 'value_credit', 'value_licai', 'bal'])
        # 将余额信息拼接
        bal_diff = end_data['bal'].diff(1).dropna()
        end_data['bal'] = bal_diff
        end_data = end_data.fillna(0)

        predict_diff = xgb.predict(end_data.iloc[:, 3:])
        predict_temp = round((predict_diff + end_data['bal'].iloc[0]), 2)
        if predict_temp < 0:
            predict_temp = 0.0
        predict.append(predict_temp)
        date.append(groups['date'].iloc[-1] + 7)

    data_copy = data_test.__deepcopy__()
    data_copy = data_copy.drop_duplicates(['id'])
    print(data_copy.head())
    print predict
    data_copy['date'] = date
    data_copy['bal'] = predict
    print(data_copy.head())
    data_copy.to_csv('./regression/test_output.txt', index=False, header=False)

last_try()