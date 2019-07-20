import os

os.system('apt-get clean')
os.system('mv /var/lib/apt/lists /var/lib/apt/lists.old')
os.system('mkdir -p /var/lib/apt/lists/partial')
os.system('apt-get clean')
os.system('apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC')
os.system('apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138')
os.system('apt-get update')
os.system('apt-get -y install gcc build-essential libdpkg-perl')
os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")
os.system("pip3 install featuretools==0.7.1")
#os.system("pip3 install category-encoders==1.2.7")
import copy
import numpy as np
import pandas as pd
import datetime
import time

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME
from merge import merge_table, merge_table_v2, FT_process
from preprocess import clean_df, clean_tables, feature_engineer, process_main_cat, process_cat_label, trans2basicInfo, trans2interval, process_relation_cat, process_relation_time, process_relation_cat_v2, trans2weekday, trans2hour, trans2day
from util import Config, log, show_dataframe, timeit
import warnings

warnings.filterwarnings('ignore')


class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        self.lables = None

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = Xs
        self.lables = y
        '''
        clean_tables(Xs)
        #if row_number > 700000 and time < 700:
        #    X = merge_table(Xs, self.config)  # 表连接
        #else:
        config2 = copy.deepcopy(self.config)
        merge_table_v2(Xs, self.config)
        # clean_tables(Xs)
        X = FT_process(Xs, self.config)  # 考虑时间窗后会改变index顺序
        #times = X["t_01"].min() + self.config["window_number"] * datetime.timedelta(seconds=self.config["timeBucket"])
        #X = X[X["t_01"] > times]
        X.sort_index(inplace=True)
        self.config["tables"] = config2["tables"]
        self.config["relations"] = config2["relations"]

        clean_df(X)
        feature_engineer(X, self.config)
        # new_y=y.loc[X.index]
        # train_X, train_y=sampling(X, new_y)
        # train(train_X, train_y, self.config)
        train(X, y.loc[X.index], self.config)
        '''

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        main_time_index = main_table[["t_01"]].sort_values("t_01")
        # catLabel_dict = process_cat_label(main_table, self.lables.loc[main_table.index]) # modified By 05.30
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table
        clean_tables(Xs, self.config, fill=True)
        main_table = Xs[MAIN_TABLE_NAME]

        main_cat_cols = [col for col in main_table.columns if (col.startswith("c_") or col.startswith("m_")) and len(main_table[col].unique())>1]
        total_num_fea = 0
        catFea_dict, total_num_fea = process_main_cat(main_table, main_cat_cols, total_num_fea)  # 专门利用主表提其他类别特征针对main的特征
        print("total_num Fea:", total_num_fea)
        catFea_dicts = []
        relation_catFea_dicts = []
        relation_time_dicts = []
        relation_catFea_dicts2 = []
        if total_num_fea < 150:  # 表示主表的衍生特征不够多，还可加
            for relation in self.config['relations']:
                tableA=relation["table_A"]
                l_type=relation["type"].split("_")[0]
                tableB = relation["table_B"]
                r_type = relation["type"].split("_")[2]
                key=relation["key"][0]
                if tableA=="main" and l_type=="many" and r_type=="one": #and "t_01" not in Xs[tableB].columns:  # 这里比较定制，后期需要改
                    '''
                    temp_main_cat = main_table[main_cat_cols]
                    relation_num_cols = [col for col in Xs[tableB].columns if col.startswith("n_")]
                    temp_tableB_num = Xs[tableB][[key]+relation_num_cols]
                    temp_tableB_num = temp_tableB_num.set_index(key)
                    temp_main_cat = temp_main_cat.join(temp_tableB_num, on=key)
                    temp_dict, total_num_fea = process_main_cat_v2(temp_main_cat, main_cat_cols, key, tableB, total_num_fea) #main的类别，relation的numerical
                    catFea_dicts.append(temp_dict)
                    if total_num_fea > 150: break
                    '''
                    Xs[tableB].drop_duplicates([key], inplace=True)
                    relation_cat_cols = [col for col in Xs[tableB].columns if
                                         (col.startswith("c_") or col.startswith("m_")) and len(Xs[tableB][col].unique()) > 1]
                    temp_tableB_cat=Xs[tableB][relation_cat_cols]
                    if key in main_table and key in temp_tableB_cat:
                        temp_main_num = main_table[[key]]
                        temp_tableB_cat = temp_tableB_cat.set_index(key)
                        temp_main_num = temp_main_num.join(temp_tableB_cat, on=key)
                        relation_temp_dict, total_num_fea = process_relation_cat(temp_main_num, relation_cat_cols, key, tableB, total_num_fea) #relation的类别，main的numerical
                        #relation_catFea_dicts.append(relation_temp_dict)
                        relation_catFea_dicts=relation_catFea_dicts+relation_temp_dict
                        # if total_num_fea > 150: break
                        '''
                        temp_tableB_cat = Xs[tableB][relation_cat_cols]
                        relation_temp_dict2, total_num_fea = process_relation_cat_v2(temp_tableB_cat, relation_cat_cols, key,
                                                                                 tableB,
                                                                                 total_num_fea)
                        relation_catFea_dicts2.append(relation_temp_dict2)
                        '''


                    relation_time_cols = [col for col in Xs[tableB].columns if col.startswith("t_")]
                    if len(relation_time_cols) > 0:
                        if key in Xs[tableB] and key in main_table and "t_01" in main_table:
                            temp_tableB_time = Xs[tableB][[key]+relation_time_cols]
                            temp_tableB_time.columns = [col+"_in_"+tableB if col.startswith("t_") else col for col in temp_tableB_time.columns]
                            temp_main_time = main_table[[key] + ["t_01"]]
                            temp_tableB_time = temp_tableB_time.set_index(key)
                            temp_main_time = temp_main_time.join(temp_tableB_time, on=key)
                            temp_main_time.drop(key, axis=1, inplace=True)
                            #print("time_test v1")
                            #print(temp_main_time.head())
                            temp_main_time = process_relation_time(temp_main_time)
                            relation_time_dicts.append(temp_main_time)


                    '''
                    temp_tableB = Xs[tableB].set_index(key)
                    temp_main_key = main_table[[key]]
                    temp_main_key = temp_main_key.join(temp_tableB, on=key)
                    relation_temp_dict2, total_num_fea = process_relation_cat_v2(temp_main_key, relation_cat_cols, key,
                                                                                 tableB, total_num_fea)
                    del temp_main_key
                    del temp_tableB
                    relation_catFea_dicts2.append(relation_temp_dict2)
                    if total_num_fea > 150: break
                    '''
        '''
        #if len(relation_time_dicts) > 0:
        main_time_col=[col for col in main_table.columns if col.startswith("t_")]
        temp_main_time = main_table[main_time_col]
        for col in main_time_col:
            temp_main_time["n_weekday_" + col], temp_main_time["n_hour_" + col], temp_main_time["n_day_" + col]=zip(*temp_main_time[col].map(trans2basicInfo))
            # temp_main_time["n_weekday_" + col] = temp_main_time[col].apply(trans2weekday)
            # temp_main_time["n_hour_" + col] = temp_main_time[col].apply(trans2hour)
            # temp_main_time["n_day_" + col] = temp_main_time[col].apply(trans2day)
            if not col.startswith("t_0"):
                temp_main_time["n_interval_" + col] = (temp_main_time[col] - temp_main_time["t_01"]).map(trans2interval)
        temp_main_time.drop(main_time_col, axis=1, inplace=True)
        relation_time_dicts.append(temp_main_time)
        print("Processing Trans to main time")
        '''



        # Xs[MAIN_TABLE_NAME] = main_table
        # clean_tables(Xs, self.config, fill=True)
        merge_table_v2(Xs, self.config)
        #clean_tables(Xs)
        X = FT_process(Xs, self.config)
        del Xs
        del self.tables
        del main_table
        #print(X.shape)
        '''
        for catLabel in catLabel_dict:
            # print(catLabel_dict[catLabel].head())
            if catLabel in X.columns:
                X = X.join(catLabel_dict[catLabel], on=catLabel)
        '''
        t1=time.time()
        useful_catFea=[catFea_dict[catFea] for catFea in catFea_dict if catFea in X.columns]
        X = pd.concat([X] + useful_catFea, axis=1)
        print("processing process_main_cat")
        '''
        for catFea in catFea_dict:
            if catFea in X.columns:
                #print(catFea_dict[catFea].head())
                X = X.join(catFea_dict[catFea], on=catFea)
                print("processing process_main_cat")
            #print(X.head())
        '''
        del catFea_dict
        '''
        for catFea_dict2 in catFea_dicts:
            for catFea in catFea_dict2:
                if catFea in X.columns:
                    #print(catFea_dict2[catFea].head())
                    X = X.join(catFea_dict2[catFea], on=catFea)
                    print("processing process_main_cat_v2")
                    #print(X.head())
        del catFea_dicts
        '''
        '''
        for relation_catFea_dict in relation_catFea_dicts:
            for relation_catFea in relation_catFea_dict:
                #print(relation_catFea_dict[relation_catFea].head())
                if relation_catFea in X.columns:
                    z=yield(relation_catFea_dict[relation_catFea])
                    # X = X.join(relation_catFea_dict[relation_catFea], on=relation_catFea)
                    print("processing process_relation_cat")
                    #print(X.head())
        '''
        X = pd.concat([X] + relation_catFea_dicts, axis=1)
        del relation_catFea_dicts

        if len(relation_time_dicts) > 0:
            X = pd.concat([X]+relation_time_dicts, axis=1)
            print("processing process_relation_time")
            #print(X.shape)
            #print(X.head())
            del relation_time_dicts
        '''
        for relation_catFea_dict2 in relation_catFea_dicts2:
            for relation_catFea in relation_catFea_dict2:
                #print(relation_catFea_dict2[relation_catFea].head())
                if relation_catFea in X.columns:
                    X = X.join(relation_catFea_dict2[relation_catFea], on=relation_catFea)
                    print("processing process_relation_cat_v2")
                    #print(X.head())
        del relation_catFea_dicts2
        '''
        t2=time.time()
        print("cat join cost time: ", t2-t1)
        #print(X.head())
        X.columns = [
            "m_" + c if (".m_" in c) and ("MEAN" not in c) and ("SUM" not in c) and (
                        "COUNT" not in c) and ("N_UNIQUE" not in c) and ("N_TIME" not in c) else c for c in X.columns]
        X.columns = [
            "c_" + c if (".c_" in c) and ("MEAN" not in c) and ("SUM" not in c) and (
                    "COUNT" not in c) and ("N_UNIQUE" not in c) and ("N_TIME" not in c) else c for c in X.columns]
        X.columns = [
            "n_" + c if not c.startswith("n_") and not c.startswith("m_") and not c.startswith("c_") and not c.startswith("t_") else c for c in X.columns]
        #print(X.columns)
        print("Column Number:",len(X.columns))

        clean_df(X, "no_table", self.config)
        feature_engineer(X, self.config, len(X.columns), self.lables)

        X_train = X[X.index.str.startswith("train")]
        X_train.index = X_train.index.map(lambda x: int(x.split('_')[1]))
        X_train.sort_index(inplace=True)
        #train(X_train, self.lables.loc[X_train.index], self.config)
        train(X_train.loc[main_time_index.index], self.lables.loc[main_time_index.index], self.config)  # 按时间排序
        del main_time_index

        X = X[X.index.str.startswith("test")]
        X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)
        result = predict(X, self.config)

        return pd.Series(result)

