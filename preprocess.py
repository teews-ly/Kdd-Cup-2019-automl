import datetime

import CONSTANT
from util import log, timeit
import pandas as pd
import random
import numpy as np
# import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter

'''
@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname])


@timeit
def clean_df(df):
    fillna(df)
'''
@timeit
def clean_tables(tables, config, fill):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname], tname, config, fill)
        #log(f"processing Time in table {tname}")
        #trans_timecol(tables[tname])

@timeit
def trans_timecol(df):
    time_cols = [col for col in df.columns if col.startswith("t_") and not col.startswith("t_0")]
    for t_col in time_cols:
        min_time = df[t_col].min()
        df["n_trans("+t_col+")"] = (df[t_col]-min_time).apply(lambda s: s.total_seconds())


@timeit
def clean_df(df, tname, config,  fill=True, drop_ratio=0.3):  # 0.3改为0.5  modified:5.22
    missing_ratio = df.isnull().sum(axis=0).values.astype('float') / df.shape[0]
    drop_columns0 = df.columns[np.arange(df.shape[1])[missing_ratio > drop_ratio]]
    drop_columns = [col for col in drop_columns0 if not col.startswith("t_0") and not col.startswith("c_0")]
    df.drop(drop_columns,axis=1,inplace=True)
    for col in drop_columns:
        if tname != "no_table":
            config["tables"][tname]["type"].pop(col)
        print(f"drop column: {col} from table: {tname}")
    if fill:
        fillna(df)


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(0, inplace=True) #-1

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)


@timeit
def feature_engineer(df, config, feature_num, labels):
    #if feature_num > 250:
    #    transform_categorical_hash(df)
    #else:
    transform_categorical_hash_v2(df, labels)
    transform_datetime(df, config)


@timeit
def transform_datetime(df, config):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(str(x).split(',')[0]))

@timeit
def transform_categorical_hash_v2(df, labels):  # 改进类别编码
    cols = df.columns
    cat_cols = [c for c in cols if c.startswith(CONSTANT.CATEGORY_PREFIX)]

    if len(cat_cols)>0:
        enc=OrdinalEncoder()
        df_encoded=enc.fit_transform(df[cat_cols])
    i=0
    for c in cat_cols:
        #l1 = len(df[c].unique())
        #if l1 > 1 and l1 < 0.6*df_len:
        #    df["n_count_"+c] = CAT_transform(df[[c]])  # 频率编码，ordinalEncoder，平均数编码
        #    print(f"Processing CatEncoder to {c}")
        #if len(df[c].unique())<1000 or flag==1:
        #if flag==1:
        #    encoder=ce.OrdinalEncoder(cols=[c])
        #    df["ordinal_enc_"+c] = encoder.fit_transform(df[[c]])
        #    print(f"Processing Ordinal Encoder to {c}")
        #df[c] = df[c].apply(lambda x: int(x))
        df[c]=df_encoded[:,i]
        i+=1

    multicat_cols=[c for c in cols if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    if len(multicat_cols)>0:
        enc=OrdinalEncoder()
        df_encoded=enc.fit_transform(df[multicat_cols])
    i=0
    for c in multicat_cols:
        #mark, x_encode= MV_transform(df[c])
        #if mark==1:
        #    df["n_count_" + c] = x_encode   # 频率编码，ordinalEncoder，平均数编码
        # df["n_count_"+c]= MV_transform(df[c])
        #df["n_count_" + c] = df[c].apply(lambda x: int(str(x).split(',')[0]))
        #df["n_count2_"+c] = df[c].apply(lambda x: len(str(x).split(','))) # 多类别编码
        df["n_count_" + c], df["n_count2_" + c] = zip(*df[c].map(mv_value))
        #print("len cat: ",len(df[c]))
        df[c]=df_encoded[:, i]
        i=i+1
        print(f"Processing MVEncoder to {c}")
        #df[c] = df[c].apply(lambda x: int(str(x).split(',')[0]))


def mv_value(x):
    x1=str(x).split(',')
    return int(x1[0]), len(x1)

def seperate(x):
    try:
        x = tuple(x.split(','))
    except AttributeError:
        x = ('-1', )
    return x


def MV_transform_v0(X, max_cat_num=20000):

    X = X.map(seperate)

    cat_count = {}
    for cats in X:
        for c in cats:
            try:
                cat_count[c] += 1
            except KeyError:
                cat_count[c] = 1
        #if len(cat_count)>20000: break
    #if len(cat_count)> 20000:
    #    return 0, X
    cat_list = np.array(list(cat_count.keys()))
    cat_num = np.array(list(cat_count.values()))
    idx = np.argsort(-cat_num)
    cat_list = cat_list[idx]
    print("cat len:",len(cat_list))

    mapping = {}
    for i, cat in enumerate(cat_list):
        mapping[cat] = min(i, max_cat_num)
    del cat_count, cat_list, cat_num

    X_encode = X.map(lambda cats: min((mapping[c] for c in cats)))

    return X_encode

def MV_transform(X):
    mvCats=",".join(list(X)).split(",")
    cat_count = Counter(mvCats)
    del mvCats
    print("cat len:", len(cat_count))

    X = X.map(seperate)
    X_encode = X.map(lambda cats: max((cat_count[c] for c in cats)))
    del cat_count
    return X_encode


def CAT_transform(X):
    col = X.columns[0]
    count = X.groupby(col).size().reset_index().rename(columns={0: col + '_count'})
    X = X.merge(count, how='left', on=col)
    X_encode = X[col + '_count'].astype('int32')
    del X, count
    return X_encode


@timeit
def sampling(X, y):
    index = list(y.index)
    pos_index = list(y[y == 1].index)
    neg_index = list(y[y == 0].index)
    pos_len = len(pos_index)
    neg_len = len(neg_index)

    if pos_len > neg_len and pos_len/neg_len > 2:
        pos_index2 = list(y.loc[pos_index].sample(frac=float(neg_len) / pos_len * 2.0, random_state=2019).index)
        index = pos_index2+neg_index
        random.shuffle(index)
    elif neg_len > pos_len and neg_len/pos_len > 2:
        neg_index2 = list(y.loc[neg_index].sample(frac=float(pos_len) / neg_len * 2.0, random_state=2019).index)
        index = neg_index2+pos_index
        random.shuffle(index)
    return X.loc[index], y.loc[index]

@timeit
def process_cat_label(table, label):
    cat_cols = [col for col in table.columns if col.startswith("c_") and len(table[col].unique())<40]  # ["mean", "sum"]
    agg_func = {"label": ['mean']}
    catLabel_dict = {}
    for cat_col in cat_cols:
        temp_table = pd.DataFrame()
        temp_table[cat_col] = table[cat_col]
        temp_table[cat_col].fillna("0", inplace=True)
        temp_table["label"] = label
        v=temp_table.groupby(cat_col).agg(agg_func)
        v.columns = v.columns.map(lambda a:
                                  f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]}_in_main_by_{cat_col}_in_main)")
        catLabel_dict[cat_col] = v
        #print(f"In process_cat_label:{cat_col}")
        #print(v.head())
    return catLabel_dict

@timeit
def process_main_cat(table, cat_cols0, num_fea):
    cat_cols=[col for col in cat_cols0 if not col.startswith("c_0")]
    num_cols = [col for col in table.columns if col.startswith("n_")]
    #agg_func = dict([(col, ['mean', 'sum']) for col in num_cols])
    agg_func=dict()
    agg_func['c_01'] = ["count"]
    catFea_dict = {}
    for cat_col in cat_cols:
        '''
        if num_fea > 150:
            num_fea += 1
            v = table.groupby(cat_col).agg({"c_01": ["count"]})
        else:
            num_fea += (len(num_cols)*2+1)
            v = table.groupby(cat_col).agg(agg_func)
        '''
        num_fea += 1
        v = table.groupby(cat_col).agg({"c_01": ["count"]})
        v.columns = v.columns.map(lambda a:
                                  f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]}_in_main_by_{cat_col}_in_main)")

        #print(f"In process_main_cat:{cat_col}")
        #print(v.head())
        v = table[[cat_col]].join(v, on=cat_col).drop(cat_col, axis=1) # modified 05.31
        catFea_dict[cat_col] = v
        #result_table = result_table.join(v, on=cat_col)
    return catFea_dict, num_fea

@timeit
def process_main_cat_v2(table, cat_cols0, key, tableB, num_fea):
    cat_cols = [col for col in cat_cols0 if col != key]
    num_cols = [col for col in table.columns if col.startswith("n_")]
    agg_func = dict([(col, ['mean', 'sum']) for col in num_cols])
    #agg_func[key] = ["count"]
    catFea_dict = {}
    flag = 0
    for cat_col in cat_cols:
        if num_fea > 150:
            flag = 1
            break
        num_fea += (len(num_cols)*2+1)
        v = table.groupby(cat_col).agg(agg_func)
        v.columns = v.columns.map(lambda a:
                                  f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]}_in_{tableB}_by_{cat_col}_in_main)")

        # print(f"In process_main_cat:{cat_col}")
        # print(v.head())
        catFea_dict[cat_col] = v
        # result_table = result_table.join(v, on=cat_col)
    return catFea_dict, num_fea

@timeit
def process_relation_cat(table, relation_cat_cols, key, tableB, num_fea):
    cat_cols = [col for col in relation_cat_cols if col != key]
    #num_cols = [col for col in table.columns if col.startswith("n_")]
    #agg_func = dict([(col, ['mean', 'sum']) for col in num_cols])
    agg_func = dict()
    agg_func[key] = ["count"]
    catFea_dict = [] # {} modified 05.31
    flag = 0
    for cat_col in cat_cols:
        #num_fea += (len(num_cols) * 2 + 1)
        num_fea += 1
        v = table.groupby(cat_col).agg(agg_func)
        v.columns = v.columns.map(lambda a:
                                  f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]}_in_main_by_{cat_col}_in_{tableB})")

        # print(f"In process_main_cat:{cat_col}")
        # print(v.head())
        v = table[[cat_col]].join(v, on=cat_col).drop(cat_col, axis=1)  # modified 05.31
        #new_name="table_"+key+"."+cat_col+"." + tableB
        # v.rename(index={cat_col:new_name}, inplace=True)
        #catFea_dict[new_name] = v
        catFea_dict.append(v)
        # result_table = result_table.join(v, on=cat_col)
    return catFea_dict, num_fea


@timeit
def process_relation_cat_v2(table, relation_cat_cols, key, tableB, num_fea):
    cat_cols = [col for col in relation_cat_cols if col != key]
    #num_cols = [col for col in table.columns if col.startswith("n_")]
    #agg_func = dict([(col, ['mean', 'sum']) for col in num_cols])
    agg_func = dict()
    agg_func[key] = ["count"]
    catFea_dict = {}
    flag = 0
    for cat_col in cat_cols:
        #num_fea += (len(num_cols) * 2 + 1)
        num_fea += 1
        v = table.groupby(cat_col).agg(agg_func)
        v.columns = v.columns.map(lambda a:
                                  f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]}_in_{tableB}_by_{cat_col}_in_{tableB})")

        # print(f"In process_main_cat:{cat_col}")
        # print(v.head())
        new_name="table_"+key+"."+cat_col+"." + tableB
        # v.rename(index={cat_col:new_name}, inplace=True)
        catFea_dict[new_name] = v
        # result_table = result_table.join(v, on=cat_col)
    return catFea_dict, num_fea

@timeit
def process_relation_time(temp_main_time):
    # hour  weekday day delta
    t_cols = [col for col in temp_main_time.columns if col.startswith("t_") and col != "t_01"]
    for col in t_cols:
        #print(temp_main_time.head())
        temp_main_time["n_weekday_" + col], temp_main_time["n_hour_" + col], temp_main_time["n_day_" + col] = zip(*temp_main_time[col].map(trans2basicInfo))
        #temp_main_time["n_weekday_" + col] = temp_main_time[col].apply(trans2weekday)
        #temp_main_time["n_hour_" + col] = temp_main_time[col].apply(trans2hour)
        #temp_main_time["n_day_" + col] = temp_main_time[col].apply(trans2day)
        temp_main_time["n_interval_" + col] = (temp_main_time[col]-temp_main_time["t_01"]).map(trans2interval)
        #print(temp_main_time.head())
    for col1 in t_cols:
        for col2 in t_cols:
            if col2>col1:
                temp_main_time["n_interval_" + col2+"_"+col1] = (temp_main_time[col2] - temp_main_time[col1]).map(trans2interval)
    temp_main_time.drop(t_cols, axis=1, inplace=True)
    temp_main_time.drop("t_01", axis=1, inplace=True)
    return temp_main_time


def trans2weekday(time):
    return time.isoweekday()


def trans2hour(time):
    return time.hour


def trans2day(time):
    return time.day


def trans2basicInfo(time):
    return time.isoweekday(), time.hour, time.day


def trans2interval(timedelta):
    return timedelta.total_seconds()

'''
@timeit
def process_relation_cat_v2(table, relation_cat_cols, key, tableB, num_fea):
    # relation 表中自身cat对应自身num
    cat_cols = [col for col in relation_cat_cols if col != key]
    num_cols = [col for col in table.columns if col.startswith("n_")]
    agg_func = dict([(col, ['mean', 'sum']) for col in num_cols])
    agg_func[key] = ["count"]
    catFea_dict = {}
    flag = 0
    for cat_col in cat_cols:
        if num_fea > 150:
            flag = 1
            break
        num_fea += (len(num_cols) * 2 + 1)
        v = table.groupby(cat_col).agg(agg_func)
        v.columns = v.columns.map(lambda a:
                                  f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]}_in_{tableB}_by_{cat_col}_in_{tableB})")

        # print(f"In process_main_cat:{cat_col}")
        # print(v.head())
        new_name = "table_" + key + "." + cat_col + "." + tableB
        # v.rename(index={cat_col: new_name}, inplace=True)
        catFea_dict[new_name] = v
        # result_table = result_table.join(v, on=cat_col)
    return catFea_dict, num_fea
'''
