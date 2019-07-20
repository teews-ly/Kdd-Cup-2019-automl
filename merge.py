import os
import time
from collections import defaultdict, deque
import featuretools as ft

from featuretools.variable_types import Categorical, Numeric, Datetime
from featuretools.primitives import make_agg_primitive, make_trans_primitive

import numpy as np
import pandas as pd

import CONSTANT
from util import Config, Timer, log, timeit

NUM_OP = [np.std, np.mean]

def bfs(root_name, graph, tconfig):
    tconfig[CONSTANT.MAIN_TABLE_NAME]['depth'] = 0
    queue = deque([root_name])
    while queue:
        u_name = queue.popleft()
        for edge in graph[u_name]:
            v_name = edge['to']
            if 'depth' not in tconfig[v_name]:
                tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                queue.append(v_name)


@timeit
def join(u, v, v_name, key, type_):
    if type_.split("_")[2] == 'many':
        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                     and not col.startswith(CONSTANT.TIME_PREFIX)
                     and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}
        v = v.groupby(key).agg(agg_funcs)
        v.columns = v.columns.map(lambda a:
                f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]})")
    else:
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")

    return u.join(v, on=key)


@timeit
def temporal_join(u, v, v_name, key, time_col):
    timer = Timer()

    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    tmp_u = u[[time_col, key]]
    timer.check("select")

    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    timer.check("concat")

    rehash_key = f'rehash_{key}'
    tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
    timer.check("rehash_key")

    tmp_u.sort_values(time_col, inplace=True)
    timer.check("sort")

    agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                 and not col.startswith(CONSTANT.TIME_PREFIX)
                 and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}

    tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
    timer.check("group & rolling & agg")

    tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
    timer.check("reset_index")

    tmp_u.columns = tmp_u.columns.map(lambda a:
        f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")

    if tmp_u.empty:
        log("empty tmp_u, return u")
        return u

    ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
    timer.check("final concat")

    del tmp_u

    return ret

def dfs(u_name, config, tables, graph):
    u = tables[u_name]
    log(f"enter {u_name}")
    for edge in graph[u_name]:
        v_name = edge['to']
        if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
            continue

        v = dfs(v_name, config, tables, graph)
        key = edge['key']
        type_ = edge['type']

        if config['time_col'] not in u and config['time_col'] in v:
            continue

        if config['time_col'] in u and config['time_col'] in v:
            log(f"join {u_name} <--{type_}--t {v_name}")
            u = temporal_join(u, v, v_name, key, config['time_col'])
        else:
            log(f"join {u_name} <--{type_}--nt {v_name}")
            u = join(u, v, v_name, key, type_)

        del v

    log(f"leave {u_name}")
    return u


@timeit
def merge_table(tables, config):
    graph = defaultdict(list)
    for rel in config['relations']:
        ta = rel['table_A']
        tb = rel['table_B']
        graph[ta].append({
            "to": tb,
            "key": rel['key'],
            "type": rel['type']
        })
        graph[tb].append({
            "to": ta,
            "key": rel['key'],
            "type": '_'.join(rel['type'].split('_')[::-1])
        })
    bfs(CONSTANT.MAIN_TABLE_NAME, graph, config['tables'])
    return dfs(CONSTANT.MAIN_TABLE_NAME, config, tables, graph)


@timeit
def merge_table_v2(tables, config):
    entity_config = config['tables']
    relation_config = config['relations']

    # change default column names
    for tbl_name, tbl_type in entity_config.items():
        if tbl_name != "main":
            ori_kvs = entity_config[tbl_name]['type']  # 某个表的type
            entity_config[tbl_name]['type'] = {}
            for k, v in ori_kvs.items():
                if k == config['time_col'] or k.split('_')[1].startswith('0'):  # key值为主时间或者外键
                    entity_config[tbl_name]['type'][k] = v
                else:
                    entity_config[tbl_name]['type'][k + "." + tbl_name] = v

    # change table accordingly
    for tbl_name, tbl_df in tables.items():
        if tbl_name != "main":
            ori_clns = tbl_df.columns.tolist()  # 某个表的columns
            new_clns = []
            for k in ori_clns:
                if k == config['time_col'] or k.split('_')[1].startswith('0'):
                    new_clns.append(k)
                else:
                    new_clns.append(k + "." + tbl_name)
            tbl_df.columns = new_clns  # 将每个表的columns名称变换

    keys = set()
    for k in relation_config:  # modified By Ly 4.22 解决one表带t_01的问题
        # key default to be one
        key = k['key'][0]
        keys.add(key)

    entity_table_dict = {}
    for k in keys:
        tables_name = set()
        for i in relation_config:
            if i['key'][0] == k:
                tables_name.add(i['table_A'])
                tables_name.add(i['table_B'])
        entity_table_dict[k] = tables_name  # 针对每一个外键 对应的包含几种表的集合

    # construct new entity dataframe
    tmp_relation_config = relation_config.copy()
    config['relations'] = []
    for k in keys:  # 针对每一个外键
        tbls = []
        for t in entity_table_dict[k]:
            tbls.append(tables[t][k])
        tables['table_' + k] = pd.DataFrame(pd.concat(tbls).unique(), columns=[k])  # 生成中间表
        entity_config['table_' + k] = {}
        entity_config['table_' + k]['type'] = {k: 'cat'}  # 对应在config文件中进行添加

        l_rlts = set()
        r_rlts = set()

        for r in tmp_relation_config:
            if r['key'][0] == k:
                l_rlts.add((r['table_A'], r['type'].split("_")[0]))  # 左表名称和one-many属性
                r_rlts.add((r['table_B'], r['type'].split("_")[2]))  # 右表名称和one-many属性
        for l_tbl, l_t in l_rlts:
            #if l_t == 'one' and ("t_01" not in tables[l_tbl].columns):  # 若左表是one，进行合并  # 4.18 modified By Ly
            if l_t == 'one':
                tables['table_' + k] = pd.merge(tables['table_' + k], tables[l_tbl], how='left', on=k)
                tables.pop(l_tbl)
                entity_config['table_' + k]['type'].update(entity_config[l_tbl]['type'])
            #elif l_t == 'one' and ("t_01" in tables[l_tbl].columns):
            #    print("Error: Unpossible Condition")
            else: # 若左表是many，添加一条关系
                config['relations'].append({
                    'table_A': l_tbl,
                    'table_B': 'table_' + k,
                    'key': [k],
                    'type': 'many_to_one'
                })

        for r_tbl, r_t in r_rlts:
            #if r_t == 'one' and ("t_01" not in tables[r_tbl].columns):  # 若右表是one，进行合并  # 4.18 modified By Ly
            if r_t == 'one':
                tables['table_' + k] = pd.merge(tables['table_' + k], tables[r_tbl], how='left', on=k)
                tables.pop(r_tbl)
                entity_config['table_' + k]['type'].update(entity_config[r_tbl]['type'])
            #elif r_t == 'one' and ("t_01" in tables[r_tbl].columns):
            #    print("Error: Unpossible Condition")
            else:  # 若右表是many，添加1条关系
                config['relations'].append({
                    'table_A': r_tbl,
                    'table_B': 'table_' + k,
                    'key': [k],
                    'type': 'many_to_one'
                })
# tables有变化，config中tables和对应type有变化，config中relation被重置了

@timeit
def FT_process(tables, config):
    es = ft.EntitySet()
    entity_config = config['tables']
    relation_config = config['relations']
    flag = 0
    for table in tables:
        id = f'{table}_id'  # 主键
        make_id = True
        if len(table.split("_")) > 2:  # 中间表
            id = table[6:]
            make_id = False
        if table == CONSTANT.MAIN_TABLE_NAME:  # "main"
            tables[table][id]=tables[table].index
            cat_cols=[col for col in tables[table].columns if col.startswith("c_") and not col.startswith("c_0")]
            if len(cat_cols) > 10:
                flag=1
            make_id = False

        variable_Types={}
        for col in tables[table].columns:
            if col.startswith(CONSTANT.MULTI_CAT_PREFIX):
                variable_Types[col] = ft.variable_types.Categorical
            if col.startswith(CONSTANT.CATEGORY_PREFIX):
                variable_Types[col] = ft.variable_types.Categorical
        '''
        if config['time_col'] in tables[table] and table == "main":  # modified 4.22, time_index的设置
            es = es.entity_from_dataframe(entity_id=table,
                                          dataframe=tables[table],
                                          make_index=make_id,
                                          index=id,
                                          time_index=config['time_col'],
                                          variable_types=variable_Types
                                          )
            #print(table,"using time_index")
        else:
            es = es.entity_from_dataframe(entity_id=table,
                                          dataframe=tables[table],
                                          make_index=make_id,
                                          index=id,
                                          variable_types=variable_Types
                                          )
        '''
        es = es.entity_from_dataframe(entity_id=table,
                                      dataframe=tables[table],
                                      make_index=make_id,
                                      index=id,
                                      variable_types=variable_Types
                                      )

        # print(es[table].variables)


    for relation in relation_config:
        tableA = relation['table_A']
        tableB = relation['table_B']
        key = relation['key'][0]
        new_relationship = ft.Relationship(es[tableB][key], es[tableA][key])
        es = es.add_relationship(new_relationship)
    '''
    ct = pd.DataFrame()
    c_id = f'{CONSTANT.MAIN_TABLE_NAME}_id'
    ct[c_id] = tables[CONSTANT.MAIN_TABLE_NAME].index
    ct["time"] = tables[CONSTANT.MAIN_TABLE_NAME][config['time_col']].values
    time0 = ct["time"].min()
    time1 = ct["time"].max()
    timeBucket = (time1 - time0) / 20
    if "timeBucket" not in config:
        config["timeBucket"] = timeBucket.total_seconds()
        config["window_number"] = 5
    '''
    # print(config["timeBucket"])
    # cluster = LocalCluster()

    '''
    if mark ==1: # modified 4.23
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="main", agg_primitives=["mean", "sum", "count"],
                                              trans_primitives=["hour", "weekday"],
                                              max_depth=2,
                                              cutoff_time=ct,
                                              training_window=ft.Timedelta(config["window_number"] * config["timeBucket"], "s"), # 参数可调
                                              approximate=ft.Timedelta(config["timeBucket"], "s"),  # 参数可调
                                              # n_jobs=3,
                                              cutoff_time_in_index=True # 参数可调
                                              )
        # print(feature_defs)
        feature_matrix.reset_index(1, drop=False, inplace=True)
        feature_matrix.rename(columns={'time': 't_01'}, inplace=True)
        print("Using Cutting off Time")
    else:
    '''
    def n_unique(column):
        return len(set(column))
    def nunique2(column):
        l1=len(column)
        return l1*1.0/len(set(column))
    def n_time(column):
        return (column.max()-column.min()).total_seconds()
    def n_time2(column):
        return (column-column.min()).apply(lambda s: s.total_seconds())
    nunique = make_agg_primitive(function=n_unique, input_types=[Categorical], return_type=Numeric)
    # ntime = make_agg_primitive(function=n_time, input_types=[Datetime], return_type=Numeric)
    # ntime2 = make_trans_primitive(function=n_time2, input_types=[Datetime], return_type=Numeric)
    if flag == 0:
        agg_trans = ["mean", "sum", "count", nunique]
    else:
        agg_trans = ["mean", "sum", "count"]

    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="main",
                                          agg_primitives=agg_trans,  # "num_unique"太耗时
                                          trans_primitives=[],  # ["hour", "weekday"],
                                          max_depth=2
                                          )
    print(feature_defs)
    # feature_matrix.columns = ["m_"+c if ((".c_" in c) or (".m_" in c)) and ("MEAN" not in c) and ("SUM" not in c) and ("COUNT" not in c) else c for c in feature_matrix.columns]
    return feature_matrix