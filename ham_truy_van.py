import CONFIG as cfg
import query_funcs as qf
import pandas as pd
from Methods.base import Base
if cfg.METHOD == 1:
    from Methods.M1.generator import decode_formula, Generator
    temp_ = Generator()
    number_data_operand = temp_.number_data_operand
    del temp_

import numpy as np


def _top_n_theo_cot(table_name, col, n_row):
    text_1 = "*"
    return f'SELECT {text_1} FROM "{table_name}" ORDER BY "{col}" DESC LIMIT {n_row};'


def top_n_theo_cot(col, n_row):
    data, conn, list_gvf, list_field = cfg.check_config()
    vis = Base(data)
    cursor = conn.cursor()
    cursor.execute(qf.get_list_table())
    list_table = [t_[0] for t_ in cursor.fetchall()]
    list_of_list_value = []
    for table in list_table:
        query = _top_n_theo_cot(table, col, n_row)
        cursor.execute(query)
        list_value = cursor.fetchall()
        n_op = int(table[1:])
        for i in range(len(list_value)):
            temp = np.array(list(list_value[i][:n_op]))
            ct = decode_formula(temp, number_data_operand)[0]
            list_value[i] = [vis.convert_arrF_to_strF(ct)] + list(list_value[i][n_op:])
        list_of_list_value += list_value

    data = pd.DataFrame(list_of_list_value, columns=["CT"]+[f_[0] for f_ in list_field])
    data.sort_values(col, inplace=True, ignore_index=True, ascending=False)
    conn.close()
    del vis
    return data.loc[:n_row-1]


def _top_n_theo_cot_chua_truong_chi_dinh(field_id, table_name, col, n_row, len_):
    text_1 = "*"
    text_2 = f'({",".join([str(field_id + k*len_) for k in range(4)])})'
    query = f'SELECT {text_1} FROM "{table_name}" WHERE'
    for i in range(int(table_name[1:])):
        query += f' "E{i}" in {text_2} or'
    
    query = query[:-2]
    query += f'ORDER BY "{col}" DESC LIMIT {n_row};'
    return query


def top_n_theo_cot_chua_truong_chi_dinh(field_name, col, n_row):
    data, conn, list_gvf, list_field = cfg.check_config()
    vis = Base(data)
    for i in range(len(vis.operand_name)):
        if vis.operand_name[i] == field_name:
            field_id = i
            break
    else:
        raise Exception("Ten truong khong dung")

    len_ = len(vis.operand_name)
    cursor = conn.cursor()
    cursor.execute(qf.get_list_table())
    list_table = [t_[0] for t_ in cursor.fetchall()]
    list_of_list_value = []
    for table in list_table:
        query = _top_n_theo_cot_chua_truong_chi_dinh(field_id, table, col, n_row, len_)
        cursor.execute(query)
        list_value = cursor.fetchall()
        n_op = int(table[1:])
        for i in range(len(list_value)):
            temp = np.array(list(list_value[i][:n_op]))
            ct = decode_formula(temp, number_data_operand)[0]
            list_value[i] = [vis.convert_arrF_to_strF(ct)] + list(list_value[i][n_op:])
        list_of_list_value += list_value

    data = pd.DataFrame(list_of_list_value, columns=["CT"]+[f_[0] for f_ in list_field])
    data.sort_values(col, inplace=True, ignore_index=True, ascending=False)
    conn.close()
    del vis
    return data.loc[:n_row-1]
