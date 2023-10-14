#
DB_PATH = "/home/nguyen/Desktop/VIS_DB_Lib/__local_db"

#
DATA_PATH = "/home/nguyen/Desktop/VIS_DB_Lib/HOSE_File3_2023_Field.xlsx"

# Label, vi du: "VN_Y" - data Viet Nam, chu ky nam
#               "VN_Q" - data Viet Nam, chu ky quy
#               "JP_Y" - data Nhat Ban, chu ky nam
LABEL = "VN_Y"

# Lai suat khi khong dau tu trong 1 chu ky
INTEREST = 1.06

# MAX_CYCLE: sinh cong thuc cho chu ky nao
# MIN_CYCLE: cat data tu chu ky nao
MAX_CYCLE = 2023
MIN_CYCLE = 2007

# Cac phuong phap sinh
# 1: Sinh cong thuc gom cac cum con dong bac
METHOD = 1

# DB_FIELDS
# Key: Cac ham sinh du lieu co trong file "get_value_funcs.py"
# Value: List cac du lieu dau ra, co dang (ten du lieu, kieu du lieu)
DB_FIELDS = {
    "get_inv_max_infor": [
        ("CtyMax", "TEXT"),
        ("GeoMax", "REAL"),
        ("HarMax", "REAL")
    ],
    "get_inv_ngn_infor": [
        ("Nguong", "REAL"),
        ("Top5Coms", "TEXT"),
        ("GeoNgn", "REAL"),
        ("HarNgn", "REAL")
    ],
    "get_tf_score": [
        ["TrFScr", "REAL"]
    ],
    "get_ac_score": [
        ["AccScr", "REAL"]
    ],
    
}

# MODE: "generate" or "update"
# N_UPDATE: Neu MODE = "update", cap nhat "N_UPDATE" ham sinh du lieu tinh tu ham cuoi cung
#           Neu MODE = "generate" thi N_UPDATE khong co y nghia gi
MODE = "generate"
N_UPDATE = 2


# ===========================================================================
import os
import pandas as pd
import json
import sqlite3
from Methods.base import Base
import get_value_funcs as gvf


def check_data_operands(op_name_1: dict, op_name_2: dict):
    if len(op_name_1) != len(op_name_2): return False

    op_1_keys = list(op_name_1.keys())
    op_2_keys = list(op_name_2.keys())
    for i in range(len(op_name_1)):
        if op_name_1[op_1_keys[i]] != op_name_2[op_2_keys[i]]:
            return False

    return True


def check_config():
    folder_data = f"{DB_PATH}/{LABEL}"
    os.makedirs(folder_data, exist_ok=True)

    data = pd.read_excel(DATA_PATH)
    data = data[data["TIME"] <= MAX_CYCLE]
    data = data[data["TIME"] >= MIN_CYCLE]
    base = Base(data)

    if not os.path.exists(folder_data + "/operand_names.json"):
        with open(folder_data + "/operand_names.json", "w") as fp:
            json.dump(base.operand_name, fp, indent=4)
        operand_name = base.operand_name
    else:
        with open(folder_data + "/operand_names.json", "r") as fp:
            operand_name = json.load(fp)

    if not check_data_operands(base.operand_name, operand_name):
        raise Exception("Sai data operands, kiem tra lai ten truong, thu tu cac truong trong data")

    folder_cycle_method = folder_data + f"/CYC_{MAX_CYCLE}/MET_{METHOD}"
    os.makedirs(folder_cycle_method, exist_ok=True)
    connection = sqlite3.connect(f"{folder_cycle_method}/formula.db")

    if MODE == "generate":
        list_gvf = [getattr(gvf, key) for key in DB_FIELDS.keys()]
        list_db_field = []
        for key in DB_FIELDS.keys():
            list_db_field += DB_FIELDS[key]
    elif MODE == "update":
        list_key = list(DB_FIELDS.keys())
        list_gvf = [getattr(gvf, key) for key in list_key[-N_UPDATE:]]
        list_db_field = []
        for key in list_key[-N_UPDATE:]:
            list_db_field += DB_FIELDS[key]
    else: raise

    return data, connection, list_gvf, list_db_field
