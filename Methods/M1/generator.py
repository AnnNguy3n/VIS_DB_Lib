import pandas as pd
import numpy as np
import numba as nb
from Methods import base
from Methods.M1 import gm1
import get_value_funcs as gvf
import query_funcs as qf
import CONFIG as cfg

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def decode_formula(f, len_):
    len_f = len(f)
    rs = np.full(len_f*2, 0)
    check = False
    n = 1
    for i in range(len_f):
        rs[2*i] = f[i] // len_
        rs[2*i+1] = f[i] % len_
        if i != 0 and not check:
            if f[i] >= len_*2:
                n += 1
            else:
                check = True

    return rs, n


@nb.njit
def encode_formula(f, len_):
    len_f = len(f)
    rs = np.full(len_f//2, 0)
    for i in range(len(rs)):
        rs[i] = f[2*i]*len_ + f[2*i+1]

    return rs


class Generator(base.Base):
    def __init__(self) -> None:
        data, connection, list_gvf, list_db_field = cfg.check_config()
        super().__init__(data)
        self.connection = connection
        self.interest = cfg.INTEREST
        self.list_gvf = list_gvf
        self.list_db_field = list_db_field

        self.cursor = connection.cursor()
        self.number_data_operand = len(self.operand_name.keys())

    def generate(self):
        self.mode = 1
        query = qf.get_list_table()
        # print(query)
        self.cursor.execute(query)
        list_table_name = [t_[0] for t_ in self.cursor.fetchall()]
        n = 1
        while True:
            if f"F{n}" not in list_table_name:
                last_number_operand = n - 1
                break
            n += 1

        if last_number_operand == 0:
            last_number_operand = 1
            query = qf.create_table(1, self.list_db_field)
            # print(query)
            self.cursor.execute(query)
            self.connection.commit()
            print(f"Đã tạo bảng F{1}")

        query = qf.get_last_row(f"F{last_number_operand}")
        # print(query)
        self.cursor.execute(query)
        last_row = self.cursor.fetchone()
        if last_row is None:
            last_formula = np.full(last_number_operand*2, 0)
            num_op_in_clus = 1
        else:
            last_formula = last_row[:last_number_operand]
            last_formula, num_op_in_clus = decode_formula(last_formula, self.number_data_operand)
            last_formula[-1] += 1

        list_divisor = [i for i in range(1, last_number_operand+1) if last_number_operand % i == 0]
        last_divisor_idx = list_divisor.index(num_op_in_clus)
        self.last_formula = last_formula
        self.last_divisor_idx = last_divisor_idx
        print(last_formula, last_divisor_idx)
        self.run()

    def run(self):
        self.list_of_list_value = []

        self.history = [self.last_formula.copy(), self.last_divisor_idx]
        self.current = [self.last_formula.copy(), self.last_divisor_idx]
        self.count = np.array([0, 100000, 0, 1000000000000])
        last_operand = self.current[0].shape[0] // 2
        num_operand = last_operand
        if self.mode == 1:
            self.current_table_name = f"F{num_operand}"
        elif self.mode == 2:
            self.current_table_name = f"T{num_operand}"
        else: raise

        while True:
            print(f"Đang chạy, số toán hạng là {num_operand}")

            list_uoc_so = [i for i in range(1, num_operand+1) if num_operand % i == 0]
            start_divisor_idx = 0
            if num_operand == last_operand:
                start_divisor_idx = self.history[1]

            formula = np.full(num_operand*2, 0)
            for i in range(start_divisor_idx, len(list_uoc_so)):
                print("Số phần tử trong 1 cụm", list_uoc_so[i])
                struct = np.array([[0, list_uoc_so[i], 1+2*list_uoc_so[i]*j, 0] for j in range(num_operand//list_uoc_so[i])])
                if num_operand != last_operand or i != self.current[1]:
                    self.current[0] = formula.copy()
                    self.current[1] = i

                self.__fill_1(formula, struct, 0, np.zeros(self.OPERAND.shape[1]), 0, np.zeros(self.OPERAND.shape[1]), 0, False, False)

            self.save()

            num_operand += 1
            if self.mode == 1:
                query = qf.create_table(num_operand, self.list_db_field)
                # print(query)
                self.cursor.execute(query)
                self.current_table_name = f"F{num_operand}"
            elif self.mode == 2:
                query = qf.create_table_update(num_operand, self.list_db_field)
                # print(query)
                self.cursor.execute(query)
                self.current_table_name = f"T{num_operand}"
            else: raise

            self.connection.commit()
            print(f"Đã tạo bảng {self.current_table_name}")

    def __fill_1(self, formula, struct, idx, temp_0, temp_op, temp_1, mode, add_sub_done, mul_div_done):
        if mode == 0: # Sinh dấu cộng trừ đầu mỗi cụm
            gr_idx = list(struct[:,2]-1).index(idx)

            start = 0
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

            for op in range(start, 2):
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                new_struct[gr_idx,0] = op
                if op == 1:
                    new_add_sub_done = True
                    new_formula[new_struct[gr_idx+1:,2]-1] = 1
                    new_struct[gr_idx+1:,0] = 1
                else:
                    new_add_sub_done = False

                self.__fill_1(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, 1, new_add_sub_done, mul_div_done)
        elif mode == 2:
            start = 2
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

            if start == 0:
                start = 2

            valid_op = gm1.get_valid_op(struct, idx, start)
            for op in valid_op:
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                if op == 3:
                    new_mul_div_done = True
                    for i in range(idx+2, 2*new_struct[0,1]-1, 2):
                        new_formula[i] = 3

                    for i in range(1, new_struct.shape[0]):
                        for j in range(new_struct[0,1]-1):
                            new_formula[new_struct[i,2] + 2*j + 1] = new_formula[2+2*j]
                else:
                    new_struct[:,3] += 1
                    new_mul_div_done = False
                    if idx == 2*new_struct[0,1] - 2:
                        new_mul_div_done = True
                        for i in range(1, new_struct.shape[0]):
                            for j in range(new_struct[0,1]-1):
                                new_formula[new_struct[i,2] + 2*j + 1] = new_formula[2+2*j]

                self.__fill_1(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, 1, add_sub_done, new_mul_div_done)
        elif mode == 1:
            start = 0
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

            valid_operand = gm1.get_valid_operand(formula, struct, idx, start, self.OPERAND.shape[0])
            if valid_operand.shape[0] > 0:
                if formula[idx-1] < 2:
                    temp_op_new = formula[idx-1]
                    temp_1_new = self.OPERAND[valid_operand].copy()
                else:
                    temp_op_new = temp_op
                    if formula[idx-1] == 2:
                        temp_1_new = temp_1 * self.OPERAND[valid_operand]
                    else:
                        temp_1_new = temp_1 / self.OPERAND[valid_operand]

                if idx + 1 == formula.shape[0] or (idx+2) in struct[:,2]:
                    if temp_op_new == 0:
                        temp_0_new = temp_0 + temp_1_new
                    else:
                        temp_0_new = temp_0 - temp_1_new
                else:
                    temp_0_new = np.array([temp_0]*valid_operand.shape[0])

                if idx + 1 != formula.shape[0]:
                    temp_list_formula = np.array([formula]*valid_operand.shape[0])
                    temp_list_formula[:,idx] = valid_operand
                    if idx + 2 in struct[:,2]:
                        if add_sub_done:
                            new_idx = idx + 2
                            new_mode = 1
                        else:
                            new_idx = idx + 1
                            new_mode = 0
                    else:
                        if mul_div_done:
                            new_idx = idx + 2
                            new_mode = 1
                        else:
                            new_idx = idx + 1
                            new_mode = 2

                    for i in range(valid_operand.shape[0]):
                        self.__fill_1(temp_list_formula[i], struct, new_idx, temp_0_new[i], temp_op_new, temp_1_new[i], new_mode, add_sub_done, mul_div_done)
                else:
                    temp_0_new[np.isnan(temp_0_new)] = -1.7976931348623157e+308
                    temp_0_new[np.isinf(temp_0_new)] = -1.7976931348623157e+308

                    formulas = np.array([formula]*valid_operand.shape[0])
                    formulas[:, idx] = valid_operand

                    self.count[0:3:2] += self.__handle(temp_0_new, formulas)
                    self.current[0][:] = formula[:]
                    self.current[0][idx] = self.OPERAND.shape[0]

                    if self.count[0] >= self.count[1] or self.count[2] >= self.count[3]:
                        self.save()

    def __handle(self, weights, formulas):
        len_ = len(weights)
        for k in range(len_):
            weight = weights[k]
            formula = formulas[k]
            if self.mode == 1:
                list_value = list(encode_formula(formula, self.number_data_operand))
            else:
                list_value = []

            for func in self.list_gvf:
                result = func(weight, self.INDEX, self.PROFIT, self.SYMBOL, self.interest)

                if type(result) == tuple or type(result) == list:
                    list_value += list(result)
                else:
                    list_value += [result]

            self.list_of_list_value.append(list_value)

        return len_

    def save(self):
        if self.count[0] == 0:
            return

        if self.mode == 1:
            n = int(self.current_table_name[1:])
            for list_value in self.list_of_list_value:
                CtyMax = list_value[n]
                if CtyMax == -1:
                    list_value[n] = ""
                else:
                    list_value[n] = self.symbol_name[CtyMax]
                
                Top5Coms = list_value[n+4]
                list_value[n+4] = "_".join([self.symbol_name[sym] for sym in Top5Coms])

                Top5ComsNgn2 = list_value[n+10]
                list_value[n+10] = "_".join(self.symbol_name[sym] for sym in Top5ComsNgn2)

                Top5ComsNgn1_2 = list_value[n+14]
                list_value[n+14] = "_".join(self.symbol_name[sym] for sym in Top5ComsNgn1_2)

                Top5ComsNgn1_3 = list_value[n+18]
                list_value[n+18] = "_".join(self.symbol_name[sym] for sym in Top5ComsNgn1_3)

        query = qf.insert_rows(self.current_table_name, self.list_of_list_value)
        # print(query)
        self.cursor.execute(query)
        self.connection.commit()
        self.list_of_list_value.clear()
        self.count[0] = 0
        print("Saved")
