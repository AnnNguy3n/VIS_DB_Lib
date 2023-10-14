def get_list_table():
    return '''SELECT name FROM sqlite_master WHERE type = "table";'''


def create_table(num_operand, list_field):
    list_formula_col = [f'"E{i}" INTEGER NOT NULL,' for i in range(num_operand)]
    list_field_col = [f'"{field[0]}" {field[1]},' for field in list_field]
    temp = "\n    "
    return f'''CREATE TABLE "F{num_operand}"(
    {temp.join(list_formula_col)}
    {temp.join(list_field_col)[:-1]}
)'''


def get_last_row(table_name):
    return f'''SELECT * FROM "{table_name}" ORDER BY ROWID DESC LIMIT 1;'''


def create_table_update(num_operand, list_field):
    list_field_col = [f'"{field[0]}" {field[1]},' for field in list_field]
    temp = "\n    "
    return f'''CREATE TABLE "T{num_operand}"(
    {temp.join(list_field_col)[:-1]}
)'''


def insert_rows(table_name, list_list_value, list_field=[]):
    if len(list_field) == 0:
        text_1 = ""
    else:
        list_field_name = [f_[0] for f_ in list_field]
        text_1 = f'''({", ".join([f'"{field}"' for field in list_field_name])})'''

    text_2 = ""
    for list_value in list_list_value:
        text = ""
        for value in list_value:
            if type(value) == str:
                text += f'"{value}",'
            else:
                text += f"{value},"

        text_2 += f"({text[:-1]}),"

    return f'''INSERT INTO "{table_name}"{text_1} VALUES {text_2[:-1]};'''


def get_last_rowid(table_name):
    return f'''SELECT ROWID FROM "{table_name}" ORDER BY ROWID DESC LIMIT 1;'''


def get_row_by_rowid(table_name, rowid):
    return f'''SELECT * FROM "{table_name}" WHERE ROWID = {rowid};'''
