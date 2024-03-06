
import pandas as pd

# Чтение данных из Excel-файла
xls = pd.ExcelFile('data.xlsx')

data = pd.read_excel(xls, sheet_name=None)

# Создание словаря для хранения данных
xls: object = pd.ExcelFile('data.xlsx')
data: dict = pd.read_excel(xls, sheet_name=None)
data_dict: dict = {}

for sheet_name, sheet_data in data.items():
    # Поиск всех ячеек с названием "Ростовская область"
    rostov_cells = sheet_data.apply(lambda row: row.astype(str).str.contains('Ростовская область', case=False)).any(
        axis=1)

    # Получение индексов строк с названиями "Ростовская область"
    rostov_row_indexes = rostov_cells[rostov_cells].index.tolist()

    # Обработка каждой ячейки с названием "Ростовская область"
    for row_index in rostov_row_indexes:
        # Получаем данные справа от ячейки "Ростовская область"
        row_values = sheet_data.iloc[row_index, 1:].tolist()

        # Формируем ключ для словаря
        key = f"{sheet_name}_{row_index}"

        # Сохраняем данные в словаре
        data_dict[key] = row_values
n = 0
# # Вывод словаря
# for i in data_dict:
#     n = n+1
#     print(i)

print(data_dict)

#  в данных есть пропуски, напиши функцию которая заполняет пропущенные значения медианными







