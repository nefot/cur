import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataReader:
    _xls: object = pd.ExcelFile('C:\\Users\\artyom\\cur\\cur\\data.xlsx')
    _data: dict = pd.read_excel(_xls, sheet_name=None)

    def __init__(self, name_region: str = 'Ростовская область', visualization=False):
        self.name_region: str = name_region
        self._data_dict: dict = {}
        self.vis = visualization
        pass

    @staticmethod
    def visualize_and_save_images(data_dict):
        # Создаем папку "img", если она не существует
        if not os.path.exists("img"):
            os.makedirs("img")

        # Группируем данные по значению после разделителя "_"
        grouped_data = {}
        for key, values in data_dict.items():
            group_key = str(key.split()[1]).split('_')[0]  # Получаем значение после разделителя "_"
            if group_key not in grouped_data:
                grouped_data[group_key] = []
            grouped_data[group_key].append((key, values))

        # Проходимся по каждой группе и создаем изображение для каждой группы
        for group_key, group_data in grouped_data.items():
            num_plots = len(group_data)
            num_cols = min(num_plots, 3)  # Максимальное количество столбцов на изображении
            num_rows = (num_plots - 1) // 3 + 1  # Количество строк на изображении

            # Вычисляем размер изображения
            fig_size = 8  # Размер графика
            width = fig_size * num_cols
            height = fig_size * num_rows

            # Создаем новое изображение для текущей группы с квадратным видом
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
            plt.suptitle(f'Группа {group_key}')

            # Проходимся по каждому ключу в группе и создаем график
            for i, (key, values) in enumerate(group_data):
                row_idx = i // num_cols
                col_idx = i % num_cols

                years = list(range(2016, 2016 + len(values)))
                if num_rows > 1:
                    ax = axes[row_idx, col_idx]
                else:
                    ax = axes[col_idx] if num_cols > 1 else axes

                ax.plot(years, values, marker='o', linestyle='-')
                ax.set_title(key)
                ax.set_xlabel('Year')
                ax.set_ylabel('Value')
                ax.grid(True)

                for year, value in zip(years, values):
                    if np.isnan(value):  # Проверяем, что значение NaN и пропускаем его
                        continue
                    ax.text(year, value, str(value), ha='center', va='bottom')

            # Удаляем лишние оси, если они есть
            for i in range(num_plots, num_rows * num_cols):
                row_idx = i // num_cols
                col_idx = i % num_cols
                if num_rows > 1:
                    ax = axes[row_idx, col_idx]
                else:
                    ax = axes[col_idx] if num_cols > 1 else axes
                ax.axis('off')

            # Сохраняем изображение в папку "img"
            file_name = f"group_{group_key}.png"
            file_path = os.path.join("img", file_name)
            plt.savefig(file_path)
            # Закрываем текущее изображение
            plt.close(fig)

    @staticmethod
    def describe_value_types(data_dict):
        type_array = {}
        for key, values in data_dict.items():
            types = [type(value) for value in values]
            type_array[key] = types
        return type_array

    # Пример использования функции

    def normalization(self):
        pass

    def multivisualization(self) -> object:
        pass

    def genDict(self) -> dict:
        """
        генерирует словарь по exel файлу
        :return: dict{
        {'ЦУР 1_5': [10.1, 11.4, 9.7, 9.6, 10.1, 9.3, 10.1],
        'ЦУР 1_10': [23.4, 24.1, 27.4, 24.5, 22.5, 21.8, 22.3], ...)}
        """
        for sheet_name, sheet_data in self._data.items():
            # Поиск всех ячеек с названием "Ростовская область"
            rostov_cells = sheet_data.apply(
                lambda row: row.astype(str).str.contains(self.name_region, case=False)).any(
                axis=1)

            # Получение индексов строк с названиями "Ростовская область"
            rostov_row_indexes = rostov_cells[rostov_cells].index.tolist()

            # Обработка каждой ячейки с названием "Ростовская область"
            for row_index in rostov_row_indexes:
                # Получаем данные справа от ячейки "Ростовская область"
                row_values = sheet_data.iloc[row_index, 1:].tolist()

                # Формируем ключ для словаря
                key = f"{sheet_name}_{row_index + 2}"

                # Сохраняем данные в словаре
                self._data_dict[key] = row_values
        return self._data_dict

    # 'ЦУР 2_23': [<class 'int'>, <class 'int'>, <class 'numpy.float64'>, <class 'int'>, <class 'numpy.float64'>, <class 'numpy.float64'>, <class 'float'>],
    # 'ЦУР 2_23': [72,             72,               73.0,                    74,               72.0,                       74.0,                   nan],

    @staticmethod
    def filter_data(data_dict):
        filtered_dict = {}
        for key, values in data_dict.items():
            # Проверяем, что не более 30% значений - NaN
            nan_count = sum(1 for value in values if isinstance(value, float) and np.isnan(value))
            if nan_count / len(values) <= 0.3:
                # Проверяем, что более 70% значений имеют тип float
                float_count = sum(1 for value in values if isinstance(value, float))
                if float_count / len(values) >= 0.7:
                    filtered_dict[key] = values
        return filtered_dict

    @staticmethod
    def convert_to_float64(data_dict):
        converted_dict = {}
        for key, values in data_dict.items():
            converted_values = []
            for value in values:
                # Если значение является строкой, заменяем его на NaN
                if isinstance(value, str):
                    converted_values.append(np.nan)
                else:
                    converted_values.append(np.float64(value))
            converted_dict[key] = converted_values
        return converted_dict

    @staticmethod
    def outDict(dicts: dict) -> print():
        for i, x in enumerate(dicts):
            print(x, dicts[x])
        print(dicts)

    def run(self) -> dict:
        dicts = self.convert_to_float64(self.filter_data(self.genDict()))
        self.outDict(dicts)

        print(self.describe_value_types(dicts))
        if self.vis:
            self.visualize_and_save_images(dicts)
        return dicts


def deletion(data):
    pass


def median(dictonary):
    pass


if __name__ == '__main__':
    rostov = DataReader(visualization=True)
    russia = DataReader(name_region="Российская Федерация")
    ufu = DataReader(name_region="Южный федеральный округ")

    deletion(rostov)
    # russia.run()
    # ufu.run()
    rostov.run()
