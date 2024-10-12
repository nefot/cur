import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataReader:
    _xls: object = pd.ExcelFile(os.path.join(os.getcwd(), 'data_mod.xlsx'))
    _data: dict = pd.read_excel(_xls, sheet_name=None)

    def __init__(
            self,
            name_region: str = 'Ростовская область',
            visualization=False,
            directory='data_mod.xlsx',
            parser="NEW",
            debug=False):
        """
        :param parser:какую версию парсера использовать
        :param name_region: имя региона/название таблицы
        :param visualization: нужно ли визуализировать данные
        :param directory: директория объекта
        """
        self.debug = debug
        self.parser = parser
        self._xls: object = pd.ExcelFile(directory)
        self._data: dict = pd.read_excel(self._xls, sheet_name=None)

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
    def describe_value_types(data_dict: dict) -> dict[Any, list]:
        type_array: dict[Any, list] = {}
        for key, values in data_dict.items():
            types = [type(value) for value in values]
            type_array[key] = types
        return type_array

    # Пример использования функции

    @staticmethod
    def interpolate_missing_values(data_dict: dict[Any, list]) -> dict[Any, list]:
        global j
        for key, values in data_dict.items():
            # Проходимся по каждому значению
            for i in range(len(values)):
                # Если значение равно NaN, то заполняем его линейной интерполяцией
                if np.isnan(values[i]):
                    prev_val = None
                    next_val = None
                    # Ищем ближайшее предыдущее и следующее не-NaN значение
                    for j in range(i - 1, -1, -1):
                        if not np.isnan(values[j]):
                            prev_val = values[j]
                            break
                    for j in range(i + 1, len(values)):
                        if not np.isnan(values[j]):
                            next_val = values[j]
                            break
                    # Если есть как минимум одно предыдущее и одно следующее значение, выполняем интерполяцию
                    if prev_val is not None and next_val is not None:
                        interpolated_val = prev_val + (next_val - prev_val) * (i - j) / (i - j + 1)
                        values[i] = interpolated_val
                    elif prev_val is not None:
                        values[i] = prev_val
                    elif next_val is not None:
                        values[i] = next_val
        return data_dict

    @staticmethod
    def normalize_data(data_dict):
        normalized_dict = {}
        for key, values in data_dict.items():
            max_val = max(values)
            min_val = min(values)
            # Check if min_val and max_val are equal to avoid division by zero
            if min_val != max_val:
                normalized_values = [(val - min_val) / (max_val - min_val) for val in values]
            else:
                # If min_val and max_val are equal, set all normalized values to 0 or some default value
                normalized_values = [0] * len(values)  # or any default value you prefer
            normalized_dict[key] = normalized_values
        return normalized_dict

    def multivisualization(self) -> object:
        pass

    def genDictNew(self) -> dict:
        result_dict = {}
        first_cell_dict = {}
        for sheet_name, df in self._data.items():
            for index, row in df.iterrows():
                key = f"{sheet_name}_{index + 1}"  # Увеличиваем индекс на 1, чтобы он начинался с 1
                values = row.values.tolist()
                result_dict[key] = values[1:]  # Сохраняем все значения строки, начиная со второй ячейки
                first_cell_dict[key] = values[0]  # Сохраняем значение первой ячейки
        return result_dict, first_cell_dict

    def genDictOld(self) -> dict:
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
    def outDict(dicts:dict) -> print():
        for i, x in enumerate(dicts):
            print(x, str(dicts[x]).replace('\n', ''))
        print(dicts)

    def run(self) -> dict:
        data_num, data_str = self.genDictNew()
        if self.debug:
            self.outDict(data_num)
            print(25 * '-')
            self.outDict(data_str)
            print('')

        dicts = self.convert_to_float64(self.filter_data(data_num))

        # dicts = self.normalize_data(dicts)
        # dicts = self.interpolate_missing_values(dicts)

        if self.debug:
            print(self.describe_value_types(dicts))
        if self.vis and (self.parser == "OLD"):
            self.visualize_and_save_images(dicts)
        elif self.vis and (self.parser == "NEW"):
            self.visualize_and_save_images2(dicts)
        return dicts

    def visualize_and_save_images2(self, data_dict):
        # Создаем папку "img", если она не существует
        if not os.path.exists("img"):
            os.makedirs("img")

        # Создаем общий график для всех данных
        fig, ax = plt.subplots(figsize=(19, 16))

        # Проходимся по каждой группе и добавляем данные на график
        for key, values in data_dict.items():
            years = list(range(2016, 2016 + len(values)))
            ax.plot(years, values, linestyle='-', label=key)

        ax.set_title('Группы данных')
        ax.set_xlabel('Год')
        ax.set_ylabel('Значение')
        ax.grid(True)

        # Сохраняем изображение в папку "img"
        file_path = os.path.join("img", "all_groups.png")
        plt.savefig(file_path)
        # Закрываем текущее изображение
        plt.close(fig)


def deletion(data):
    pass


def median():
    pass


if __name__ == '__main__':
    rostov = DataReader(visualization=True, debug=True)
    russia = DataReader(name_region="Российская Федерация", visualization=True)
    ufu = DataReader(name_region="Южный федеральный округ", visualization=True)

    deletion(rostov)
    russia.run()
    ufu.run()
    rostov.run()
