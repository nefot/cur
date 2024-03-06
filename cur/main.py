import os

import matplotlib.pyplot as plt
import pandas as pd


class DataReader:
    _xls: object = pd.ExcelFile(os.curdir +'/cur//data.xlsx')
    _data: dict = pd.read_excel(_xls, sheet_name=None)

    def __init__(self, name_region: str = 'Ростовская область'):
        self.name_region: str = name_region
        self._data_dict: dict = {}
        pass

    @staticmethod
    def visualize_and_save_images(data_dict):
        # Создаем папку "img", если она не существует
        if not os.path.exists("img"):
            os.makedirs("img")

        # Проходимся по каждому элементу словаря
        for key, values in data_dict.items():
            years = list(range(2016, 2016 + len(values)))
            # Создаем график
            plt.figure(figsize=(8, 6))
            plt.plot(years, values, marker='o', linestyle='-')
            plt.title(key)
            plt.xlabel('Year')
            plt.ylabel('Value')
            plt.grid(True)
            # Отображаем значения над точками
            for year, value in zip(years, values):
                # Проверяем, что значения конечны
                if isinstance(year, (int, float)) and isinstance(value, (int, float)):
                    plt.text(year, value, str(value), ha='center', va='bottom')

            # Сохраняем график в папку "img"
            file_name = key.replace(" ", "_").replace(",", "").replace(".", "") + ".png"
            file_path = os.path.join("img", file_name)
            plt.savefig(file_path)

            # Закрываем текущий график, чтобы не отображался в консоли
            plt.close()

    # Пример использования функции с вашим словарем data_dict

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
                key = f"{sheet_name}_{row_index+2}"

                # Сохраняем данные в словаре
                self._data_dict[key] = row_values
        return self._data_dict

    @staticmethod
    def outDict(dicts: dict) -> print():
        print(dicts)

    def run(self):
        dicts = self.genDict()
        self.outDict(dicts)
        self.visualize_and_save_images(dicts)


if __name__ == '__main__':
    rostov = DataReader()
    russia = DataReader(name_region="Российская Федерация")
    ufu = DataReader(name_region="Южный федеральный округ")

    # russia.run()
    # ufu.run()
    rostov.run()
