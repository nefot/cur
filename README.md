# Класс DataReader

## Обзор

Класс `DataReader` предназначен для чтения данных из Excel-файла, их обработки и предоставления различных функций, таких
как визуализация и анализ.

## Структура класса

Класс содержит следующие методы:

1. `__init__(name_region: str = 'Ростовская область', visualization=False)`: Инициализирует объект `DataReader` с
   указанным именем региона и флагом визуализации.

2. `visualize_and_save_images(data_dict)`: Визуализирует данные и сохраняет изображения в каталог "img".

3. `describe_value_types(data_dict: dict) -> dict[Any, list]`: Описывает типы значений в словаре данных.

4. `interpolate_missing_values(data_dict: dict[Any, list]) -> dict[Any, list]`: Интерполирует отсутствующие значения в
   словаре данных.

5. `normalization()`: Заглушка для нормализации данных.

6. `multivisualization() -> object`: Заглушка для визуализации многомерных данных.

7. `genDict() -> dict`: Генерирует словарь из Excel-файла.

8. `filter_data(data_dict)`: Фильтрует словарь данных на основе определенных критериев.

9. `convert_to_float64(data_dict)`: Преобразует значения данных в тип `np.float64`.

10. `outDict(dicts: dict) -> print()`: Выводит содержимое словаря.

11. `run() -> dict`: Выполняет обработку данных и возвращает обработанный словарь данных.

## Использование

Чтобы использовать класс `DataReader`:

1. Создайте экземпляр класса с опциональными параметрами.
2. Вызовите метод `run()` для выполнения обработки данных.

Пример:

```python
rostov = DataReader(visualization=True)
rostov.run()
```

## Дополнительные функции

Помимо класса `DataReader`, в проекте также присутствуют заглушки функций `deletion` и `median`, которые не реализованы
в предоставленном коде.

## Информация о проекте

### Запуск проекта

Для запуска проекта необходимо выполнить следующие шаги:

1. Установить зависимости проекта с помощью poetry: 

`poetry install`

2. Запустить файл командой
   1. `poetry shell`
   2. `python cur/main.py`

### Настройки

Проект использует файл конфигурации `pyproject.toml`, в котором указаны зависимости и другие настройки проекта.