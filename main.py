import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.use('Agg')


###############################################################################################################
# Developed by: Golubev Max, Bulatnikov Ivan, Matvei Klevcov, Matvey Klevtsov, Danila Pashkov, Daniil Khvatov #
###############################################################################################################

# Анализ и предобработка данных
def analyze_data(df):
    print("Основные статистики:")
    print(df.describe())

    # Выбираем только числовые столбцы для корреляции
    numerical_df = df.select_dtypes(include=[np.number])
    if not numerical_df.empty:
        # Визуализация зависимости между процессами
        plt.figure(figsize=(10, 6))
        sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f")
        plt.title("Корреляция между процессами")
        plt.savefig(f'correlation_heatmap_{df.name}.png')  # Сохранение графика в файл
        plt.close()  # Закрываем фигуру
    else:
        print("Нет числовых данных для анализа корреляции.")


# Проверка и очистка названий столбцов
def clean_column_names(df):
    df.columns = df.columns.str.strip()  # Удаляем лишние пробелы
    print("Названия столбцов после очистки:")
    print(df.columns)


# # Моделирование для Листа 1
# def regression_analysis_sheet1(df):
#     # Пример регрессии для прогнозирования времени обработки
#     X = df[['Truck unloading mean time', 'Truck loading mean time', 'Order loading mean waiting time']]
#     y = df['Order assembling mean time']
#
#     # Проверяем на наличие NaN и убираем их
#     mask = X.notna().all(axis=1) & y.notna()
#     X = X[mask]
#     y = y[mask]
#
#     if not X.empty and len(y) == len(X):
#         model = LinearRegression()
#         model.fit(X, y)
#
#         # Прогнозирование
#         predictions = model.predict(X)
#         df['Predicted assembling time'] = np.nan  # Создаем новый столбец с NaN
#         df.loc[mask.index, 'Predicted assembling time'] = predictions  # Заполняем прогнозами
#
#         # Визуализация
#         plt.figure(figsize=(10, 6))
#         plt.plot(df.index[mask], y, label='Фактическое время', marker='o')
#         plt.plot(df.index[mask], predictions, label='Предсказанное время', linestyle='--', marker='x')
#         plt.legend()
#         plt.title('Сравнение фактического и предсказанного времени сборки заказов для Листа 1')
#         plt.xlabel('Индекс')
#         plt.ylabel('Время сборки')
#         plt.savefig(f'predicted_vs_actual_L1.png')  # Сохранение графика в файл
#         plt.close()  # Закрываем фигуру


# Моделирование для Листов 1, 2, 3 и 4
def regression_analysis_other_sheets(df):
    # Вывод названий столбцов для диагностики
    print("Названия столбцов для регрессии:")
    print(df.columns)

    X = df[['Truck unloading mean time', 'Truck loading mean time', 'Order loading mean waiting time']]
    y = df['Order assembling mean time']

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    if not X.empty and len(y) == len(X):
        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        df['Predicted assembling time'] = np.nan
        df.loc[mask.index, 'Predicted assembling time'] = predictions

        plt.figure(figsize=(10, 6))
        plt.plot(df.index[mask], y, label='Фактическое время', marker='o')
        plt.plot(df.index[mask], predictions, label='Предсказанное время', linestyle='--', marker='x')

        plt.legend()
        plt.title(f'Сравнение фактического и предсказанного времени сборки заказов для {df.name}')
        plt.xlabel('Индекс')
        plt.ylabel('Время сборки')
        plt.grid()
        plt.savefig(f'predicted_vs_actual_{df.name}.png')  # Сохранение графика в файл
        plt.close()


# Генерация отчетов
def generate_report(df):
    report = df.describe().transpose()
    report.to_csv(f'report_{df.name}.csv')
    print(f"Отчет сохранен в report_{df.name}.csv")


if __name__ == '__main__':
    sheets = ["Лист1", "Лист2", "Лист3", "Лист4"]
    all_data = []

    for sheet in sheets:
        df = pd.read_excel("/home/tango-home/PycharmProjects/FactoryAnalysis/data/data_model.xlsx", sheet_name=sheet)

        # Удаление пустых столбцов
        df = df.dropna(axis=1, how='all')

        print(f"Обрабатываемый лист: {sheet}")

        clean_column_names(df)  # Очистка названий столбцов

        df.name = sheet

        analyze_data(df)

        regression_analysis_other_sheets(df)

        generate_report(df)

        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_report = combined_data.describe().transpose()
    combined_report.to_csv('combined_report.csv')
    print("Общий отчет сохранен в combined_report.csv")
