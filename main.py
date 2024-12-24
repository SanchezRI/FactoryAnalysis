import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib
import os
import argparse
import skfuzzy as fuzz
matplotlib.use('Agg')


def analyze_data(df, output_dir):
    print("Основные статистики:")
    print(df.describe())

    numerical_df = df.select_dtypes(include=[np.number])
    if not numerical_df.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f")
        plt.title("Корреляция между процессами")
        plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{df.name}.png'))
        plt.close()
    else:
        print("Нет числовых данных для анализа корреляции.")


def clean_column_names(df):
    df.columns = df.columns.str.strip()
    print("Названия столбцов после очистки:")
    print(df.columns)


def regression_analysis_other_sheets(df, output_dir):
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

        plt.figure(figsize=(15, 6))
        plt.plot(df.index[mask], y, label='Фактическое время', marker='o')
        plt.plot(df.index[mask], predictions, label='Предсказанное время', linestyle='--', marker='x')

        plt.legend()
        plt.title(f'Сравнение фактического и предсказанного времени сборки заказов для {df.name}')
        plt.xlabel('Индекс')
        plt.ylabel('Время сборки')
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'predicted_vs_actual_{df.name}.png'))
        plt.close()


def fuzzy_quality_assessment(df):
    unloading_time = df['Truck unloading mean time'].values
    loading_time = df['Truck loading mean time'].values
    waiting_time = df['Order loading mean waiting time'].values

    # Проверка на наличие отсутствующих значений
    if np.any(np.isnan(unloading_time)) or np.any(np.isnan(loading_time)) or np.any(np.isnan(waiting_time)):
        print("Обнаружены отсутствующие значения в данных. Пропускаем нечеткий анализ.")
        return

    # Определяем диапазоны
    x_quality = np.arange(0, 101, 1)

    # Членства
    unloading_low = fuzz.trimf(x_quality, [0, 0, 50])
    unloading_medium = fuzz.trimf(x_quality, [0, 50, 100])
    unloading_high = fuzz.trimf(x_quality, [50, 100, 100])

    loading_low = fuzz.trimf(x_quality, [0, 0, 50])
    loading_medium = fuzz.trimf(x_quality, [0, 50, 100])
    loading_high = fuzz.trimf(x_quality, [50, 100, 100])

    waiting_low = fuzz.trimf(x_quality, [0, 0, 50])
    waiting_medium = fuzz.trimf(x_quality, [0, 50, 100])
    waiting_high = fuzz.trimf(x_quality, [50, 100, 100])

    # Определяем массив для хранения результатов оценки качества
    quality_assessments = []

    # Применение нечетких правил
    for i in range(len(df)):
        unloading_lvl = fuzz.interp_membership(x_quality, unloading_low, unloading_time[i])
        loading_lvl = fuzz.interp_membership(x_quality, loading_low, loading_time[i])
        waiting_lvl = fuzz.interp_membership(x_quality, waiting_low, waiting_time[i])

        # Применение нечетких правил
        quality_activation_good = np.fmax(unloading_lvl, loading_lvl)
        quality_activation_average = np.fmax(waiting_lvl, unloading_lvl)
        quality_activation_poor = np.fmax(waiting_lvl, loading_lvl)

        # Финальное качество
        aggregated_quality = np.fmax(quality_activation_good,
                                      np.fmax(quality_activation_average, quality_activation_poor))

        # Проверяем длины перед вызовом defuzz
        if len(x_quality) == aggregated_quality:
            quality_result = fuzz.defuzz(x_quality, aggregated_quality, 'centroid')
            quality_assessments.append(quality_result)
        else:
            print(f"Ошибка: длины массивов не совпадают (x_quality: {len(x_quality)}, aggregated_quality: {aggregated_quality}).")
            quality_assessments.append(np.nan)  # Добавляем NaN для отсутствующей оценки

    df['Quality Assessment'] = quality_assessments


def generate_report(df, output_dir):
    report = df.describe().transpose()
    report.to_csv(os.path.join(output_dir, f'report_{df.name}.csv'))
    print(f"Отчет сохранен в {output_dir}/report_{df.name}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Анализ данных с сохранением отчетов в указанной директории.")
    parser.add_argument('output_dir', type=str, help="Директория для сохранения отчетов и графиков")
    args = parser.parse_args()

    # Проверка существования директории
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sheets = ["Лист1", "Лист2", "Лист3", "Лист4"]
    all_data = []

    for sheet in sheets:
        df = pd.read_excel("data/data_model.xlsx", sheet_name=sheet)

        df = df.dropna(axis=1, how='all')

        print(f"Обрабатываемый лист: {sheet}")

        clean_column_names(df)

        df.name = sheet

        analyze_data(df, args.output_dir)

        regression_analysis_other_sheets(df, args.output_dir)

        fuzzy_quality_assessment(df)

        generate_report(df, args.output_dir)

        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_report = combined_data.describe().transpose()
    combined_report.to_csv(os.path.join(args.output_dir, 'combined_report.csv'))
    print(f"Общий отчет сохранен в {args.output_dir}/combined_report.csv")
