import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib
import os
import argparse

matplotlib.use('Agg')


###############################################################################################################
# Developed by: Golubev Max, Bulatnikov Ivan, Matvei Klevcov, Matvey Klevtsov, Danila Pashkov, Daniil Khvatov #
###############################################################################################################

def analyze_data(df, output_dir):
    print("Основные статистики:")
    print(df.describe())

    numerical_df = df.select_dtypes(include=[np.number])
    if not numerical_df.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f")
        plt.title("Корреляция между процессами")
        plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{df.name}.png'))  # Сохранение графика в файл
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

        plt.figure(figsize=(10, 6))
        plt.plot(df.index[mask], y, label='Фактическое время', marker='o')
        plt.plot(df.index[mask], predictions, label='Предсказанное время', linestyle='--', marker='x')

        plt.legend()
        plt.title(f'Сравнение фактического и предсказанного времени сборки заказов для {df.name}')
        plt.xlabel('Индекс')
        plt.ylabel('Время сборки')
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'predicted_vs_actual_{df.name}.png'))  # Сохранение графика в файл
        plt.close()


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
        df = pd.read_excel("/home/tango-home/PycharmProjects/FactoryAnalysis/data/data_model.xlsx", sheet_name=sheet)

        df = df.dropna(axis=1, how='all')

        print(f"Обрабатываемый лист: {sheet}")

        clean_column_names(df)

        df.name = sheet

        analyze_data(df, args.output_dir)

        regression_analysis_other_sheets(df, args.output_dir)

        generate_report(df, args.output_dir)

        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_report = combined_data.describe().transpose()
    combined_report.to_csv(os.path.join(args.output_dir, 'combined_report.csv'))
    print(f"Общий отчет сохранен в {args.output_dir}/combined_report.csv")
