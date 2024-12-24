import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.use('Agg')


# Анализ и предобработка данных
def analyze_data(df):
    print("Основные статистики:")
    print(df.describe())

    # Визуализация зависимости между процессами
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.title("Корреляция между процессами")
    plt.savefig('correlation_heatmap.png')  # Сохранение графика в файл
    plt.close()  # Закрываем фигуру


# Моделирование
def regression_analysis(df):
    # Пример регрессии для прогнозирования времени обработки
    X = df[['Truck unloading mean time', 'Truck loading mean time', 'Order loading mean waiting time']]
    y = df['Order assembling mean time'].dropna()  # Убедитесь, что нет NaN значений

    # Убираем строки, где отсутствуют значения в X или y
    common_indices = X.index.intersection(y.index)

    X = X.loc[common_indices]
    y = y.loc[common_indices]

    model = LinearRegression()
    model.fit(X, y)

    # Прогнозирование
    predictions = model.predict(X)
    df['Predicted assembling time'] = np.nan  # Создаем новый столбец с NaN
    df.loc[common_indices, 'Predicted assembling time'] = predictions  # Заполняем прогнозами

    # Визуализация
    plt.plot(df['Order assembling mean time'], label='Фактическое время')
    plt.plot(df['Predicted assembling time'], label='Предсказанное время', linestyle='--')
    plt.legend()
    plt.title('Сравнение фактического и предсказанного времени сборки заказов')
    plt.xlabel('Индекс')
    plt.ylabel('Время сборки')
    plt.savefig('predicted_vs_actual.png')  # Сохранение графика в файл
    plt.close()  # Закрываем фигуру


# Генерация отчетов и визуализация результатов
def generate_report(df):
    # Генерация базового отчета
    report = df.describe().transpose()
    report.to_csv('report.csv')
    print("Отчет сохранен в report.csv")


if __name__ == '__main__':
    df = pd.read_excel("/home/tango-home/PycharmProjects/FactoryAnalysis/data/data_model.xlsx")
    print(df.columns)

    df.columns = df.columns.str.strip()

    # Запуск анализа
    analyze_data(df)
    regression_analysis(df)
    generate_report(df)