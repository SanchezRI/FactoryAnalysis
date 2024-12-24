import pandas as pd


# Загрузка данных из Excel
file_path = '/home/tango-home/PycharmProjects/FactoryAnalysis/data/data_model.xlsx'  # Укажите путь к вашему файлу
sheet_name = 'Лист2'  # Укажите имя листа, если необходимо
df = pd.read_excel(file_path, sheet_name=sheet_name)


# Удаление пустых столбцов
df_cleaned = df.dropna(axis=1, how='all')

# Сохранение результата в новый файл Excel
output_file_path = '/home/tango-home/PycharmProjects/FactoryAnalysis/data/очищенный_файл2.xlsx'  # Укажите путь для сохранения
df_cleaned.to_excel(output_file_path, index=False)
print("Пустые столбцы удалены и данные сохранены в", output_file_path)
