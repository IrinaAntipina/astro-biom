import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def run_clustering(df):
    print("🤖 Запускаю Машинное Обучение (K-Means)...")
    
    # 1. Подготовка данных
    # Выбираем физические параметры для кластеризации
    # Мы ищем группы планет, похожих по массе, размеру, плотности и температуре
    features = ['pl_bmasse', 'pl_rade', 'pl_density', 'pl_eqt']
    
    # Создаем рабочий набор данных (удаляем строки, если где-то еще остались пропуски в этих 4 колонках)
    X = df[features].dropna()
    
    # Сохраняем индексы, чтобы потом вернуть метки обратно в главный датафрейм
    indices = X.index
    
    # 2. Масштабирование (Scaling)
    # Это критически важно для K-Means, чтобы Масса (5000) не "задавила" Радиус (1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Кластеризация (ищем 4 группы)
    # Гипотеза: 1. Землеподобные, 2. Нептуны, 3. Гиганты, 4. Горячие планеты
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Записываем результаты обратно в датафрейм
    df.loc[indices, 'cluster_id'] = clusters
    
    # 4. Анализ и Авто-название кластеров
    # Мы смотрим на средние показатели каждой группы, чтобы понять, кто есть кто
    print("\n📊 Средние показатели кластеров:")
    summary = df.groupby('cluster_id')[features].mean()
    print(summary)
    
    # Логика автоматического присвоения имен (Mapping)
    # ИИ дает просто цифры (0, 1, 2, 3). Мы должны дать им понятные имена.
    cluster_names = {}
    
    for cluster_id, row in summary.iterrows():
        rad = row['pl_rade']
        mass = row['pl_bmasse']
        temp = row['pl_eqt']
        
        if rad > 8.0:
            name = "Gas Giant (Jovian)" # Как Юпитер
        elif rad > 3.0:
            name = "Ice Giant (Neptunian)" # Как Нептун
        elif mass > 2000 or temp > 2000:
            name = "Hot Jupiter / Star" # Очень горячие или тяжелые
        else:
            name = "Rocky / Super-Earth" # Каменистые (наша цель!)
            
        cluster_names[cluster_id] = name
    
    # Применяем имена к датафрейму
    df['Planet_Type_ML'] = df['cluster_id'].map(cluster_names)
    
    print("\n🏷 Итоговые типы планет (определены ИИ):")
    print(df['Planet_Type_ML'].value_counts())
    
    return df

if __name__ == "__main__":
    # Определяем пути (чтобы работать из любой папки)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ВХОД: Берем файл, который сделал data_processor.py
    INPUT_FILE = os.path.join(current_dir, "data", "astrobiom_processed.csv")
    
    # ВЫХОД: Создаем финальный файл для Дашборда
    OUTPUT_FILE = os.path.join(current_dir, "data", "astrobiom_final.csv")
    
    if os.path.exists(INPUT_FILE):
        print(f"📂 Читаю файл: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        
        # Запускаем ML
        df_final = run_clustering(df)
        
        # Сохраняем ФИНАЛЬНЫЙ файл
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Финальные данные сохранены: {OUTPUT_FILE}")
        print("🎉 Теперь можно запускать streamlit run app.py!")
    else:
        print("❌ Ошибка: Файл 'astrobiom_processed.csv' не найден.")
        print("💡 Сначала запустите: python data_processor.py")