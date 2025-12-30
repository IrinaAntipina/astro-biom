import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def run_clustering(df):
    

    features = ['pl_bmasse', 'pl_rade', 'pl_density', 'pl_eqt']
    
    X = df[features].dropna()
    
    indices = X.index
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    

    df.loc[indices, 'cluster_id'] = clusters
    


    summary = df.groupby('cluster_id')[features].mean()
    print(summary)
    
   
    cluster_names = {}
    
    for cluster_id, row in summary.iterrows():
        rad = row['pl_rade']
        mass = row['pl_bmasse']
        temp = row['pl_eqt']
        
        if rad > 8.0:
            name = "Gas Giant (Jovian)" 
        elif rad > 3.0:
            name = "Ice Giant (Neptunian)" 
        elif mass > 2000 or temp > 2000:
            name = "Hot Jupiter / Star" 
        else:
            name = "Rocky / Super-Earth" 
            
        cluster_names[cluster_id] = name
    

    df['Planet_Type_ML'] = df['cluster_id'].map(cluster_names)
    
   
    print(df['Planet_Type_ML'].value_counts())
    
    return df

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    

    INPUT_FILE = os.path.join(current_dir, "data", "astrobiom_processed.csv")
    

    OUTPUT_FILE = os.path.join(current_dir, "data", "astrobiom_final.csv")
    
    if os.path.exists(INPUT_FILE):
   
        df = pd.read_csv(INPUT_FILE)
        

        df_final = run_clustering(df)
        

        df_final.to_csv(OUTPUT_FILE, index=False)

    else:
        print("Error: 'astrobiom_processed.csv' not found")
       