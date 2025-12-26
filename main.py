import requests
import pandas as pd
import io

def fetch_nasa_exoplanets():
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    query = """
    SELECT 
        pl_name, hostname, discoverymethod, disc_year, 
        pl_rade, pl_bmasse, pl_orbper, pl_orbsmax, pl_orbeccen, pl_eqt,
        st_mass, st_rad, st_teff, st_lum, st_spectype, sy_dist
    FROM ps 
    WHERE default_flag = 1
    """

    
    response = requests.get(base_url, params={"query": query, "format": "csv"})
    
    if response.status_code == 200:
        print("✅")
        return pd.read_csv(io.StringIO(response.text))
    else:
        print(f"❌ {response.text}")
        return None


if __name__ == "__main__":
    df = fetch_nasa_exoplanets()
    if df is not None:
        df.to_csv("nasa_exoplanets.csv", index=False)
        print(f"💾: {len(df)}")
        print(df.head())