import requests
import pandas as pd
import io
import os
import numpy as np

url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# documentation https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html

def get_data():

    query=query = """
    SELECT 
        pl_name,           
        hostname,          
        discoverymethod,   
        disc_year,         
        pl_rade,           
        pl_bmasse,        
        pl_orbper,         
        pl_orbsmax,        
        pl_orbeccen,       
        pl_eqt,            
        st_mass,           
        st_rad,            
        st_teff,           
        st_lum,            
        st_spectype,       
        sy_dist            
    FROM ps 
    WHERE default_flag = 1
    """

    response=requests.get(url, params={"query":query, "format":"csv"})

    if response.status_code==200:
        df=pd.read_csv(io.StringIO(response.text))

        if not os.path.exists("data"):
            os.makedirs("data")

        df.to_csv("data/astrobiom_data.csv")

    else: 
        print("Server error", response.status_code)



if __name__ == "__main__":
    get_data()