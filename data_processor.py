import pandas as pd
import numpy as np
import os

def process_data(df):

    df_clean = df.copy()

    # 1 data recovery
    
    # lum
    mask_lum = df_clean['st_lum'].isnull() & df_clean['st_rad'].notnull() & df_clean['st_teff'].notnull()
    df_clean.loc[mask_lum, 'st_lum'] = np.log10(
        (df_clean.loc[mask_lum, 'st_rad']**2) * ((df_clean.loc[mask_lum, 'st_teff'] / 5778)**4)
    )

    # orb
    mask_orbit = df_clean['pl_orbsmax'].isnull() & df_clean['pl_orbper'].notnull() & df_clean['st_mass'].notnull()
    if mask_orbit.any():
        P_years = df_clean.loc[mask_orbit, 'pl_orbper'] / 365.25
        M_star = df_clean.loc[mask_orbit, 'st_mass']
        df_clean.loc[mask_orbit, 'pl_orbsmax'] = (M_star * (P_years**2))**(1/3)

    # mass
    mask_mass = df_clean['pl_bmasse'].isnull() & df_clean['pl_rade'].notnull()
    df_clean.loc[mask_mass, 'pl_bmasse'] = df_clean.loc[mask_mass, 'pl_rade'] ** 2.06

  
    df_final = df_clean.dropna(subset=['st_lum', 'pl_orbsmax', 'pl_bmasse', 'pl_rade', 'st_teff']).copy()


    # 2 Physics


    df_final['pl_density'] = df_final['pl_bmasse'] / (df_final['pl_rade'] ** 3)
    df_final['insolation'] = (10 ** df_final['st_lum']) / (df_final['pl_orbsmax'] ** 2)
    
    # temperature
    mask_no_temp = df_final['pl_eqt'].isnull()
    df_final.loc[mask_no_temp, 'pl_eqt'] = df_final.loc[mask_no_temp, 'st_teff'] * np.sqrt(
        df_final.loc[mask_no_temp, 'st_rad'] / (2 * df_final.loc[mask_no_temp, 'pl_orbsmax'] * 215.032)
    )
    
    # Habitable zone
    def classify_habitability(flux):
        if flux > 1.11: return "Too Hot (Hot Zone)"
        elif flux < 0.36: return "Too Cold (Cold Zone)"
        else: return "Habitable Zone (Goldilocks)"
    df_final['habitable_type'] = df_final['insolation'].apply(classify_habitability)

    # ESI
    def calculate_esi(radius, density, temp):
        if pd.isna(temp): return 0
        r_ref, d_ref, t_ref = 1.0, 1.0, 288.0
        w_r, w_d, w_t = 0.57, 1.07, 5.58
        esi_r = (1 - abs((radius - r_ref) / (radius + r_ref))) ** w_r
        esi_d = (1 - abs((density - d_ref) / (density + d_ref))) ** w_d
        esi_t = (1 - abs((temp - t_ref) / (temp + t_ref))) ** w_t
        return (esi_r * esi_d * esi_t) ** (1/3)

    df_final['ESI'] = df_final.apply(lambda row: calculate_esi(row['pl_rade'], row['pl_density'], row['pl_eqt']), axis=1)


    # 3. Science


    # A. Bio Class (Schulze-Makuch)
    def classify_life(temp_k):
        if pd.isna(temp_k): return "Unknown"
        temp_c = temp_k - 273.15
        if -18 <= temp_c <= 105: return "Complex Life Possible"
        elif -18 <= temp_c <= 122: return "Microbial Life Only"
        else: return "Extreme Environment"
    df_final['Bio_Class'] = df_final['pl_eqt'].apply(classify_life)

    # B. Atmosphere (Zahnle & Catling 2017)
    df_final['v_esc'] = 11.186 * np.sqrt(df_final['pl_bmasse'] / df_final['pl_rade'])
    def check_atmosphere(row):
        if row['v_esc'] < 3.0: return "No Atmosphere (Likely)"
        if row['insolation'] > (row['v_esc'] / 6.0) ** 4: return "Atmosphere Risk (Erosion)"
        else: return "Atmosphere Likely"
    df_final['Atmosphere_Class'] = df_final.apply(check_atmosphere, axis=1)

    # C. Adams 2025 
    def calculate_adams_score(row):
        score = 0.0
        temp_c = row['pl_eqt'] - 273.15
        if not (0 <= temp_c <= 100): return 0.0
        
        score += 1.0
        period = row['pl_orbper']
        if pd.isna(period): return score
        
        if period < 20: score += 2.0 
        else: score += 0.5
        
        if 10 <= period <= 20: score += 1.0
        return score
    
    df_final['Adams_Score'] = df_final.apply(calculate_adams_score, axis=1)


    def categorize_adams(score):
        if score >= 4.0: return "Prime Habitability (Adams 2025)"
        elif score >= 2.0: return "Habitable (Fast Rotator)"
        elif score > 0: return "Marginal (Slow Rotator)"
        else: return "Not Habitable"
        
    df_final['Adams_Category'] = df_final['Adams_Score'].apply(categorize_adams)


    # D. Final Score
    df_final['AstroBiom_Score'] = (df_final['ESI'] * 10) + df_final['Adams_Score']

    print(f"Catalog: {len(df_final)} planets.")
    return df_final

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_FILE = os.path.join(current_dir, "data", "astrobiom_data.csv")
    PROCESSED_FILE = os.path.join(current_dir, "data", "astrobiom_processed.csv")
    
    if os.path.exists(RAW_FILE):
        df_raw = pd.read_csv(RAW_FILE)
        df_result = process_data(df_raw)
        df_result.to_csv(PROCESSED_FILE, index=False)
        
        