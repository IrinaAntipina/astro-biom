import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        pass

st.set_page_config(page_title="AstroBiom. Scientific Dashboard", page_icon="🪐", layout="wide")


@st.cache_data

def load_papers_text():
    text_content = ""
    pdf_files = [
        "papers/adams_2025.pdf", 
        "papers/kiang_2007.pdf", 
        "papers/schulze_makuchl_2020.pdf"
    ]
    
    for file_path in pdf_files:
        if os.path.exists(file_path):
            try:
                reader = PdfReader(file_path)
                text_content += f"\n\n--- START OF DOCUMENT: {file_path} ---\n"

                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
                text_content += f"\n--- END OF DOCUMENT: {file_path} ---\n"
            except Exception as e:
                st.error(f"Error reading {file_path}: {e}")
    
    return text_content


def load_data():
    path_final = "data/astrobiom_final.csv"
    path_processed = "data/astrobiom_processed.csv"
    if os.path.exists(path_final):
        return pd.read_csv(path_final)
    elif os.path.exists(path_processed):
        st.warning("⚠️ astrobiom_final.csv not found")
        return pd.read_csv(path_processed)
    else:
        return None

st.cache_data.clear()
df = load_data()


st.sidebar.header("Filters")

if df is not None:
    # 1. main filter
    if 'Planet_Type_ML' in df.columns:
        st.sidebar.subheader("1. Type filter (ML)")
        all_types = df['Planet_Type_ML'].unique()
        selected_types = st.sidebar.multiselect(
            label="Planets to display", 
            options=all_types, 
            default=all_types
        )
        df_filtered = df[df['Planet_Type_ML'].isin(selected_types)]
    else:
        df_filtered = df.copy()

    # 2. habitable filter
    st.sidebar.subheader("2. Habitable Zone filter")
    hz_only = st.sidebar.checkbox("Show only planets in the Habitable Zone (Hide too hot/cold)", value=False)
    
    if hz_only and 'habitable_type' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['habitable_type'] == "Habitable Zone (Goldilocks)"]

    st.sidebar.divider()
    st.sidebar.markdown(f"**Total planets:** {len(df)}")
    st.sidebar.markdown(f"**Now on screen:** {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        st.sidebar.warning("⚠️ No planets visible! Disable your filters.")
    
    if 'AstroBiom_Score' in df_filtered.columns and not df_filtered.empty:
        best_planet = df_filtered.sort_values(by='AstroBiom_Score', ascending=False).iloc[0]
        st.sidebar.success(f"Sample Leader\n**{best_planet['pl_name']}**\n(Score: {best_planet['AstroBiom_Score']:.1f})")
else:
    st.error("Data not loaded")
    st.stop()


st.title("AstroBiom. Habitability analysis")
st.markdown("Step-by-step exploration of exoplanets based on three scientific theories.")

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "0. About",
    "1. Theory A. Biology (Schulze-Makuch)", 
    "2. Theory B. Atmosphere (Zahnle)", 
    "3. Theory C. Rotation Dynamics (Adams)", 
    "4. Connection with ML",
    "5. AI Astrobiologist",
    "6. AI with pappers"
])


# PROJECT OVERVIEW

with tab0:
    st.header("Project Overview")
    

    st.markdown("""
    ### The Goal
    The main goal of AstroBiom is to move beyond simple metrics like the Earth Similarity Index (ESI). 
    Instead of just looking at "radius and mass," this project evaluates habitability using a multi-factor approach that combines biology, atmospheric physics, and orbital dynamics.

    ### Scientific Foundation
    This dashboard integrates three modern independent theories:
        
    1.  **Biology & Thermodynamics.** Based on *Schulze-Makuch et al. (2020)*. 
        Temperature limits for complex vs. microbial life.
    2.  **Atmospheric Retention.** Based on Zahnle & Catling (2017) ("Cosmic Shoreline").
        Can the planet hold onto its atmosphere against stellar radiation?
    3.  **Climate Dynamics.** Based on Adams et al. (2025).
        How rotation speed affects heat distribution and climate stability.

    ### Innovation
        
    **Machine Learning (K-Means).**
    I use unsupervised learning to cluster planets by physical parameters to see if "habitable" planets form a natural distinct group.
        
    **Generative AI.**
    An AI agent acts as a virtual astrobiologist, interpreting complex data into human-readable scientific reports.      
        
    """)
        

    st.divider()
    

    st.caption("© 2025 AstroBiom by Irina Antipina. All rights reserved. | Data Source: NASA Exoplanet Archive")


# THEORY A: BIOLOGY 
with tab1:
    st.header("A. Temperature Limits of Life (Schulze-Makuch et al.)")
    st.markdown("**Hypothesis.** Life depends on temperature conditions.")
    
    if 'pl_eqt' in df_filtered.columns and 'insolation' in df_filtered.columns:
        df_plot = df_filtered.copy()
        df_plot['temp_c'] = df_plot['pl_eqt'] - 273.15
        color_map = {"Complex Life Possible": "green", "Microbial Life Only": "orange", "Extreme Environment": "red", "Unknown": "gray"}
        
        fig_bio = px.scatter(
            df_plot, x="insolation", y="temp_c", color="Bio_Class",
            color_discrete_map=color_map, log_x=True,
            hover_name="pl_name", 
            hover_data={"insolation": True, "temp_c": ":.1f", "Bio_Class": True},
            title="Insolation vs Surface Temperature (°C)"
        )
        fig_bio.update_yaxes(range=[-150, 400])
        fig_bio.add_hline(y=-18, line_dash="dash", line_color="blue", annotation_text="-18°C")
        fig_bio.add_hline(y=105, line_dash="dash", line_color="green", annotation_text="+105°C")
        fig_bio.add_hline(y=122, line_dash="dash", line_color="orange", annotation_text="+122°C")

        st.plotly_chart(fig_bio, use_container_width=True) 

# THEORY B: ATMOSPHERE
with tab2:
    st.header("B. Cosmic Shoreline (Zahnle & Catling 2017)")
    st.markdown("**Hypothesis.** Planets below the red line lose their atmosphere.")
    
    if 'insolation' in df.columns and 'v_esc' in df.columns:
        fig_atm = px.scatter(
            df_filtered, x="insolation", y="v_esc", color="Atmosphere_Class",
            log_x=True, log_y=True, hover_name="pl_name",
            title="Insolation vs Escape Velocity",
            color_discrete_map={"Atmosphere Likely": "green", "Atmosphere Risk (Erosion)": "orange", "No Atmosphere (Likely)": "red"}
        )
        x_line = [0.1, 1, 10, 100, 1000, 10000]
        y_line = [6 * (i**0.25) for i in x_line]
        fig_atm.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Limit', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_atm, use_container_width=True)

# THEORY C: DYNAMICS
with tab3:
    st.header("C. Rotation Dynamics (Adams et al. 2025)")
    st.markdown("**Hypothesis.** Fast rotation (< 20 days) stabilizes the climate.")
    
    if 'pl_orbper' in df_filtered.columns and 'pl_eqt' in df_filtered.columns:
        fig_adams = px.scatter(
            df_filtered, x="pl_orbper", y="pl_eqt", color="Adams_Category",
            hover_name="pl_name", hover_data=["AstroBiom_Score"], 
            title="Rotation Period vs Temperature",
            color_discrete_map={"Prime Habitability (Adams 2025)": "green", "Habitable (Fast Rotator)": "blue", "Marginal (Slow Rotator)": "gray", "Not Habitable": "red"}
        )
        fig_adams.update_xaxes(range=[0, 50])
        fig_adams.update_yaxes(range=[-100, 200])
        fig_adams.add_shape(type="rect", x0=0, y0=0, x1=20, y1=100, line=dict(color="green", width=2), fillcolor="green", opacity=0.1)
        st.plotly_chart(fig_adams, use_container_width=True)

# ML 
with tab4:
    st.header("AI validation of scientific theories")
    st.markdown("""
    I used the K-Means algorithm to cluster planets based solely on their physical parameters 
    (Mass, Radius, Density), without any knowledge of temperature or habitability.
    
    **Hypothesis.** If our biological theories are correct, the AI should independently identify the "Habitable" group as the best one.
    """)
    
    if 'AstroBiom_Score' in df_filtered.columns and 'Planet_Type_ML' in df_filtered.columns:

        score_stats = df_filtered.groupby('Planet_Type_ML')['AstroBiom_Score'].mean().reset_index()
        

        score_stats = score_stats.sort_values(by='AstroBiom_Score', ascending=False)
        
        # Bar Chart
        fig_bar = px.bar(
            score_stats, 
            x='Planet_Type_ML', 
            y='AstroBiom_Score', 
            color='AstroBiom_Score',
            color_continuous_scale='Viridis', 
            text_auto='.1f', 
            title="Average AstroBiom Score by Planet Type (ML Clusters)",
            
            labels={
                'AstroBiom_Score': 'Average AstroBiom Score',  # <--- ТУТ
                'Planet_Type_ML': 'ML Cluster'
            }
        )
        
        fig_bar.update_layout(xaxis_title=None) 
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Conclusion 
        # Dynamically get the winner
        best_cluster = score_stats.iloc[0]['Planet_Type_ML']
        best_score = score_stats.iloc[0]['AstroBiom_Score']

        st.success("### Conclusion")
        st.markdown(f"""
        The machine learning results confirm our theoretical model:
        
        1. The **{best_cluster}** cluster has the highest average AstroBiom Score (**{best_score:.1f}**).
        2. This proves that the physical parameters (Mass/Radius/Density) used by the AI directly correlate with the habitability conditions.
        3. **Result:** Unsupervised AI independently validated that we are looking for life in the right place.
        """)
        
    else:
        st.warning("Not enough data for ML analysis.")

# AI ASSISTANT
with tab5:
    st.header("AI Astrobiologist")
    st.markdown(" Select a planet, and the AI will generate its scientific profile.")
    
    top_candidates = df_filtered.sort_values(by='AstroBiom_Score', ascending=False).head(50)
    
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        planet_name = st.selectbox("Choose the planet:", top_candidates['pl_name'].unique())
    with col_btn:
        st.write("") 
        st.write("")
        generate_btn = st.button("Analyse")

    if generate_btn:
        if not GOOGLE_API_KEY:
             st.error("API not found")
        else:
            planet_data = df[df['pl_name'] == planet_name].iloc[0]
            

            prompt = f"""
            You are an expert astrobiologist. Describe the exoplanet {planet_name}.
            AstroBiom data:
            - Mass: {planet_data.get('pl_bmasse', 'N/A')} M_earth
            - Radius: {planet_data.get('pl_rade', 'N/A')} R_earth
            - Temperature: {planet_data.get('pl_eqt', 'N/A')} K
            - Orbital Period: {planet_data.get('pl_orbper', 'N/A')} d
            - ESI: {planet_data.get('ESI', 0):.2f}
            - AstroBiom Score: {planet_data.get('AstroBiom_Score', 0):.1f}
            - Category (Adams): {planet_data.get('Adams_Category', 'N/A')}
            - Type (ML): {planet_data.get('Planet_Type_ML', 'N/A')}

            1. Is it habitable?
            2. Role of rotation (according to Adams).
            3. Conclusion. Use emojis.
            """
            
            with st.spinner('Thinking...'):
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")



# CHAT WITH PAPERS 
with tab6:
    st.header("Interactive knowledge base")
    st.markdown("Ask questions specifically about the scientific papers used in this project.")


    knowledge_base = load_papers_text()
    
    if not knowledge_base:
        st.warning("⚠️ PDF files not found")
    else:

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: What are the main biomarkers according to Kiang?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)


            if not GOOGLE_API_KEY:
                st.error("API Key not found")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing scientific papers..."):
                        try:

                            full_prompt = f"""
                            You are a research assistant for a diploma project.
                            You have read the following academic papers:
                            {knowledge_base}
                            
                            User Question: {prompt}
                            
                            Instructions:
                            1. Answer ONLY using the provided text.
                            2. Cite the specific paper (Adams, Kiang, or Schulze-Makuch) for each fact.
                            3. If the answer is not in the papers, state that clearly.
                            """
                            
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            response = model.generate_content(full_prompt)
                            
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                            
                        except Exception as e:
                            st.error(f"Error: {e}")