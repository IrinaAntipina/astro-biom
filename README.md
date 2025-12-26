# AstroBiom. AI-Powered Habitability Dashboard

**AstroBiom** is a scientific dashboard that evaluates exoplanet habitability using a multi-factor approach. Instead of relying solely on the Earth Similarity Index (ESI), this project integrates **Physics, Unsupervised Machine Learning, and Generative AI** to identify the most promising candidates for life.

---

## Key Features

### 1. Multi-Theory Analysis
Visualizing exoplanet data based on three modern scientific frameworks:
* **Biology:** Temperature limits for complex life (Schulze-Makuch et al., 2020).
* **Atmosphere:** Cosmic Shoreline and atmospheric retention (Zahnle & Catling, 2017).
* **Dynamics:** Spin-Orbit dynamics and climate stability (Adams et al., 2025).

### 2. Machine Learning Validation
* Uses **K-Means Clustering** (Unsupervised Learning) to group planets based on physical parameters (Mass, Radius, Density).
* **Result:** The AI independently identifies a cluster of planets that correlates with high habitability scores, validating the theoretical models.

### 3. AI Astrobiologist (Gemini 2.5)
* Integrated **Google Gemini 2.5 Flash** to act as a virtual research assistant.
* **Generative Reports:** Select any planet, and the AI generates a detailed scientific profile based on the data.
* **RAG (Chat with Papers):** A feature to interact directly with the PDF research papers used in this project. You can ask questions like *"Why are red dwarfs dangerous?"* and get answers cited from the source texts.

---

## Tech Stack

* **Language:** Python 3.12
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (K-Means)
* **Visualization:** Plotly Express
* **Generative AI:** Google Generative AI (Gemini API)
* **PDF Processing:** `pypdf` (for RAG context)

---

## Project Structure

```text
astro-biom/
├── app.py                # Main Streamlit application
├── data/
│   ├── astrobiom_final.csv      # Processed dataset
│   └── astrobiom_processed.csv  # Backup dataset
├── papers/               # PDF Scientific papers for RAG
│   ├── adams_2025.pdf
│   ├── kiang_2007.pdf
│   └── schulze_makuchl_2020.pdf
├── .env                  # API Keys (not included in repo)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation


```

## Scientific Sources
This project relies on data from the NASA Exoplanet Archive and the following papers:

* Adams et al. (2025) - Spin-Orbit Dynamics
* Schulze-Makuch et al. (2020) - Superhabitable Worlds
* Zahnle & Catling (2017) - The Cosmic Shoreline
* Kiang et al. (2007) - Spectral Signatures


## Installation & Setup

1. Clone the repository: git clone [https://github.com/your-username/astro-biom.git](https://github.com/your-username/astro-biom.git)
2. Create a virtual environment (optional but recommended)
3. Install dependencies: pip install -r requirements.txt
4. Set up API Keys: Create a .env file in the root directory and add your Google Gemini API key: GOOGLE_API_KEY=your_api_key_here
5. Run the application: streamlit run app.py
   
## © Author
Irina Antipina | 2025