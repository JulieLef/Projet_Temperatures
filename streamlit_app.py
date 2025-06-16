####################### Import des librairies#################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
import pycountry
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import base64
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.io as pio
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize

######################## Import des dataframes NASA et OWID, préprocessing et SARIMAX dans un cache #############################
@st.cache_data
def load_data():
    North_hemisphere = pd.read_csv(r'NH.Ts+dSST.csv', header = 1, na_values = ['***'])
    South_hemisphere = pd.read_csv(r'SH.Ts+dSST.csv', header = 1, na_values = ['***'])
    Global=pd.read_csv(r'GLB.Ts+dSST.csv', header = 1, na_values = ['***'])
    Zone=pd.read_csv(r'ZonAnn.Ts+dSST.csv', index_col = 'Year')
    df_owid=pd.read_csv(r'owid-co2-data.csv')
     ### NASA ###
    Global[['D-N', 'DJF']] = Global[['D-N', 'DJF']].apply(pd.to_numeric, errors='coerce')
    South_hemisphere[['D-N', 'DJF']] = South_hemisphere[['D-N', 'DJF']].apply(pd.to_numeric, errors='coerce')
    North_hemisphere[['D-N', 'DJF']] = North_hemisphere[['D-N', 'DJF']].apply(pd.to_numeric, errors='coerce')
    Global_DJF = round(((Global.loc[Global['Year'] == 1880, 'Jan'].values[0] + Global.loc[Global['Year'] == 1880, 'Feb'].values[0]) / 2), 2)
    North_hemisphere_DJF = round(((North_hemisphere.loc[North_hemisphere['Year'] == 1880, 'Jan'].values[0] + North_hemisphere.loc[North_hemisphere['Year'] == 1880, 'Feb'].values[0]) / 2), 2)
    South_hemisphere_DJF = round(((South_hemisphere.loc[South_hemisphere['Year'] == 1880, 'Jan'].values[0] +  South_hemisphere.loc[South_hemisphere['Year'] == 1880, 'Feb'].values[0]) / 2), 2)
    Global.fillna(Global_DJF, inplace = True)
    North_hemisphere.fillna(North_hemisphere_DJF, inplace = True)
    South_hemisphere.fillna(South_hemisphere_DJF, inplace = True)
    Global['file_name']="Global"
    North_hemisphere['file_name']="Hémisphère Nord"
    South_hemisphere['file_name']="Hémisphère Sud"
    glb=pd.concat([North_hemisphere,South_hemisphere,Global])                                        
                          ### OWID ###
    owid=df_owid[df_owid['year'] >= 1880]
    owid=owid.sort_values(by=['country','year'])
    owid = owid.groupby('country', group_keys=False).apply(lambda group: group.ffill())
    owid = owid.fillna(0) 
    iso_codes_valides = {country.alpha_3 for country in pycountry.countries}
    df_pays = owid[owid["iso_code"].isin(iso_codes_valides)]
    df_ML = owid[['country', 'iso_code','year','population','gdp','total_ghg','temperature_change_from_ghg']]
    df_sarimax =df_ML.copy()
    future_years = list(range(2024, 2101))
    all_forecasts = []
    for country in df_sarimax['country'].unique():
      df_pays_ml = df_sarimax[df_sarimax['country'] == country]
      iso = df_pays_ml['iso_code'].iloc[0]
      exog_preds = pd.DataFrame(index=future_years)
      exog_preds['year'] = future_years
      exog_preds['iso_code'] = iso
      exog_preds['country']= country
      for col in ['gdp', 'population', 'total_ghg']:
        colonnes = df_pays_ml[col]
        model = ARIMA(colonnes, order=(1,1,1))
        result = model.fit()
        forecast = result.forecast(steps=len(future_years))
        exog_preds[col] = forecast.values
      endog = df_pays_ml['temperature_change_from_ghg']
      exog = df_pays_ml[['total_ghg', 'gdp', 'population']]
      exog_future = exog_preds[['total_ghg', 'gdp', 'population']]
      sarimax_model = SARIMAX(endog, exog=exog, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), trend='c')
      sarimax_result = sarimax_model.fit(disp=False) 
      start = len(endog)
      end = start + len(future_years) - 1
      forecast_temp = sarimax_result.predict(start=start, end=end, exog=exog_future)
      df_forecast = pd.DataFrame({'country': country ,'iso_code': iso ,'year': future_years, 'temperature_change_from_ghg': forecast_temp})
      df_forecast = df_forecast.merge(exog_preds, on=['country', 'iso_code', 'year'])
      all_forecasts.append(df_forecast)
      df_final_forecasts = pd.concat(all_forecasts, ignore_index=True)
      df_final = pd.concat([df_sarimax, df_final_forecasts], ignore_index=True)
    return North_hemisphere, South_hemisphere, Global, Zone, df_owid, df_final, glb, owid, df_pays, iso_codes_valides
North_hemisphere, South_hemisphere, Global, Zone, df_owid, df_final, glb, owid, df_pays, iso_codes_valides = load_data()            

############################# Création de la structure du streamlit###############################
# Gestion des images
def get_base64_im(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
# Mise en page et style de titres et fond d'écran
color_palette = ["#384454", "#317AC1", "#E1A624", "#AD956B", "#D4D3DC", "#68A691", "#C94C4C", "#6E4B9E", "#4BA3C3"]
def add_custom_style(): 
    st.markdown(f"""
        <style>
        .stApp {{background-color: #DADBDA;}}
        h1 {{color: {color_palette[0]};text-align: center;font-weight: 800;font-size: 3em;}}
        h2 {{color: {color_palette[1]};text-align: center;font-weight: 700;font-size: 2em;}}
        h3 {{color: {color_palette[5]};text-align: center;font-weight: 600;font-size: 1.5em;}}
        h4 {{color: {color_palette[3]};text-align: center;font-size: 1.1em;}}
        p {{color: {color_palette[0]};font-size: 1.1em;}}
        /* Bannière des titres */
        .banner-container {{position: relative;width: 100%;border-radius: 20px;}}
        .banner-image {{width: 100%;height: auto;display: block;border-radius: 20px;}}
        .banner-overlay {{position: absolute;top: 0; left: 0; border-radius: 20px; width: 100%; height: 100%;background-color: rgba(255, 255, 255, 0.6);display: flex;align-items: center;justify-content: center;}}
        .banner-title {{{color_palette[0]}; font-size: 2.5em; font-weight: 800;text-align: center; padding: 10px 20px; border-radius: 20px; }}
        /* Bas de sidebar */
        section[data-testid="stSidebar"] {{ position: relative;  }}
        .sidebar-footer {{  text-align: right; font-size: 12px; margin-top: 20px; margin-bottom: 10px;}}
        .sidebar-footer img {{width: 80px;height: auto;margin-bottom: 5px;  }}
        </style>
    """, unsafe_allow_html=True)
# Appliquer les styles
add_custom_style()
# Mise en page unique des graphiques
def set_global_style():
     # Seaborn & Matplotlib
    sns.set_theme(style="whitegrid", palette=color_palette)
    plt.rcParams.update({ 'font.family': 'sans-serif',  'font.size': 12,  'axes.titlesize': 16, 'axes.titleweight': 'normal', 'axes.labelsize': 14, 'axes.labelweight': 'normal',      # 🚫 LABELS X/Y non gras
        'axes.edgecolor': '#000000','axes.labelcolor': '#000000', 'xtick.labelsize': 12, 'ytick.labelsize': 12,  'xtick.color': '#000000',    'ytick.color': '#000000',
        'legend.fontsize': 12,  'figure.titlesize': 18, 'figure.titleweight': 'bold', 'text.color': '#000000' })

    # Plotly
    pio.templates["custom"] = pio.templates["plotly_white"].update({  "layout": { "font": { "family": "sans-serif", "size": 12, "color": "#000000" },
            "title": { "x": 0.5, "xanchor": "center","font": { "family": "sans-serif", "size": 18, "color": "#000000" } },
            "xaxis": {"title": { "font": {"family": "sans-serif", "size": 14, "color": "#000000" }},
            "tickfont": {"color": "#000000" }},
            "yaxis": {"title": {"font": {"family": "sans-serif", "size": 14,"color": "#000000"  }},
            "tickfont": { "color": "#000000"  }},
            "legend": {"font": {"family": "sans-serif","size": 12,"color": "#000000" }},
            "colorway": color_palette } })
    pio.templates.default = "custom"
set_global_style()
#  Image de bandeau
image_path= r'bandeau streamlit_2.png'
image_base64 = get_base64_im(image_path)
st.markdown(
    f"""
    <div class="banner-container">
        <img src="data:image/png;base64,{image_base64}" class="banner-image" alt="Bandeau">
        <div class="banner-overlay">
            <div class="banner-title">Projet analyse des températures terrestres</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True)
# Affichage du contenu de la sidebar
image_path = r'planète.jpg'
image_base64 = get_base64_im(image_path)
with st.sidebar:
   st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{image_base64}" style="width: 100%; opacity: 0.6;border-radius: 20px;" />
    </div>
    """, unsafe_allow_html=True)
   st.header("Menu")
   pages=["Introduction et exploration des datasets", "DataVisualisation", "Modélisation", "Conclusion"]
   page=st.sidebar.radio("Allez vers", pages)
   image_path = r'datascientest.jpg'
   image_base64 = get_base64_im(image_path)
   st.markdown(f"""
        <div class="sidebar-footer">
            <div>Jennifer GAUTHIER</div>
            <div>Julie LEFEBVRE</div>
            <div>Antoine DIACRE</div>
             <img src="data:image/png;base64,{image_base64}" alt="Logo" />
        </div>
    """, unsafe_allow_html=True)
############## Page 1 Contenu de la page Introduction et exploration des datasets ##############
if page == pages[0] : 
    st.markdown("## Introduction")
    st.write("L’objectif de cette étude consiste à constater le réchauffement climatique global à l’échelle de la planète depuis 1880, d’en observer les nuances par zones géographiques et à l’échelle du temps.")
    st.write('Nous étudierons les liens éventuels entre les émissions de différents gaz avec l’analyse faite précédemment.')
    st.markdown("## Exploration des datasets")
    choix_df = ['NASA: Datasets par hémisphère et Global', 'NASA: Dataset par zones', 'OWID: Dataset']
    option_df = st.selectbox('Quel(s) tableau(x) souhaitez-vous afficher?', choix_df)
    st.write('Le dataset étudié est ', option_df)
    if option_df == 'NASA: Datasets par hémisphère et Global':
        col1, col2 = st.columns([1, 4])  # Largeur relative : 4/5 pour le texte, 1/5 pour l'image
        with col1:
         image = Image.open(r'nasa.png')  
         st.image(image, width=100)  
        with col2:
          st.write("### Exploration des datasets de la NASA") 
        st.dataframe(Global.head(10))
        st.markdown("## 🔧 Étapes de Prétraitement")
        st.markdown("Les fichiers ne comportent pas de `NaN` (valeurs manquantes) mais des *** dans les colonnes   `D-N` et `DJF` sur l'année 1880 ")
        st.markdown("### 1️⃣ Conversion des colonnes `D-N` et `DJF` du format 'Object' au format numérique")
        if st.checkbox("👀 Voir l’aperçu du code", key="code_preview_1"):
            st.code("""
South_hemisphere[['D-N', 'DJF']] = South_hemisphere[['D-N', 'DJF']].apply(pd.to_numeric, errors='coerce')
North_hemisphere[['D-N', 'DJF']] = North_hemisphere[['D-N', 'DJF']].apply(pd.to_numeric, errors='coerce')
Global[['D-N', 'DJF']] = Global[['D-N', 'DJF']].apply(pd.to_numeric, errors='coerce')
""", language='python')
        st.markdown("> Les valeurs non convertibles (comme l'année 1880) sont transformées en `NaN` (valeurs manquantes).")
        st.markdown("### 2️⃣ Gestion des valeurs manquantes par la moyenne des mois de Janvier et Février 1880")
        if st.checkbox("👀 Voir l’aperçu du code", key="code_preview_2"):
            st.code("""
Global_DJF = round(((Global.loc[1880, 'Jan'] + Global.loc[1880, 'Feb']) / 2),2)
North_hemisphere_DJF = round(((North_hemisphere.loc[1880, 'Jan'] + North_hemisphere.loc[1880, 'Feb']) / 2),2)
South_hemisphere_DJF = round(((South_hemisphere.loc[1880, 'Jan'] + South_hemisphere.loc[1880, 'Feb']) / 2),2)
Global.fillna(Global_DJF, inplace = True)
North_hemisphere.fillna(North_hemisphere_DJF, inplace = True)
South_hemisphere.fillna(South_hemisphere_DJF, inplace = True)
""")
        st.markdown("> Les `NaN` sont remplacés par la moyenne des valeurs des colonnes 'Jan' et 'Feb' pour l'année 1880.")
        st.markdown("### 3️⃣ Ajout d'une colonne pour identifier la provenance des données")
        if st.checkbox("👀 Voir l’aperçu du code", key="code_preview_3"):
            st.code("""
Global['file_name'] = "Global"
North_hemisphere['file_name'] = "Hémisphère Nord"
South_hemisphere['file_name'] = "Hémisphère Sud"
""")
        st.markdown("### 4️⃣ Fusion des trois datasets")
        if st.checkbox("👀 Voir l’aperçu du code", key="code_preview_4"):
            st.code("""
glb = pd.concat([North_hemisphere, South_hemisphere, Global])
""")
        st.markdown("> Cela permet de faire des **visualisations groupées** ou comparatives.")
    elif option_df == 'NASA: Dataset par zones':
        col5, col6 = st.columns([1, 4])
        with col5:
         image = Image.open(r'nasa.png')  
         st.image(image, width=100)  
        with col6:
          st.write("### Exploration des datasets de la NASA")  
        st.dataframe(Zone.head(10))
        col1, col2 = st.columns(2)
        with col1:
          st.markdown("**🎯 Aucune valeur manquante**")
          st.metric(label="Valeurs manquantes", value=Zone.isna().sum().sum())
        with col2:
          st.markdown("**🧬 Colonnes bien typées**")
          st.dataframe(pd.DataFrame(Zone.dtypes, columns=["Type"]).reset_index().rename(columns={"index": "Colonne"}))
    elif option_df == 'OWID: Dataset':
        col7, col8 = st.columns([1, 4])
        with col7:
          image = Image.open(r'OWID.png')  
          st.image(image, width=100) 
        with col8:
          st.write("### Exploration du dataset OWID")  
        st.dataframe(df_owid.head(10))
        st.markdown("## 🔧 Étapes de Prétraitement")
        st.markdown("### 1️⃣ Filtrage temporel : données depuis 1880")
        if st.checkbox("👀 Voir l’aperçu du code", key="code_preview_5"):
            st.code("owid = df_owid[df_owid['year'] >= 1880]")
        st.markdown("📆 On conserve uniquement les données pertinentes pour notre étude temporelle en cohérence avec les données de la NASA")
        st.markdown("### 2️⃣ Tri des données par pays et année")
        if st.checkbox("👀 Voir l’aperçu du code", key="code_preview_6"):
            st.code("owid = owid.sort_values(by=['country', 'year'])")
        st.markdown("🔀 Cela garantit un ordre cohérent avant de remplir les valeurs manquantes.")
        st.markdown("### 3️⃣ Remplissage intelligent des valeurs manquantes")
        if st.checkbox("📊 Visualiser les NA avant le nettoyage"):
          na_percentage = df_owid.isna().mean().sort_values(ascending=False) * 100
          top_na = na_percentage[:10]
          fig, ax = plt.subplots(figsize=(12, 4))
          ax.barh(top_na.index, top_na.values, color=color_palette[6])
          ax.set_title("🔴 Pourcentage de valeurs manquantes par colonne (avant nettoyage)")
          ax.set_xlabel("% de valeurs manquantes")
          st.pyplot(fig)
        if st.checkbox("👀 Voir l’aperçu du code", key="code_preview_7"):
            st.code("""
owid = owid.groupby('country', group_keys=False).apply(lambda group: group.ffill())
owid = owid.fillna(0)
    """)
        with st.expander("🧠 Pourquoi ce choix ?"):
          st.markdown("""
        - On remplit d'abord les **valeurs manquantes avec les valeurs précédentes** pour chaque pays (`ffill`).
        - Ensuite, les **NA restants** (début de série) sont mis à **0**.
        """)
        st.markdown("### 4️⃣ Filtrage des pays  via codes ISO")
        if st.checkbox("👀 Voir l’aperçu du code", key="code_preview_8"):
            st.code("""
iso_codes_valides = {country.alpha_3 for country in pycountry.countries}
df_pays = owid[owid["iso_code"].isin(iso_codes_valides)]
    """)
        st.markdown("🌐 On conserve un dataframe contenant les données de World et un second dans lequel on exclut les entités non reconnues comme pays (ex: continents, zones économiques...).")
        if st.checkbox("✅ Aperçu du DataFrame final (`df_pays`)"):
          st.dataframe(df_pays.head(10))
############################# Page 2 Contenu de la page Dataviz #########################
if page == pages[1] : 
  st.write("## Datavisualisation")
##### EVOLUTION DE LA TEMPERATURE DEPUIS 1880 ######
  st.write("### Evolution de la température depuis 1880")
  with st.expander("### 🌡️ Evolution de la température depuis 1880 (cliquez pour développer)", expanded=True):
    tab1, tab2, tab3 = st.tabs(["📈 Variation annuelle des températures", "📦 Variation des températures par zone", "📦 Variation des températures par décennie"])
   # GRAPHIQUE 1 : Évolution des températures
    with tab1:
      fig, ax = plt.subplots(figsize=(12, 5))
      ax.plot(Global['Year'], Global['J-D'], label="Global")
      ax.plot(North_hemisphere['Year'], North_hemisphere['J-D'], label="Hémisphère Nord")
      ax.plot(South_hemisphere['Year'], South_hemisphere['J-D'], label="Hémisphère Sud")
      ax.set_title("Variation annuelle des températures")
      ax.set_xlabel("Années")
      ax.set_ylabel("Anomalie de température (°C)")
      ax.legend(loc='best')
      st.pyplot(fig)
  # GRAPHIQUE 2 : Boxplot par zone
    with tab2:
      fig = go.Figure()
      fig.add_trace(go.Box(y=Global['J-D'], name='Global'))
      fig.add_trace(go.Box(y=North_hemisphere['J-D'], name='Hémisphère Nord'))
      fig.add_trace(go.Box(y=South_hemisphere['J-D'], name='Hémisphère Sud'))
      fig.update_layout(title={'text': "Répartition des anomalies par zone géographique", 'x': 0.5,'xanchor': 'center'},yaxis_title="Anomalie de température (°C)", boxmode='group', height=300, margin=dict(t=50, b=50, l=40, r=40))
      st.plotly_chart(fig, use_container_width=True)
 # GRAPHIQUE 3 : Boxplot par décennie
    with tab3:
      glb['Decennie'] = (glb['Year'] // 10) * 10
      fig, ax = plt.subplots(figsize=(10, 5))
      glb.boxplot(column='J-D', by='Decennie', ax=ax, color=color_palette[1])
      ax.set_title('Répartition des variations de températures par décennie')
      ax.set_ylabel('Anomalie de température (°C)')
      plt.suptitle('')
      plt.tight_layout()
      st.pyplot(fig)
##### SAISONNALITE DU RECHAUFFEMENT CLIMATIQUE ######
  st.write("### Saisonnalité du réchauffement climatique")
  with st.expander("### 🌡️ Saisonnalité du réchauffement climatique (cliquez pour développer)", expanded=False):
    tab1, tab2, tab3 = st.tabs(["📈 Variation par saison", "📊 Ecart de température par saison", "📊 Ecart de température par mois"])
  # GRAPHIQUE 4 : VARIATION DES TEMPERATURES PAR SAISON 
    with tab1:
      fig, axs = plt.subplots(2, 2, figsize=(12, 12))
      saisons = ['DJF', 'MAM', 'JJA', 'SON']
      titres = ['Décembre/Janvier/Février', 'Mars/Avril/Mai', 'Juin/Juillet/Août', 'Septembre/Octobre/Novembre']
      positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
      for saison, titre, pos in zip(saisons, titres, positions):
        ax = axs[pos]
        sns.lineplot(x=South_hemisphere['Year'], y=South_hemisphere[saison], label='Hémisphère Sud', linewidth=0.6, ax=ax, color=color_palette[1])
        sns.lineplot(x=North_hemisphere['Year'], y=North_hemisphere[saison], label='Hémisphère Nord', linewidth=0.6, ax=ax, color=color_palette[6])
        ax.set_xlim(1880, 2024)
        ax.set_ylim(-1.5, 2)
        ax.set_title(titre)
        ax.set_xlabel('Année')
        ax.set_ylabel('Évolution de la variable')
        ax.grid(True, linestyle='--')
      plt.tight_layout()
      st.pyplot(fig)
  # GRAPHIQUE 5 : VARIATION DES TEMPERATURES PAR SAISON 2
    with tab2:
      df_long = glb.melt(id_vars=['Year', 'file_name'], var_name='Période', value_name='Valeur')
      df_filtre = df_long[df_long['Year'].isin([1880, 2024])]
      df_minmax = df_filtre.pivot(index=['file_name', 'Période'], columns='Year', values='Valeur').reset_index()
      df_minmax['Différence'] = df_minmax[2024] - df_minmax[1880]
      df_minmax['Période'] = df_minmax['Période'].replace({'MAM': 'Printemps','JJA': 'Été','SON': 'Automne', 'DJF': 'Hiver'})
      fig, ax = plt.subplots(figsize=(10, 5))
      sns.barplot(x='Période', y='Différence', hue='file_name', data=df_minmax,order=['Printemps', 'Été', 'Automne', 'Hiver'], ax=ax)
      ax.set_title("Différence moyenne de température entre 1880 et 2024 par saison")
      ax.set_xlabel("Saison")
      ax.set_ylabel("Différence de température en °C")
      ax.legend(title=None, loc="lower left")
      st.pyplot(fig)
  # GRAPHIQUE 6 : VARIATION DES TEMPERATURES PAR MOIS
    with tab3:
      ordre_mois = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
      df_minmax['Période'] = pd.Categorical(df_minmax['Période'], categories=ordre_mois, ordered=True)
      fig, ax = plt.subplots(figsize=(10, 5))
      sns.barplot(x='Période', y='Différence', hue='file_name', data=df_minmax, ax=ax)
      df_global = df_minmax[df_minmax["file_name"] == "Global"]
      sns.lineplot(x='Période', y='Différence', data=df_global, color="red", linewidth=2,label="Tendance générale", ax=ax)
      ax.set_title("Différence de température entre 1880 et 2024 par mois")
      ax.set_xlabel("Mois de l'année")
      ax.set_ylabel("Différence de température en °C")
      ax.legend(title=None, loc="lower left")
      st.pyplot(fig)
##### RECHAUFFEMENT CLIMATIQUE PAR ZONES GEOGRAPHIQUES ######
  st.write("### Réchauffement climatique par zones géographiques")
  with st.expander("### 🌡️ Réchauffement climatique par zone géographique (cliquez pour développer)", expanded=False):
    tab1, tab2, tab3 = st.tabs(["📦 Variation température par zone", "🔵 Analyse des valeurs extrêmes", "🌍 Evolution température par latitude"])
   # GRAPHIQUE 7 : BOXPLOT VARIATION TEMPERATURE PAR ZONE
    Zonnal_2 = Zone.drop(['Glob', 'NHem', 'SHem', '24N-90N','24S-24N', '90S-24S'], axis = 1)
    with tab1:
      fig, ax = plt.subplots(figsize=(10, 5))
      sns.boxplot(data=Zonnal_2, orient = 'h', ax=ax)
      ax.set_title("Répartition des variations de température, par zone, sur la période 1880 - 2024")
      st.pyplot(fig)
  # GRAPHIQUE 8 : REPARTITION DES VALEURS EXTREMES
    Q1 = Zonnal_2.quantile(0.25)
    Q3 = Zonnal_2.quantile(0.75)
    IQR = Q3 - Q1
    borne_basse = Q1 - 1.5 * IQR
    borne_haute= Q3 + 1.5 * IQR
    outliers = Zonnal_2[(Zonnal_2 < borne_basse) | (Zonnal_2 > borne_haute)].dropna(how='all') 
    with tab2:
      fig, ax = plt.subplots(figsize=(10, 5))
      for col in Zonnal_2.columns:
        ax.scatter(x=Zonnal_2.index, y=Zonnal_2[col], color=color_palette[1])
      for col in Zonnal_2.columns:
        outliers_col = outliers[col].dropna()
        ax.scatter(x=outliers_col.index, y=outliers_col, color=color_palette[6], s=100)
      ax.set_title("Analyse des valeurs extrèmes")
      ax.set_xlabel("Année")
      ax.set_ylabel("Anomalie de température")
      st.pyplot(fig)
  # GRAPHIQUE 9 : CARTE EVOLUTION TEMPERATURE PAR ZONE GEO
    with tab3:
      Zone=Zone.reset_index()
      zone_geo=Zone[['Year','24N-90N','24S-24N','90S-24S','64N-90N','44N-64N','24N-44N','EQU-24N','24S-EQU','44S-24S','64S-44S','90S-64S']]
      df_long = zone_geo.melt(id_vars=["Year"], var_name="Zone",value_name="Valeur")
      df_final_2024 = df_long.pivot(index=["Zone"], columns="Year", values="Valeur").reset_index()
      df_final_2024["Différence"] = df_final_2024[2024] - df_final_2024[1880]
      zone_latitudes = {"24N-90N": (24, 90),"24S-24N": (-24, 24),"90S-24S": (-90, -24),"64N-90N": (64, 90),"44N-64N": (44, 64),"24N-44N": (24, 44), "EQU-24N": (0, 24),"24S-EQU": (-24, 0),"44S-24S": (-44, -24),"64S-44S": (-64, -44), "90S-64S": (-90, -64),}
      fig, ax = plt.subplots(figsize=(12, 6))
      m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, ax=ax)
      m.drawcoastlines()
      m.drawcountries()
      norm = Normalize(vmin=df_final_2024["Différence"].min(), vmax=df_final_2024["Différence"].max())
      cmap = plt.cm.inferno_r
      for _, row in df_final_2024.iterrows():
        zone = row["Zone"]
        lat_min, lat_max = zone_latitudes[zone]
        valeur = row["Différence"]
        ax.fill_betweenx([lat_min, lat_max], -180, 180,  color=cmap(norm(valeur)), alpha=0.6)
      sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
      cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05)
      cbar.set_label("Différence de température (°C)")
      cbar.set_ticks([df_final_2024["Différence"].min(), df_final_2024["Différence"].max()])
      plt.title("Carte représentant l'évolution de la température entre 1880 et 2024 par zones géographiques")
      st.pyplot(fig)
##### EVOLUTION TEMPERATURE DANS LE MONDE ######
  st.write("### Evolution de la température dans le monde et Evolution des gaz à effet de serre")
  with st.expander("### 🌡️ Evolution température dans le monde (cliquez pour développer)", expanded=False):
    tab1, tab2, tab3 = st.tabs(["📈 Emissions GES par continent", "📈 Emissions GES par type de gaz", "📈 Impact émissions sur températures"]) 
  # GRAPHIQUE 10 : EMISSIONS DE GES PAR CONTINENT
    continents = ['Africa', 'Europe', 'Asia','North America', 'Oceania', 'South America', 'World']
    df_continents = owid[owid['country'].isin(continents)]
    emission_GES = df_continents[df_continents['country'] != 'World'][['year','country', 'total_ghg']].copy()
    emission_GES.rename(columns={'year' : 'Année','country' : 'Continent','total_ghg': 'Gaz à effet de serre (GES)'}, inplace=True)
    with tab1:
      fig = px.area(emission_GES, x='Année',  y='Gaz à effet de serre (GES)',  color='Continent',  line_group='Continent',  labels={'Gaz à effet de serre (GES)': 'Émissions des GES (en millions de tonnes)'}) 
      fig.update_layout(width=1100, height=400, title={'text':'Émissions des gaz à effet de serre par continent','x': 0.5,'xanchor': 'center'},
      title_font=dict(size=16))
      st.plotly_chart(fig)
  # GRAPHIQUE 11 : EMISSIONS MONDIALE PAR TYPE DE GAZ
    emission = owid[['year','country', 'methane', 'co2_including_luc', 'nitrous_oxide']]
    emission_world= emission[emission['country']=='World']
    rename_dict2 = {'year' : 'Année','methane': 'Méthane (CH4)', 'co2_including_luc': 'Dioxyde de carbone (CO2)', 'nitrous_oxide': "Oxyde d'azote (NO2)"}
    emission_world.rename (columns =rename_dict2, inplace=True)
    emission_world_melted = emission_world.melt(id_vars=['Année'], value_vars=['Dioxyde de carbone (CO2)','Méthane (CH4)', "Oxyde d'azote (NO2)"],var_name="Type d'émission", value_name='Émissions')
    with tab2:
      fig = px.area(emission_world_melted, x='Année', y='Émissions',  color="Type d'émission", labels={'Émissions': 'Émissions (en millions de tonnes)'})
      fig.update_layout(xaxis=dict(tickmode='linear', tick0=emission_world_melted['Année'].min(),  dtick=20), width=1100,  height=400,title={'text': "Émissions mondiales de gaz à effet de serre par type de gaz",'x': 0.5,'xanchor': 'center'},
      title_font=dict(size=16))
      st.plotly_chart(fig)
  # GRAPHIQUE 12 : IMPACT DES DIFFERENTS GAZ SUR L EVOLUTION DE LA TEMPERATURE
    with tab3:
      owid_2 = owid.set_index('year')
      fig, ax = plt.subplots(figsize=(8, 5))
      sns.lineplot(data=owid_2, x=owid_2.index, y='temperature_change_from_ch4', label='Méthane', ax=ax)
      sns.lineplot(data=owid_2, x=owid_2.index, y='temperature_change_from_co2', label='CO2', ax=ax)
      sns.lineplot(data=owid_2, x=owid_2.index, y='temperature_change_from_n2o', label='N2O', ax=ax)
      ax.set_xlabel('Année')
      ax.set_xlim(1880, 2024)
      ax.grid(True, linestyle='--')
      ax.set_ylabel("Impact des différents gaz\nsur le réchauffement de la température")
      ax.legend()
      plt.tight_layout()
      plt.savefig('GES.png', dpi=200) 
      st.pyplot(fig)  
##### ZOOM CO2 ######
  st.write("### Zoom sur le C02")
  with st.expander("### ♻️ Zoom sur le Co2 (cliquez pour développer)", expanded=False):
    tab1, tab2 = st.tabs(["📈 Emissions C02 par origine", "🍰 Part C02 par type"]) 
  # GRAPHIQUE 13 :  EMISSION DE C02 PAR SECTEUR D'ORIGINE AU FIL DU TEMPS
    sector = owid[['year','country', 'cement_co2', 'coal_co2', 'consumption_co2', 'flaring_co2', 'gas_co2','land_use_change_co2','oil_co2','other_industry_co2', 'trade_co2']]
    sector_world= sector[sector['country']=='World']
    rename_dict3 = {'year' : 'Année','cement_co2': 'Ciment','coal_co2': 'Charbon', 'consumption_co2': 'Consommation','flaring_co2': 'Torchage','gas_co2':'Gaz','land_use_change_co2':'Changement d\'affectation des terres',
    'oil_co2':'Pétrole','other_industry_co2':'Autres sources industrielles','trade_co2':'Commerce'}
    sector_world.rename (columns =rename_dict3, inplace=True)
    value_vars = [col for col in sector_world.columns if col != 'country']
    sector_world_melted = sector_world.melt(id_vars=['Année'],value_vars=value_vars,var_name='Origine du CO2', value_name='Émissions')
    categories_CO2 = ['Consommation', 'Pétrole', 'Gaz', 'Charbon', 'Commerce', "Changement d'affectation des terres", 'Ciment', 'Torchage', 'Autres sources industrielles']  
    with tab1:
      fig = px.area(sector_world_melted,  x='Année', y='Émissions', color='Origine du CO2',labels={'Émissions': 'Émissions (en millions de tonnes)'},category_orders={'Origine du CO2': categories_CO2[::-1]}, color_discrete_sequence=color_palette) 
      fig.update_layout( legend=dict(traceorder='reversed'),  width=100,  height=400,  title={'text': "Émissions de CO2 par secteur d'origine", 'x': 0.5,'xanchor': 'center'},
      title_font=dict(size=16))
      st.plotly_chart(fig,use_container_width=True)
  # GRAPHIQUE 14 :  PART DES CAUSES D EMISSIONS DE C02
    df_world= owid[owid['country']=='World'] 
    df_cause = df_world[['year','cement_co2', 'coal_co2', 'consumption_co2', 'flaring_co2', 'gas_co2', 'land_use_change_co2','oil_co2','other_industry_co2', 'trade_co2']]
    years = df_cause['year'].unique()
    rename_dict4 = {'cement_co2': 'Ciment','coal_co2': 'Charbon','consumption_co2': 'Consommation','flaring_co2': 'Torchage','gas_co2':'Gaz','land_use_change_co2':'Changement d\'affectation des terres', 'oil_co2':'Pétrole','other_industry_co2':'Autres sources industrielles','trade_co2':'Commerce'}
    with tab2:
      selected_year = st.slider("Choisissez une année :", min_value=min(years), max_value=max(years), value=2023)
      df_year = df_cause[df_cause['year'] == selected_year].melt(id_vars=['year'], var_name='Origine du CO2', value_name='Emissions')
      df_year['Origine du CO2'] = df_year['Origine du CO2'].replace(rename_dict4)
      fig = go.Figure(data=[go.Pie(labels=df_year['Origine du CO2'], values=df_year['Emissions'],  name=str(selected_year))])
      fig.update_layout(title={'text':f"Cause des émissions de CO2 en {selected_year}", 'x': 0.5,'xanchor': 'center'})
      st.plotly_chart(fig)
  ##### ZOOM POLLUEURS ######
  st.write("### Zoom sur les plus gros pollueurs de la planète")
  with st.expander("### 🚗 Zooms pollueurs (cliquez pour développer)", expanded=False):
    tab1 = st.tabs(["🌍 Zoom Top pollueurs"])
    # GRAPHIQUE 15 :  ZOOM SUR LES PLUS GROS POLLUEURS DE LA PLANETE
    fig = px.choropleth( df_pays,  locations="iso_code",  color="total_ghg", hover_name="country", animation_frame="year", projection="natural earth",color_continuous_scale="inferno_r", labels={"total_ghg": "Émissions de GES (millions de tonnes)"})
    if 'sliders' in fig['layout']:
        for i, step in enumerate(fig['layout']['sliders'][0]['steps']):
            if step['label'] == '2023':
                fig['layout']['sliders'][0]['active'] = i
                break
    fig.update_layout(title={'text': "Evolution des émissions de gaz à effet de serre à travers le monde", 'x': 0.5,'xanchor': 'center'})
    st.plotly_chart(fig)
##### LIEN GES/ AUTRES VARIABLES ######
  st.write("### Lien entre émissions de GES et autres variables")
  with st.expander("### 🏭 Liens GES et autres variables (cliquez pour développer)", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["🔵 Lien émissions et démographie", "🌍 Emissions GES par habitant", "📈 Lien évol PIB mondial et évol C02", "🔵 Emissions Co2, consommation et PIB" ]) 
  # GRAPHIQUE 16 :  LIEN ENTRE EVOLUTION EMISSIONS ET NOMBRE HABITANTS
    with tab1:
      fig1 = px.scatter(x=df_continents['year'], y=df_continents['methane'],   color=df_continents['country'],   size=df_continents['population'],   labels={'x': 'Année', 'y': 'Méthane'})
      fig2 = px.scatter( x=df_continents['year'], y=df_continents['co2_including_luc'], color=df_continents['country'],size=df_continents['population'], labels={'x': 'Année', 'y': 'CO2 (incl. LUC)'})
      fig3 = px.scatter(x=df_continents['year'],y=df_continents['nitrous_oxide'], color=df_continents['country'], size=df_continents['population'],labels={'x': 'Année', 'y': 'Oxyde d\'azote'})
      fig4 = px.scatter(x=df_continents['year'], y=df_continents['total_ghg'],  color=df_continents['country'], size=df_continents['population'],labels={'x': 'Année', 'y': 'Émissions GES'})
      fig = make_subplots( rows=2, cols=2, subplot_titles=[ "Méthane", "CO2 (avec LUC)", "Oxyde d'azote", "Total GES" ])
      for trace in fig1.data:
          fig.add_trace(trace, row=1, col=1)
      for trace in fig2.data:
          trace.showlegend = False
          fig.add_trace(trace, row=1, col=2)
      for trace in fig3.data:
         trace.showlegend = False
         fig.add_trace(trace, row=2, col=1)
      for trace in fig4.data:
         trace.showlegend = False
         fig.add_trace(trace, row=2, col=2)
      fig.update_layout(title={'text': "Lien entre émissions de gaz et population (par type de gaz)", 'x': 0.5,'xanchor': 'center'}, annotations=[dict(font_size=12) for _ in fig.layout.annotations],showlegend=True)
      st.plotly_chart(fig, use_container_width=True)
  # GRAPHIQUE 17 :  EVOLUTION GES PAR HABITANT PAR PAYS
    GES_hab = df_pays[['year', 'iso_code', 'country', 'ghg_per_capita']]
    rename_dict5 = { 'year' : 'Année', 'iso_code' : 'iso_code', 'country': 'Pays','ghg_per_capita': 'GES par habitant'}
    GES_hab.rename (columns =rename_dict5, inplace=True)
    with tab2:
      fig = px.choropleth( GES_hab,  locations="iso_code",  color="GES par habitant", hover_name="Pays", animation_frame="Année", projection="natural earth",color_continuous_scale="inferno_r", labels={"GES par habitant": "Émissions de GES par habitant (tonnes)"})
      if 'sliders' in fig['layout']:
        for i, step in enumerate(fig['layout']['sliders'][0]['steps']):
            if step['label'] == '2023':
                fig['layout']['sliders'][0]['active'] = i
                break
      fig.update_layout(title={'text': "Émissions de GES par habitant par pays au fil des années", 'x': 0.5,'xanchor': 'center'})
      st.plotly_chart(fig) 
  # GRAPHIQUE 18: TAUX D AUGMENTATION DU PIB MONDIAL ET DES EMISSIONS DE CO2
    df_pib= df_pays.groupby('year')[['gdp', 'co2_including_luc']].sum().reset_index()
    df_pib = df_pib.sort_values(by='year')
    df_pib['gdp_var'] = df_pib['gdp'].pct_change() * 100
    df_pib['co2_including_luc_var'] = df_pib['co2_including_luc'].pct_change() * 100
    with tab3:
      fig, ax1 = plt.subplots(figsize=(10, 5))
      sns.lineplot(data=df_pib, x='year', y='gdp_var', ax=ax1,  label='PIB', color= color_palette[1])
      ax1.set_ylabel("Variation du PIB mondial en %")
      ax1.legend(loc='upper left')
      ax2 = ax1.twinx()
      sns.lineplot(data=df_pib, x='year', y='co2_including_luc_var',ax=ax2,label='Émissions CO2', color= color_palette[6])
      ax2.set_ylabel("Variation des émissions de co2 en %")
      ax1.set_xlabel("Année")
      plt.title("Taux d'augmentation du PIB mondial et des émissions de Co2")
      st.pyplot(fig)
  # GRAPHIQUE 19: EMISSIONS DE CO2 LIEES A LA CONSOMMATION PAR HABITANT PAR RAPPORT AU PIB PAR HABITANT
    PIB_CO2_hab = owid[['year','country', 'consumption_co2_per_capita', 'gdp','population']]
    PIB_CO2_hab = PIB_CO2_hab[(PIB_CO2_hab['country']!='World') & (PIB_CO2_hab['gdp'] >0 ) & (PIB_CO2_hab['year']==2023) ]
    PIB_CO2_hab['PIB_par_habitant'] = PIB_CO2_hab['gdp'] / PIB_CO2_hab['population']
    unique_countries = PIB_CO2_hab['country'].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_countries))
    country_colors = {country: colors(i) for i, country in enumerate(unique_countries)}
    sizes = PIB_CO2_hab['population'] / 1000000
    with tab4:
      fig, ax = plt.subplots(figsize=(10, 6))
      for country in unique_countries:
        country_data = PIB_CO2_hab[PIB_CO2_hab['country'] == country]
        ax.scatter(country_data['PIB_par_habitant'],country_data['consumption_co2_per_capita'],s=sizes[country_data.index],c=[country_colors[country]], alpha=0.7)
        if country in ['Iran','Germany', 'France', 'United Kingdom', 'Russia', 'South Africa', 'Japan','South Korea', 'United States', 'United Arab Emirates', 'Nigeria', 'Brazil', 'China',  'India', 'Mexico', 'Tanzania', 'Ethiopia' ]:
          ax.text(country_data['PIB_par_habitant'].values[0], country_data['consumption_co2_per_capita'].values[0], country, fontsize=9,ha='left',va='bottom',color='black') 
      ax.set_xscale('log')
      ax.set_xlabel('PIB par habitant')
      ax.set_ylabel('CO2 par habitant')
      ax.set_title("Émissions de CO2 liées à la consommation par habitant par rapport au PIB par habitant")  
      ax.legend().set_visible(False) 
      plt.tight_layout()
      st.pyplot(fig)
##### Page 3 Contenu de la page Modélisation ##################
# Création de la liste déroulante en fonction du ML choisi
if page == pages[2] : 
  st.write("## Modélisation")
  st.markdown('Pour la suite de nos travaux, nous avons choisi de poursuivre notre analyse sur les données issues du jeu de données OWID, en prenant pour variable cible l’évolution des températures attribuée aux gaz à effet de serre (« temperature_change_from_ghg »).')
  choix_ml = ['Machine learning "classique"', 'Modèle ARIMA', 'Modèle SARIMAX']
  option_ml = st.selectbox('Choix ', choix_ml)

##### Entrainement de nos données##### 
  df_ml=df_pays[['country','year','population','gdp','cement_co2','co2','co2_including_luc','coal_co2','consumption_co2','flaring_co2','gas_co2','land_use_change_co2','methane','nitrous_oxide','oil_co2','other_industry_co2','primary_energy_consumption','total_ghg','trade_co2','temperature_change_from_ghg']]
  data = df_ml.drop('temperature_change_from_ghg', axis=1)
  target = df_ml['temperature_change_from_ghg']
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42) 
  enc = OneHotEncoder(handle_unknown='ignore')
  X_train_enc = enc.fit_transform(X_train)
  X_test_enc = enc.transform(X_test) 
#Application de la régression linéaire
  if option_ml == 'Machine learning "classique"':
    st.write("### Machine learning classique")
    with st.expander("📈 Régression Linéaire", expanded=False):
        model = joblib.load(r"model.pkl")
        col1, col2 = st.columns(2)
        col1.metric("Score entraînement", f"{model.score(X_train_enc, y_train):.2f}")
        col2.metric("Score test", f"{model.score(X_test_enc, y_test):.2f}")
        pred_test=model.predict(X_test_enc)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(pred_test, y_test, color= color_palette[1])
        ax.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color_palette[6])
        ax.set_title("Droite de régression et nuage de points sur le jeu de test")
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Valeurs réelles")
        st.pyplot(fig)
  #Décision Tree Regressor
    with st.expander("🌳 Decision Tree Regressor", expanded=False):
        model_tree = joblib.load(r"model_tree.pkl")
        col1, col2 = st.columns(2)
        col1.metric("Score entraînement", f"{model_tree.score(X_train_enc, y_train):.2f}")
        col2.metric("Score test", f"{model_tree.score(X_test_enc, y_test):.2f}")
        st.markdown("#### Importance des variables")
        feat_importances = pd.DataFrame(model_tree.feature_importances_[:len(data.columns)], 
                                        index=data.columns, 
                                        columns=["Importance"])
        feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        feat_importances.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel("Score")
        st.pyplot(fig)
  #Random Forest Regressor
    with st.expander("🌲 Random Forest Regressor", expanded=False):
        rfr = joblib.load(r"rfr.pkl")
        col1, col2 = st.columns(2)
        col1.metric("Score entraînement", f"{rfr.score(X_train_enc, y_train):.2f}")
        col2.metric("Score test", f"{rfr.score(X_test_enc, y_test):.2f}")
   # Calcul des métriques
    with st.expander("🧠 Calcul des métriques sur le machine learning", expanded=False):
### Régression linéaire
        y_pred_lin_test = model.predict(X_test_enc)
        y_pred_lin_train = model.predict(X_train_enc)
# jeu d'entraînement 
        mae_lin_train = mean_absolute_error(y_train, y_pred_lin_train)
        mse_lin_train = mean_squared_error(y_train, y_pred_lin_train)
        rmse_lin_train = mean_squared_error(y_train, y_pred_lin_train, squared=False)
# jeu de test 
        mae_lin_test = mean_absolute_error(y_test, y_pred_lin_test)
        mse_lin_test = mean_squared_error(y_test, y_pred_lin_test)
        rmse_lin_test = mean_squared_error(y_test, y_pred_lin_test, squared=False)
### DecisionTree
        y_pred_tree_test = model_tree.predict(X_test_enc)
        y_pred_tree_train = model_tree.predict(X_train_enc)
# jeu d'entraînement 
        mae_tree_train = mean_absolute_error(y_train, y_pred_tree_train)
        mse_tree_train = mean_squared_error(y_train, y_pred_tree_train)
        rmse_tree_train = mean_squared_error(y_train, y_pred_tree_train, squared=False)
# jeu de test 
        mae_tree_test = mean_absolute_error(y_test, y_pred_tree_test)
        mse_tree_test = mean_squared_error(y_test, y_pred_tree_test)
        rmse_tree_test = mean_squared_error(y_test, y_pred_tree_test, squared=False)
### RandomForest
        y_pred_rfr_test = rfr.predict(X_test_enc)
        y_pred_rfr_train = rfr.predict(X_train_enc)
# jeu d'entraînement 
        mae_rfr_train = mean_absolute_error(y_train, y_pred_rfr_train)
        mse_rfr_train = mean_squared_error(y_train, y_pred_rfr_train)
        rmse_rfr_train = mean_squared_error(y_train, y_pred_rfr_train, squared=False)
# jeu de test 
        mae_rfr_test = mean_absolute_error(y_test, y_pred_rfr_test)
        mse_rfr_test = mean_squared_error(y_test, y_pred_rfr_test)
        rmse_rfr_test = mean_squared_error(y_test, y_pred_rfr_test, squared=False)

        data = {'MAE train': [mae_lin_train, mae_tree_train, mae_rfr_train],'MAE test':  [mae_lin_test, mae_tree_test, mae_rfr_test],'MSE train': [mse_lin_train, mse_tree_train, mse_rfr_train],
        'MSE test':  [mse_lin_test, mse_tree_test, mse_rfr_test],'RMSE train': [rmse_lin_train, rmse_tree_train, rmse_rfr_train],'RMSE test':  [rmse_lin_test, rmse_tree_test, rmse_rfr_test]}
        df = pd.DataFrame(data, index = ['Régression linéaire', 'Decision Tree', 'Random Forest '])
        st.dataframe(df.head())
    with st.expander("BONUS 🌲 Random Forest Regressor avec normalisation des données", expanded=False):    
       clf = joblib.load(r"clf.pkl")
       y_pred_random_forest = clf.predict(X_test)
       mae_random_forest_test = mean_absolute_error(y_test,y_pred_random_forest)
       mse_random_forest_test = mean_squared_error(y_test, y_pred_random_forest)
       rmse_random_forest_test = np.sqrt(mse_random_forest_test)
       col1, col2, col3 = st.columns(3)
       col1.metric("MAE", f"{mean_absolute_error(y_test,y_pred_random_forest):.4f}")
       col2.metric("MSE", f"{mean_squared_error(y_test, y_pred_random_forest):.4f}")  
       col3.metric("RMSE", f"{np.sqrt(mse_random_forest_test):.4f}") 
       fig, ax = plt.subplots(figsize=(10, 5))
       ax.scatter(y_pred_random_forest, y_test, color=color_palette[1])
       ax.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color=color_palette[6])
       ax.set_title("Droite de régression et nuage de points sur le jeu de test")
       ax.set_xlabel("Prédictions")
       ax.set_ylabel("Valeurs réelles")
       st.pyplot(fig)
  elif option_ml =="Modèle ARIMA":
    st.write("### Modèle ARIMA")
    st.markdown('Nous avons choisi d’adapter notre modèle ARIMA avec des paramètres (p=5, d=1, q=0), adaptés à la structure temporelle des données puis nous avons réalisé des prévisions pour la période 2025-2100.')
    df_world= owid[owid['country']=='World']
    df_arima =df_world[['year','temperature_change_from_ghg']]
    y = df_arima['temperature_change_from_ghg']
    model_fit = joblib.load(r"model_arima.pkl")
    years_future = np.arange(2024, 2101)
    forecast = model_fit.forecast(steps=len(years_future))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_arima['year'], df_arima['temperature_change_from_ghg'], label='Température historique', color=color_palette[1])
    ax.plot(years_future, forecast, label='Extrapolation température (ARIMA)', linestyle='--',color=color_palette[6])
    ax.set_xlabel('Année')
    ax.set_ylabel('Température')
    ax.set_title("Prévision de la température jusque 2100 avec un modèle ARIMA")
    st.pyplot(fig)
  elif option_ml =="Modèle SARIMAX":
    st.write("### Modèle SARIMAX")
    st.markdown("""Le modèle SARIMAX nous permet de réaliser ces trois visualisations :
1. **Carte montrant le changement de température mondiale dû aux GHG jusque 2100**
2. **Carte montrant l'évolution des émissions de GHG jusque 2100**
3. **Prévision de la température mondiale jusque 2100**
  """)
    df_pays = df_final[df_final["iso_code"].isin(iso_codes_valides)]
    fig = px.choropleth(df_pays, locations="iso_code", color="temperature_change_from_ghg",  hover_name="iso_code",animation_frame='year',projection="natural earth",  color_continuous_scale="inferno_r", labels={"temperature_change_from_ghg": "Température en °C"})
    if 'sliders' in fig['layout']:
        for i, step in enumerate(fig['layout']['sliders'][0]['steps']):
            if step['label'] == '2100':
                fig['layout']['sliders'][0]['active'] = i
                break
    fig.update_layout(title={'text': "Evolution de la température à travers le monde jusqu'en 2100",'x': 0.5,'xanchor': 'center',}, title_font=dict(size=16))
    st.plotly_chart(fig) 
    fig = px.choropleth(df_pays, locations="iso_code", color="total_ghg",hover_name="iso_code", animation_frame='year', projection="natural earth", color_continuous_scale="inferno_r", labels={"total_ghg": "Emissions de GHG"})
    if 'sliders' in fig['layout']:
        for i, step in enumerate(fig['layout']['sliders'][0]['steps']):
            if step['label'] == '2100':
                fig['layout']['sliders'][0]['active'] = i
                break
    fig.update_layout(title={'text': "Evolution des émissions de GHG à travers le monde jusqu'en 2100",'x': 0.5,'xanchor': 'center',}, title_font=dict(size=16))
    st.plotly_chart(fig) 
    fig, ax = plt.subplots(figsize=(10, 5))
    df_world=df_final[df_final['country']=="World"]
    df_reel = df_world[df_world['year'] < 2024]
    sns.lineplot(data=df_reel,x='year',y='temperature_change_from_ghg',ax=ax, label='Données réelles', color=color_palette[1])
    df_pred = df_world[df_world['year'] >= 2024]
    sns.lineplot(data=df_pred,x='year', y='temperature_change_from_ghg',ax=ax, label='Prédictions',linestyle='--', color=color_palette[6])
    ax.set_xlabel('Année')
    ax.set_ylabel('Température')
    ax.set_title("Prévision de la température jusque 2100 avec un modèle SARIMAX")
    fig = ax.get_figure()
    st.pyplot(fig)
#################################### PAGE CONCLUSION ###############################3
if page == pages[3] : 
  st.markdown("## Conclusion")
  st.markdown('L’étude montre une hausse mondiale des températures depuis 1880, liée aux émissions de CO₂, surtout des pays industrialisés puis asiatiques. Les projections prévoient +2,5 à +3 °C d’ici 2100. Notre analyse souligne l’importance de considérer les facteurs économiques, démographiques et territoriaux pour comprendre et agir face au changement climatique')
  st.markdown('Notre projet propose des projections climatiques basées sur des données réelles et des modèles simples, proches des scénarios pessimistes du GIEC, tout en reconnaissant que notre analyse pourrait être enrichie par des modèles plus avancés, des scénarios alternatifs et l’intégration de dimensions politiques, géographiques et sociales')
