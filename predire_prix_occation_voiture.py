
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# --- Configuration de la page ---
st.set_page_config(page_title="Prédicteur & Gestion de Véhicules", layout="wide")

# --- Constantes de dépréciation dynamique ---
TAUX_BASE_ANNUEL      = 0.01   # 1% par année
PENALITE_PAR_ACCIDENT = 0.003  # 0,3% par accident
PENALITE_PAR_TITRE    = 0.003  # 0,3% par titre non propre

# --- Mappings des valeurs en français ---
MAPPING_CARBURANT = {
    'gasoline': 'essence',
    'diesel':   'diesel',
    'electric': 'électrique',
    'hybrid':   'hybride'
}
MAPPING_TRANSMISSION = {
    'automatic': 'automatique',
    'manual':    'manuelle'
}

# --- Chargement & nettoyage des données ---
@st.cache_data
def charger_et_nettoyer(chemin_csv: str = "used_cars.csv") -> pd.DataFrame:
    df = pd.read_csv(chemin_csv)
    # Extraction numérique
    df['kilometrage'] = (
        df['milage'].str.replace(",", "").str.extract(r"(\d+)").astype(float)
    )
    df['prix'] = (
        df['price'].str.replace("[$,]", "", regex=True).astype(float)
    )
    # Flags numériques
    df['accident']     = df['accident'].str.lower().eq('none reported').map({True: 0, False: 1})
    df['titre_propre'] = df['clean_title'].fillna('no').str.lower().eq('yes').astype(int)
    # Renommage colonnes source
    df = df.rename(columns={
        'brand':     'marque',
        'model':     'modele',
        'model_year':'annee_modele',
        'fuel_type': 'type_carburant',
        'transmission':'transmission'
    })
    # Traduction catégories (pour model et UI)
    df['type_carburant'] = df['type_carburant'].map(MAPPING_CARBURANT).fillna(df['type_carburant'])
    df['transmission']   = df['transmission'].map(MAPPING_TRANSMISSION).fillna(df['transmission'])
    # Feature non linéaire
    df['kilometrage_carre'] = df['kilometrage'] ** 2
    # Nettoyage
    cols = ['marque','modele','annee_modele','kilometrage','kilometrage_carre',
            'type_carburant','transmission','accident','titre_propre','prix']
    return df[cols].dropna()

# --- Préparation des données pour affichage ---
# @st.cache_data
def preparer_affichage(df: pd.DataFrame) -> pd.DataFrame:
    df_aff = df.copy()
    # Traduction drapeaux
    df_aff['Accident']     = df_aff['accident'].map({0: 'Non', 1: 'Oui'})
    df_aff['Titre propre'] = df_aff['titre_propre'].map({0: 'Non', 1: 'Oui'})
    # Renommage colonnes pour UI
    df_aff = df_aff.rename(columns={
        'marque': 'Marque',
        'modele': 'Modèle',
        'annee_modele': 'Année modèle',
        'kilometrage': 'Kilométrage (km)',
        'kilometrage_carre': 'Kilométrage² (km²)',
        'type_carburant': 'Carburant',
        'transmission': 'Transmission',
        'prix': 'Prix ($)'
    })
    # Ordre colonnes UI
    ui_cols = ['Marque', 'Modèle', 'Année modèle', 'Kilométrage (km)', 'Kilométrage² (km²)',
               'Carburant', 'Transmission', 'Accident', 'Titre propre', 'Prix ($)']
    return df_aff[ui_cols]

# --- Entraînement du modèle ---
@st.cache_resource
def entrainer_modele(df: pd.DataFrame):
    X = pd.get_dummies(
        df.drop(columns=['prix','modele']),
        columns=['marque','type_carburant','transmission'],
        drop_first=True
    )
    y = df['prix']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modele = RandomForestRegressor(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, modele.predict(X_test))
    return modele, X.columns.tolist(), mae

# --- Fonction de dépréciation ---
def calcul_depreciation(prix_initial: float,
                        annee_modele:   int,
                        accidents:      int,
                        titre_propre:   int) -> tuple[float, float]:
    age = max(datetime.now().year - annee_modele, 0)
    taux = (TAUX_BASE_ANNUEL * age
            + PENALITE_PAR_ACCIDENT * accidents
            + PENALITE_PAR_TITRE    * (1 - titre_propre))
    prix_net = prix_initial * ((1 - taux) ** age)
    return prix_net, taux

# --- Initialisation session ---
if 'reserved' not in st.session_state:
    st.session_state['reserved'] = []
if 'purchased' not in st.session_state:
    st.session_state['purchased'] = []

# --- Chargement et préparation ---
df = charger_et_nettoyer()
df_aff = preparer_affichage(df)
modele, caracteristiques, mae = entrainer_modele(df)

# --- Menu principal ---
mode = st.sidebar.radio("Mode", [
    "Prédiction manuelle",
    "Prédiction depuis le tableau",
    "Réservation",
    "Achat"
])

# --- Fonctions par mode ---
# 1. Prédiction manuelle
def pred_manuel():
    st.header("Prédiction manuelle de prix d’une voiture d’occasion")
    marque_sel       = st.selectbox("Marque", sorted(df['marque'].unique()))
    annee_sel        = st.slider("Année du modèle", 1990, datetime.now().year, datetime.now().year-5)
    km_sel           = st.selectbox("Kilométrage (km)", list(range(0,300001,2000)), index=25)
    carburant_sel    = st.selectbox("Carburant", sorted(df['type_carburant'].unique()))
    transmission_sel = st.selectbox("Transmission", sorted(df['transmission'].unique()))
    accident_sel     = st.radio("A eu un accident ?", ['Non', 'Oui'])
    titre_sel        = st.radio("Titre propre ?", ['Non', 'Oui'])

    if st.button("Prédire"):
        inp = {
            'annee_modele':       annee_sel,
            'kilometrage':        km_sel,
            'kilometrage_carre':  km_sel ** 2,
            'accident':           1 if accident_sel=='Oui' else 0,
            'titre_propre':       1 if titre_sel=='Oui' else 0
        }
        for col in caracteristiques:
            if col.startswith('marque_'):
                inp[col] = int(col == f"marque_{marque_sel}")
            elif col.startswith('type_carburant_'):
                inp[col] = int(col == f"type_carburant_{carburant_sel}")
            elif col.startswith('transmission_'):
                inp[col] = int(col == f"transmission_{transmission_sel}")
            else:
                inp.setdefault(col, 0)
        X_in = pd.DataFrame([inp])
        brut = modele.predict(X_in)[0]
        net, taux = calcul_depreciation(brut, annee_sel, inp['accident'], inp['titre_propre'])
        age = datetime.now().year - annee_sel
        st.success(f"💰 Prix brut estimé : {brut:,.0f} $")
        st.info(f"📆 Âge du véhicule : {age} ans")
        st.success(f"Prix après dépréciation ({taux*100:.2f}%/an sur {age} ans) : {net:,.0f} $")

# 2. Prédiction depuis le tableau
def pred_tableau():
    st.header("Prédiction depuis le jeu de données")
    st.dataframe(df_aff)
    marque_tab = st.selectbox("Marque", sorted(df['marque'].unique()))
    modele_tab = st.selectbox("Modèle", sorted(df[df['marque']==marque_tab]['modele'].unique()))
    df_filtre  = df[(df['marque']==marque_tab)&(df['modele']==modele_tab)]
    aff_filtre = preparer_affichage(df_filtre)
    st.dataframe(aff_filtre)

    if st.button("Prédire ce véhicule"):
        veh = df_filtre.iloc[0]
        km = float(veh['kilometrage'])
        inp = {'annee_modele':int(veh['annee_modele']), 'kilometrage':km,
               'kilometrage_carre':km**2, 'accident':int(veh['accident']),
               'titre_propre':int(veh['titre_propre'])}
        for col in caracteristiques:
            if col.startswith('marque_'):
                inp[col] = int(col == f"marque_{veh['marque']}")
            elif col.startswith('type_carburant_'):
                inp[col] = int(col == f"type_carburant_{veh['type_carburant']}")
            elif col.startswith('transmission_'):
                inp[col] = int(col == f"transmission_{veh['transmission']}")
            else:
                inp.setdefault(col, 0)
        X_in = pd.DataFrame([inp])
        brut = modele.predict(X_in)[0]
        net, taux = calcul_depreciation(brut, veh['annee_modele'], inp['accident'], inp['titre_propre'])
        age = datetime.now().year - int(veh['annee_modele'])
        st.success(f"💰 Prix brut estimé : {brut:,.0f} $")
        st.info(f"📆 Âge du véhicule : {age} ans")
        st.success(f"Prix après dépréciation ({taux*100:.2f}%/an sur {age} ans) : {net:,.0f} $")

# 3. Réservation
def gestion_reservation():
    st.header("Réservation de voiture")
    st.dataframe(df_aff)
    choix = st.selectbox("Index à réserver", df.index.tolist(), key='reserve_sel')
    if st.button("Réserver"):
        if choix in st.session_state['reserved']:
            st.warning("❗ Cette voiture est déjà réservée.")
        elif choix in st.session_state['purchased']:
            st.warning("❗ Cette voiture a déjà été achetée.")
        else:
            st.session_state['reserved'].append(choix)
            st.success(f"✅ Voiture #{choix} réservée.")
    if st.session_state['reserved']:
        st.subheader("Mes réservations")
        res_df = preparer_affichage(df.loc[st.session_state['reserved']])
        st.dataframe(res_df)
        ann = st.selectbox("Annuler réservation de", st.session_state['reserved'], key='ann_res')
        if st.button("Annuler réservation"):
            st.session_state['reserved'].remove(ann)
            st.success(f"🗑️ Réservation de #{ann} annulée.")

# 4. Achat
def gestion_achat():
    st.header("Achat de voiture")
    st.dataframe(df_aff)
    choix2 = st.selectbox("Index à acheter", df.index.tolist(), key='achat_sel')
    if st.button("Acheter"):
        if choix2 in st.session_state['purchased']:
            st.warning("❗ Cette voiture est déjà achetée.")
        else:
            if choix2 in st.session_state['reserved']:
                st.session_state['reserved'].remove(choix2)
            st.session_state['purchased'].append(choix2)
            st.success(f"✅ Voiture #{choix2} achetée.")
    if st.session_state['purchased']:
        st.subheader("Mes achats")
        ach_df = preparer_affichage(df.loc[st.session_state['purchased']])
        st.dataframe(ach_df)
        ann2 = st.selectbox("Annuler achat de", st.session_state['purchased'], key='ann_ach')
        if st.button("Annuler achat"):
            st.session_state['purchased'].remove(ann2)
            st.success(f"🗑️ Achat de #{ann2} annulé.")

# Dispatch selon le mode
if mode == "Prédiction manuelle":
    pred_manuel()
elif mode == "Prédiction depuis le tableau":
    pred_tableau()
elif mode == "Réservation":
    gestion_reservation()
elif mode == "Achat":
    gestion_achat()

# Signature
st.markdown("""
<div class=\"footer\">
    Réalisé par <strong>SOULEYMANE DAFFE - DATA SCIENTIST</strong>
</div>
""", unsafe_allow_html=True)

