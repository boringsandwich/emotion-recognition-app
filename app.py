import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import random
import glob
from PIL import Image

# Importy do Twojego modelu (TensorFlow)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Import do nowej funkcji kamery (DeepFace)
from deepface import DeepFace

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="System Analizy Emocji", layout="wide")

# ==========================================
# KONFIGURACJA 1: TWÓJ MODEL (BADANIA)
# ==========================================
MODEL_PATH = 'moj_model_fer.h5'
DATASET_TEST_PATH = './dane_fer/test'
DEFAULT_USER_FOLDER = 'test_real'

CLASSES = ['angry', 'happy', 'sad']
TRANSLATION = {'angry': 'ZŁOŚĆ', 'happy': 'RADOŚĆ', 'sad': 'SMUTEK'}
COLORS = {'angry': '#FF4B4B', 'happy': '#2ECC71', 'sad': '#3498DB'}

# ==========================================
# KONFIGURACJA 2: KAMERA (DEEPFACE)
# ==========================================
# Tutaj definiujemy TYLKO te emocje, które nas interesują
TARGET_EMOTIONS = {
    'angry': 'ZŁOŚĆ',
    'happy': 'RADOŚĆ',
    'sad': 'SMUTEK'
}
# Kolory dla wykresów w trybie DeepFace
DEEPFACE_COLORS_HEX = {
    'ZŁOŚĆ': '#FF4B4B',
    'RADOŚĆ': '#2ECC71',
    'SMUTEK': '#3498DB'
}


# --- FUNKCJE POMOCNICZE (TWÓJ MODEL) ---

@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    try:
        # Twój model wymaga 48x48 grayscale
        img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img)
        img_array /= 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception:
        return None


def get_dataset_images(limit=None):
    search_path = os.path.join(DATASET_TEST_PATH, '**', '*.jpg')
    found_files = glob.glob(search_path, recursive=True)
    if not found_files: return []
    if limit is not None and len(found_files) > limit:
        return random.sample(found_files, limit)
    return found_files


# =========================================================
# GŁÓWNY INTERFEJS (SELEKCJA TRYBU)
# =========================================================

st.sidebar.title("🎛️ Panel Sterowania")
app_mode = st.sidebar.selectbox(
    "Wybierz tryb aplikacji:",
    ["📂 Badanie (Mój Model - Oryginał)", "📷 Kamera (Live Foto - DeepFace)"]
)

# ---------------------------------------------------------
# TRYB 1: BADANIE (To jest Twój stary kod)
# ---------------------------------------------------------
if app_mode == "📂 Badanie (Mój Model - Oryginał)":

    st.title("🧠 System Rozpoznawania Emocji (Twój Model)")

    # 1. Ładowanie Modelu
    model = load_ai_model()

    if model is None:
        st.error(f"Nie znaleziono modelu '{MODEL_PATH}'. Uruchom najpierw trening (emotion.py)!")
        st.stop()

    st.sidebar.header("⚙️ Źródło Danych")
    source_option = st.sidebar.radio(
        "Skąd pobrać zdjęcia?",
        ("📂 Własny folder (test_real)", "📚 Zbiór testowy (Dataset)")
    )

    image_paths = []
    current_source_name = ""

    # Logika wyboru plików
    if source_option == "📂 Własny folder (test_real)":
        folder_path = st.sidebar.text_input("Ścieżka do folderu:", value=DEFAULT_USER_FOLDER)
        current_source_name = "Moje zdjęcia"
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_paths = [os.path.join(folder_path, f) for f in files]
        else:
            st.sidebar.warning("Folder nie istnieje.")

    else:  # Dataset
        current_source_name = "Zbiór Testowy (Dataset)"
        if os.path.exists(DATASET_TEST_PATH):
            load_all = st.sidebar.checkbox("Wczytaj WSZYSTKIE dostępne zdjęcia", value=False)
            if load_all:
                image_paths = get_dataset_images(limit=None)
                st.sidebar.warning(f"⚠️ Uwaga: Wczytano {len(image_paths)} zdjęć.")
            else:
                sample_size = st.sidebar.number_input("Liczba losowych zdjęć:", min_value=1, value=200, step=50)
                image_paths = get_dataset_images(limit=sample_size)
                st.sidebar.info(f"Pobrano losowe {len(image_paths)} zdjęć.")
        else:
            st.sidebar.error(f"Nie znaleziono folderu {DATASET_TEST_PATH}.")

    if not image_paths:
        st.warning("Brak zdjęć do analizy.")
        st.stop()

    # --- ANALIZA DANYCH ---
    session_key = f'analysis_{current_source_name}_{len(image_paths)}'

    if session_key not in st.session_state:
        st.session_state[session_key] = None

    if st.session_state[session_key] is None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        has_ground_truth = False
        total_imgs = len(image_paths)

        with st.spinner(f"Przetwarzanie {total_imgs} zdjęć Twoim modelem..."):
            for i, path in enumerate(image_paths):
                if i % (max(1, total_imgs // 20)) == 0:
                    progress_bar.progress(i / total_imgs)
                    status_text.text(f"Analiza obrazu {i + 1}/{total_imgs}")

                processed_img = preprocess_image(path)
                if processed_img is not None:
                    # Predykcja Twoim modelem
                    pred = model.predict(processed_img, verbose=0)[0]
                    idx = np.argmax(pred)
                    label = CLASSES[idx]
                    confidence = np.max(pred)

                    filename = os.path.basename(path)
                    parent_folder = os.path.basename(os.path.dirname(path))
                    true_label_eng = parent_folder if parent_folder in CLASSES else None
                    true_label_pl = TRANSLATION.get(true_label_eng, "-")

                    if true_label_eng: has_ground_truth = True

                    results.append({
                        "Plik": filename,
                        "Prawdziwa_Etykieta": true_label_pl,
                        "Wykryta_Emocja": TRANSLATION[label],
                        "Pewność": confidence,
                        "Raw_Angry": pred[0], "Raw_Happy": pred[1], "Raw_Sad": pred[2],
                        "Ścieżka": path,
                        "Poprawne": true_label_pl == TRANSLATION[label] if true_label_eng else None
                    })
            progress_bar.progress(1.0)
            status_text.empty()

        st.session_state[session_key] = pd.DataFrame(results)
        st.session_state[f"{session_key}_has_gt"] = has_ground_truth

    df = st.session_state[session_key]
    has_gt = st.session_state.get(f"{session_key}_has_gt", False)

    # --- ZAKŁADKI ---
    tab1, tab2, tab3 = st.tabs(["📊 Raporty i Statystyki", "🔍 Przeglądarka (Interakcja)", "🧮 Wnioskowanie Bayesowskie"])

    # TAB 1: RAPORTY
    with tab1:
        st.header(f"Raport dla: {current_source_name}")
        with st.expander("📈 Zobacz historię treningu"):
            if os.path.exists("wykresy_treningu.png"):
                st.image("wykresy_treningu.png", use_container_width=True)
            else:
                st.warning("Brak pliku wykresy_treningu.png")

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(df, names='Wykryta_Emocja', title='Rozkład Wykrytych Emocji',
                             color='Wykryta_Emocja', color_discrete_map={v: COLORS[k] for k, v in TRANSLATION.items()})
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            avg_conf = df.groupby('Wykryta_Emocja')['Pewność'].mean().reset_index()
            fig_bar = px.bar(avg_conf, x='Wykryta_Emocja', y='Pewność', title='Średnia Pewność',
                             color='Wykryta_Emocja', color_discrete_map={v: COLORS[k] for k, v in TRANSLATION.items()})
            fig_bar.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(df[['Plik', 'Prawdziwa_Etykieta', 'Wykryta_Emocja', 'Pewność']], use_container_width=True)

    # TAB 2: PRZEGLĄDARKA
    with tab2:
        st.header("Interaktywna Przeglądarka")
        df_sorted = df.sort_values(by=['Poprawne', 'Pewność'], ascending=[False, False], na_position='last')

        c1, c2 = st.columns(2)
        target = c1.selectbox("Filtruj po emocji:", ["Wszystkie"] + list(TRANSLATION.values()))
        amount = c2.slider("Liczba zdjęć:", 1, len(df), min(20, len(df)))

        if target == "Wszystkie":
            filtered_df = df_sorted.head(amount)
        else:
            filtered_df = df_sorted[df_sorted['Wykryta_Emocja'] == target].head(amount)

        cols = st.columns(5)
        for index, row in filtered_df.iterrows():
            with cols[index % 5]:
                st.image(row['Ścieżka'], use_container_width=True)
                color_style = "green" if row['Poprawne'] else "red"
                if row['Prawdziwa_Etykieta'] == "-": color_style = "black"
                st.markdown(f"**{row['Wykryta_Emocja']}** ({row['Pewność']:.0%})", unsafe_allow_html=True)

    # TAB 3: BAYES
    with tab3:
        st.header("Eksperyment: Wnioskowanie Bayesowskie")
        col_left, col_right = st.columns([1, 2])
        with col_left:
            sel_file = st.selectbox("Wybierz zdjęcie:", df['Plik'])
            row_data = df[df['Plik'] == sel_file].iloc[0]
            st.image(row_data['Ścieżka'], width=200)

            st.markdown("### Kontekst (Prior)")
            p_ang = st.slider("Złość (Kontekst)", 0.0, 1.0, 0.33)
            p_hap = st.slider("Radość (Kontekst)", 0.0, 1.0, 0.33)
            p_sad = st.slider("Smutek (Kontekst)", 0.0, 1.0, 0.33)

            priors = np.array([p_ang, p_hap, p_sad])
            if priors.sum() == 0: priors = np.ones(3)
            priors /= priors.sum()

        with col_right:
            likelihood = np.array([row_data['Raw_Angry'], row_data['Raw_Happy'], row_data['Raw_Sad']])
            posterior = likelihood * priors
            posterior /= posterior.sum()

            fig = go.Figure()
            emotions_list = list(TRANSLATION.values())
            fig.add_trace(go.Bar(x=emotions_list, y=likelihood, name='Model (Oczy)'))
            fig.add_trace(go.Bar(x=emotions_list, y=posterior, name='Bayes (Oczy + Kontekst)'))
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Decyzja Bayesa: **{emotions_list[np.argmax(posterior)]}**")


# ---------------------------------------------------------
# TRYB 2: KAMERA (NOWOŚĆ - DeepFace) - TYLKO SNAPSHOT
# ---------------------------------------------------------
elif app_mode == "📷 Kamera (Live Foto - DeepFace)":

    st.title("📷 Kamera (Analiza Emocji)")
    st.info(
        "Zrób zdjęcie, aby model DeepFace przeanalizował emocje. Wynik zostanie ograniczony tylko do: Złość, Radość, Smutek.")

    img_buffer = st.camera_input("Uśmiechnij się!")

    if img_buffer is not None:
        # 1. Zapisz zdjęcie tymczasowo
        temp_filename = "temp_snap.jpg"
        with open(temp_filename, "wb") as f:
            f.write(img_buffer.getbuffer())

        col_res1, col_res2 = st.columns([1, 1.5])

        with col_res1:
            st.image(temp_filename, caption="Twoje zdjęcie", use_container_width=True)

        with col_res2:
            with st.spinner("Analiza w toku..."):
                try:
                    # DeepFace analizuje zdjęcie (wszystkie emocje)
                    # enforce_detection=False pozwala działać nawet gdy twarz jest niewyraźna
                    res = DeepFace.analyze(temp_filename, actions=['emotion'], enforce_detection=False)

                    if isinstance(res, list): res = res[0]

                    all_emotions = res['emotion']  # np. {'angry': 10, 'happy': 0.1, 'neutral': 80...}

                    # 2. FILTROWANIE (Kluczowy moment)
                    # Wybieramy tylko te 3 emocje, które zdefiniowaliśmy w TARGET_EMOTIONS
                    filtered_scores = {k: all_emotions.get(k, 0) for k in TARGET_EMOTIONS.keys()}

                    # Obliczamy sumę tych trzech, żeby przeliczyć procenty na nowo (żeby sumowały się do 100%)
                    total_score = sum(filtered_scores.values())
                    if total_score == 0: total_score = 1  # Zabezpieczenie przez dzieleniem przez 0

                    # Normalizacja
                    normalized_scores = {k: (v / total_score) for k, v in filtered_scores.items()}

                    # Znalezienie zwycięzcy
                    winner_key = max(normalized_scores, key=normalized_scores.get)
                    winner_pl = TARGET_EMOTIONS[winner_key]
                    winner_conf = normalized_scores[winner_key]

                    # Wyświetlenie wyniku
                    st.success(f"Wykryta emocja: **{winner_pl}**")
                    st.metric("Pewność (wśród badanych 3)", f"{winner_conf:.1%}")

                    # Wykres
                    chart_data = pd.DataFrame({
                        'Emocja': [TARGET_EMOTIONS[k] for k in normalized_scores.keys()],
                        'Prawdopodobienstwo': list(normalized_scores.values())
                    })

                    fig = px.bar(chart_data, x='Emocja', y='Prawdopodobienstwo',
                                 title="Rozkład (Złość vs Radość vs Smutek)",
                                 color='Emocja', color_discrete_map=DEEPFACE_COLORS_HEX)
                    fig.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Wystąpił błąd analizy lub nie wykryto twarzy. Spróbuj ponownie.\nSzczegóły: {e}")