import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import math

# --- KONFIGURACJA ---
folder_z_nowymi_zdjeciami = 'test_real'
sciezka_do_modelu = 'moj_model_fer.h5'

klasy_ang = ['angry', 'happy', 'sad']
tlumaczenie = {'angry': 'ZŁOŚĆ', 'happy': 'RADOŚĆ', 'sad': 'SMUTEK'}

# --- 1. Ładowanie modelu ---
if not os.path.exists(sciezka_do_modelu):
    print("Brak modelu! Najpierw uruchom trening.")
    exit()

print("Ładowanie modelu...")
model = tf.keras.models.load_model(sciezka_do_modelu)

# --- 2. Pobieranie plików ---
pliki = os.listdir(folder_z_nowymi_zdjeciami)
zdjecia = [f for f in pliki if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not zdjecia:
    print("Folder pusty!")
    exit()

print(f"Testowanie {len(zdjecia)} zdjęć...")

# --- 3. Przygotowanie siatki wykresów ---
cols = 4
rows = math.ceil(len(zdjecia) / cols)
plt.figure(figsize=(16, 5 * rows))

# --- 4. Pętla przez zdjęcia ---
for i, plik in enumerate(zdjecia):
    sciezka = os.path.join(folder_z_nowymi_zdjeciami, plik)

    try:
        # KROK A: Ładujemy oryginał do wyświetlenia (KOLOROWY, DUŻY)
        img_display = load_img(sciezka)

        # KROK B: Ładujemy wersję dla SIECI (SZARY, 48x48)
        # load_img zrobi resize i grayscale automatycznie w locie
        img_for_ai = load_img(sciezka, color_mode='grayscale', target_size=(48, 48))

        # Preprocessing dla AI
        img_array = img_to_array(img_for_ai)
        img_array /= 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # KROK C: Predykcja
        prediction = model.predict(img_batch, verbose=0)
        idx = np.argmax(prediction)
        etykieta = tlumaczenie[klasy_ang[idx]]
        pewnosc = np.max(prediction) * 100

        # Kolorki napisów
        kolor_txt = 'black'
        bg_color = 'white'
        if klasy_ang[idx] == 'happy':
            kolor_txt = 'darkgreen'
            bg_color = '#e6fffa'  # Jasny zielony tło
        elif klasy_ang[idx] == 'angry':
            kolor_txt = 'darkred'
            bg_color = '#ffe6e6'  # Jasny czerwony tło
        elif klasy_ang[idx] == 'sad':
            kolor_txt = 'darkblue'
            bg_color = '#e6f0ff'  # Jasny niebieski tło

        # KROK D: Wyświetlanie
        ax = plt.subplot(rows, cols, i + 1)

        # Wyświetlamy ładne, kolorowe zdjęcie
        plt.imshow(img_display)

        # Tytuł z wynikiem
        plt.title(f"{etykieta}\n({pewnosc:.1f}%)",
                  color=kolor_txt, fontsize=14, fontweight='bold',
                  bbox=dict(facecolor=bg_color, edgecolor='none', alpha=0.7))

        plt.axis('off')
        print(f"Zdjęcie: {plik} -> {etykieta}")

    except Exception as e:
        print(f"Błąd pliku {plik}: {e}")

plt.tight_layout()
plt.show()