import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import math

# --- KONFIGURACJA ---
folder_z_nowymi_zdjeciami = 'test_real'
sciezka_do_modelu = 'moj_model_fer.h5'

# USTAWIENIA WYŚWIETLANIA
KOLUMNY = 4
WIERSZE_NA_STRONE = 3  # Ile rzędów na jednym ekranie
ZDJEC_NA_STRONE = KOLUMNY * WIERSZE_NA_STRONE  # Np. 12 zdjęć na okno

klasy_ang = ['angry', 'happy', 'sad']
tlumaczenie = {'angry': 'ZŁOŚĆ', 'happy': 'RADOŚĆ', 'sad': 'SMUTEK'}

# --- 1. Ładowanie modelu ---
if not os.path.exists(sciezka_do_modelu):
    print("Brak modelu! Najpierw uruchom trening.")
    exit()

print("Ładowanie modelu...")
try:
    model = tf.keras.models.load_model(sciezka_do_modelu)
except Exception as e:
    print(f"Błąd ładowania modelu: {e}")
    exit()

# --- 2. Pobieranie plików ---
if not os.path.exists(folder_z_nowymi_zdjeciami):
    print(f"Brak folderu {folder_z_nowymi_zdjeciami}")
    exit()

pliki = os.listdir(folder_z_nowymi_zdjeciami)
zdjecia = [f for f in pliki if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not zdjecia:
    print("Folder pusty!")
    exit()

print(f"Znaleziono {len(zdjecia)} zdjęć. Rozpoczynam testowanie...")

# --- 3. Pętla Paginacji (Wyświetlanie stronami) ---
total_images = len(zdjecia)
num_pages = math.ceil(total_images / ZDJEC_NA_STRONE)

for page in range(num_pages):
    start_idx = page * ZDJEC_NA_STRONE
    end_idx = min((page + 1) * ZDJEC_NA_STRONE, total_images)
    batch_zdjecia = zdjecia[start_idx:end_idx]

    print(f"\n--- Strona {page + 1} z {num_pages} (Zdjęcia {start_idx + 1}-{end_idx}) ---")

    # Obliczamy ile rzędów potrzeba na TEJ konkretnej stronie
    current_batch_size = len(batch_zdjecia)
    current_rows = math.ceil(current_batch_size / KOLUMNY)

    # Ustawiamy rozmiar okna (wysokość zależy od liczby rzędów)
    plt.figure(figsize=(16, 5 * current_rows))
    plt.suptitle(f"Wyniki - Strona {page + 1} / {num_pages}", fontsize=20)

    for i, plik in enumerate(batch_zdjecia):
        sciezka = os.path.join(folder_z_nowymi_zdjeciami, plik)

        try:
            # KROK A: Ładujemy oryginał (KOLOR)
            img_display = load_img(sciezka)

            # KROK B: Ładujemy wersję dla AI (GRAYSCALE, 48x48)
            img_for_ai = load_img(sciezka, color_mode='grayscale', target_size=(48, 48))

            # Preprocessing
            img_array = img_to_array(img_for_ai)
            img_array /= 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # KROK C: Predykcja
            prediction = model.predict(img_batch, verbose=0)
            idx = np.argmax(prediction)
            etykieta = tlumaczenie[klasy_ang[idx]]
            pewnosc = np.max(prediction) * 100

            # Kolory
            kolor_txt = 'black'
            bg_color = 'white'
            if klasy_ang[idx] == 'happy':
                kolor_txt = 'darkgreen'
                bg_color = '#e6fffa'
            elif klasy_ang[idx] == 'angry':
                kolor_txt = 'darkred'
                bg_color = '#ffe6e6'
            elif klasy_ang[idx] == 'sad':
                kolor_txt = 'darkblue'
                bg_color = '#e6f0ff'

            # KROK D: Wyświetlanie
            ax = plt.subplot(current_rows, KOLUMNY, i + 1)
            plt.imshow(img_display)

            plt.title(f"{etykieta}\n({pewnosc:.1f}%)",
                      color=kolor_txt, fontsize=14, fontweight='bold',
                      bbox=dict(facecolor=bg_color, edgecolor='none', alpha=0.7))
            plt.axis('off')

            print(f"[{i + 1}/{current_batch_size}] {plik} -> {etykieta}")

        except Exception as e:
            print(f"Błąd przy pliku {plik}: {e}")

    plt.tight_layout()
    print("Wyświetlam okno wykresu. ZAMKNIJ JE, ABY ZOBACZYĆ NASTĘPNĄ STRONĘ.")
    plt.show()

print("\n--- Koniec przeglądania ---")