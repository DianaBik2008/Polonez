# Polonez — Schematyczna animacja szkoleniowa (16 par)

Robienie tańca polonez, jego modelu matematycznego oraz animacji szkoleniowej z widokiem z góry.

## Zawartość repozytorium

- **`polonez_16_par_schemat.mp4`** — gotowa animacja szkoleniowa Full HD (`1920×1080`, `30 kl./s`, z rytmicznym akompaniamentem w metrum `3/4`) przedstawiająca figurę z nagrania referencyjnego: **„Łańcuch” / Mijankę wężykiem z podawaniem rąk (*Grand Chaîne*)** dla **16 par (32 tancerzy)** z pełnymi opisami po polsku:
  - widok całej sali z góry ze ścieżkami ruchu **Kolumny Pań (1–16)** i **Kolumny Panów (1–16)**,
  - okno **Zbliżenia 2.2×** na środek górnej ściany pokazujące dokładnie podawanie **PRAWEJ (`P`)** i **LEWEJ (`L`)** ręki oraz mijanie odpowiednimi ramionami,
  - ściągawkę dla par nieparzystych (`1, 3, 5…` → start od **PRAWEJ** ręki) i parzystych (`2, 4, 6…` → start od **LEWEJ** ręki) oraz spotkania z własną parą na środku sali,
  - licznik taktów poloneza (`RAZ – dwa – trzy`).
- **`generate_polonez.py`** — skrypt w Pythonie generujący animację (4K supersampling → 1080p, Pillow + OpenCV + PyAV).
- **`video_2026-11-01_14-08-38.mp4`** — materiał wideo z nagraniem figury referencyjnej.

## Ponowne wygenerowanie wideo

```bash
pip install av pillow numpy opencv-python-headless
python3 generate_polonez.py --out polonez_16_par_schemat.mp4
```
