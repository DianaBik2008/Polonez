#!/usr/bin/env python3
"""
Generator animacji szkoleniowej tańca Polonez (widok z góry) dla 16 par (32 osób).
Figura z materiału wideo: „Łańcuch” / Mijanka wężykiem z podawaniem rąk (Grand Chaîne)
wzdłuż górnej ściany sali oraz zamiana stron kolumn Pań i Panów.

Renderowanie 4K-supersampled -> Full HD (1920x1080, 30 kl./s) przy użyciu Pillow + OpenCV + PyAV.
"""

import argparse
import math
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import av

# ==============================================================================
# KONFIGURACJA WIDEO I GEOMETRII SALI
# ==============================================================================
WIDTH = 1920
HEIGHT = 1080
SS = 2                 # 2x supersampling (3840x2160 -> 1920x1080) dla idealnego antyaliasingu
W_SS = WIDTH * SS
H_SS = HEIGHT * SS
FPS = 30

N_COUPLES = 16
D_SPACING = 64.0       # Odstęp między tancerzami w kolumnie (w jednostkach hali = px w 1080p)
R_DANCER = 12.8        # Promień kółka tancerza (średnica 25.6 px < D_SPACING/2 = 32 px -> brak kolizji)
A_WEAVE = 21.5         # Amplituda wychylenia wężyka w pionie (2*A = 43 px)

# Wymiary ścieżki w układzie hali (środek hali = (0, 0), oś Y skierowana W GÓRĘ!)
X_WALL = 566.0         # Pozycja X kolumn bocznych (-566 lewa, +566 prawa)
Y_TOP = 356.0          # Oś Y górnej linii mijanki
Y_BOT = -368.0         # Oś Y dolnej linii (przód sali)
R_CORNER = 36.0        # Promień zaokrąglenia narożników

# Położenie karty sali na ekranie 1920x1080
HALL_RECT = (28, 96, 1284, 1054)  # left, top, right, bottom (1256 x 958)
HALL_CX = (HALL_RECT[0] + HALL_RECT[2]) * 0.5   # 656.0
HALL_CY = (HALL_RECT[1] + HALL_RECT[3]) * 0.5 + 6.0  # 581.0

# Panel boczny po prawej stronie
SIDE_X0 = 1304
SIDE_X1 = 1892

# Czasy trwania poszczególnych faz (w sekundach)
T_INTRO_HOLD = 2.4     # Plansza początkowa (nieruchome ustawienie startowe)
T_APPROACH = 9.6       # Marsz w górę i wyjście na linię górną do 1. spotkania
T_STEP = 1.20          # Czas jednego minięcia / taktu poloneza (36 klatek = 3 uderzenia po 12 klatek)
T_WEAVE = 30 * T_STEP  # Od spotkania (1,1) [k=2] do spotkania (16,16) [k=32] -> 30 odstępów = 36.0 s
T_EXIT = 10.2          # Zejście kolumn na przeciwległe ściany
T_OUTRO_HOLD = 3.2     # Zatrzymanie końcowe (pokazanie zamienionych stron)

TOTAL_DURATION = T_INTRO_HOLD + T_APPROACH + T_WEAVE + T_EXIT + T_OUTRO_HOLD  # 61.4 s


# ==============================================================================
# MATEMATYKA ŚCIEŻKI I KINEMATYKI TANCERZY
# ==============================================================================
def build_lady_segments():
    """Buduje ścieżkę bazową Kolumny Pań (od dołu-środka -> lewa ściana -> góra -> prawa ściana -> dół)."""
    segs = []
    l1 = X_WALL - R_CORNER
    segs.append(('line', (0.0, Y_BOT), (-X_WALL + R_CORNER, Y_BOT), l1))
    segs.append(('arc', (-X_WALL + R_CORNER, Y_BOT + R_CORNER), R_CORNER,
                 -math.pi / 2, -math.pi, R_CORNER * math.pi / 2))
    l3 = (Y_TOP - R_CORNER) - (Y_BOT + R_CORNER)
    segs.append(('line', (-X_WALL, Y_BOT + R_CORNER), (-X_WALL, Y_TOP - R_CORNER), l3))
    segs.append(('arc', (-X_WALL + R_CORNER, Y_TOP - R_CORNER), R_CORNER,
                 math.pi, math.pi / 2, R_CORNER * math.pi / 2))
    l5 = 2.0 * (X_WALL - R_CORNER)
    segs.append(('line', (-X_WALL + R_CORNER, Y_TOP), (X_WALL - R_CORNER, Y_TOP), l5))
    segs.append(('arc', (X_WALL - R_CORNER, Y_TOP - R_CORNER), R_CORNER,
                 math.pi / 2, 0.0, R_CORNER * math.pi / 2))
    segs.append(('line', (X_WALL, Y_TOP - R_CORNER), (X_WALL, Y_BOT + R_CORNER), l3))
    segs.append(('arc', (X_WALL - R_CORNER, Y_BOT + R_CORNER), R_CORNER,
                 0.0, -math.pi / 2, R_CORNER * math.pi / 2))
    segs.append(('line', (X_WALL - R_CORNER, Y_BOT), (0.0, Y_BOT), l1))
    return segs


SEGS = build_lady_segments()
TOTAL_TRACK_LEN = sum(s[-1] for s in SEGS)
S_MID = TOTAL_TRACK_LEN * 0.5
S_TOP_START = sum(s[-1] for s in SEGS[:4])
S_TOP_END = sum(s[-1] for s in SEGS[:5])

S1_START = S_TOP_START - (R_CORNER * math.pi / 2) - 18.0
U_FIRST_MEET = S_MID - S1_START
U_LAST_MEET = U_FIRST_MEET + (2 * N_COUPLES - 2) * (D_SPACING * 0.5)
S16_END = S_TOP_END + (R_CORNER * math.pi / 2) + 42.0
S1_END = S16_END + (N_COUPLES - 1) * D_SPACING
U_END = S1_END - S1_START


def smoothstep(e0, e1, x):
    if e1 <= e0:
        return 1.0 if x >= e0 else 0.0
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def progress_u_at_time(t):
    t1 = T_INTRO_HOLD
    t2 = t1 + T_APPROACH
    t3 = t2 + T_WEAVE
    t4 = t3 + T_EXIT
    v_weave = (D_SPACING * 0.5) / T_STEP

    if t <= t1:
        return 0.0
    elif t <= t2:
        tau = (t - t1) / T_APPROACH
        h01 = -2 * tau**3 + 3 * tau**2
        h11 = tau**3 - tau**2
        return h01 * U_FIRST_MEET + h11 * (v_weave * T_APPROACH)
    elif t <= t3:
        return U_FIRST_MEET + v_weave * (t - t2)
    elif t <= t4:
        tau = (t - t3) / T_EXIT
        du = U_END - U_LAST_MEET
        h10 = tau**3 - 2 * tau**2 + tau
        h01 = -2 * tau**3 + 3 * tau**2
        return U_LAST_MEET + h10 * (v_weave * T_EXIT) + h01 * du
    else:
        return U_END


def eval_base_track(s):
    s = max(0.0, min(TOTAL_TRACK_LEN, s))
    acc = 0.0
    for seg in SEGS:
        L = seg[-1]
        if s <= acc + L + 1e-9:
            tau = max(0.0, min(1.0, (s - acc) / L))
            if seg[0] == 'line':
                (x0, y0), (x1, y1) = seg[1], seg[2]
                dx, dy = x1 - x0, y1 - y0
                return x0 + tau * dx, y0 + tau * dy, math.atan2(dy, dx)
            else:
                (cx, cy), R, a0, a1 = seg[1], seg[2], seg[3], seg[4]
                a = a0 + tau * (a1 - a0)
                x = cx + R * math.cos(a)
                y = cy + R * math.sin(a)
                sgn = 1.0 if a1 > a0 else -1.0
                return x, y, math.atan2(math.cos(a) * sgn, -math.sin(a) * sgn)
        acc += L
    return 0.0, Y_BOT, math.pi


def get_dancer_state(role, idx, u):
    s = S1_START + u - (idx - 1) * D_SPACING
    bx, by, bang = eval_base_track(s)

    x_min = (1 - idx) * D_SPACING * 0.5
    x_max = (N_COUPLES - idx) * D_SPACING * 0.5
    margin = 0.68 * D_SPACING
    inner = 0.25 * D_SPACING

    dy = 0.0
    slope = 0.0
    in_weave = False

    if (S_TOP_START - 20.0) <= s <= (S_TOP_END + 20.0):
        env_in = smoothstep(x_min - margin, x_min - inner, bx)
        env_out = 1.0 - smoothstep(x_max + inner, x_max + margin, bx)
        env = env_in * env_out
        if env > 1e-4:
            in_weave = True
            phase = 2.0 * math.pi * bx / D_SPACING
            dy = A_WEAVE * env * math.cos(phase)
            slope = -A_WEAVE * env * (2.0 * math.pi / D_SPACING) * math.sin(phase)

    if role == 'lady':
        x = bx
        y = by + dy
        ang = math.atan2(slope, 1.0) if in_weave else bang
    else:
        x = -bx
        y = by - dy
        ang = math.atan2(-slope, -1.0) if in_weave else math.atan2(math.sin(bang), -math.cos(bang))

    return x, y, ang, in_weave


def hall_to_screen(hx, hy):
    return (HALL_CX + hx) * SS, (HALL_CY - hy) * SS


# ==============================================================================
# CZCIONKI I POMOCNIKI RYSOWANIA (4K SUPERSAMPLED PILLOW)
# ==============================================================================
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_FONT_CACHE = {}


def get_font(kind, size_1080):
    key = (kind, round(size_1080 * SS, 1))
    if key not in _FONT_CACHE:
        fname = {
            "sans": "DejaVuSans.ttf",
            "bold": "DejaVuSans-Bold.ttf",
            "serif_bold": "DejaVuSerif-Bold.ttf",
        }[kind]
        _FONT_CACHE[key] = ImageFont.truetype(os.path.join(FONT_DIR, fname), int(round(size_1080 * SS)))
    return _FONT_CACHE[key]


def S(v):
    return int(round(v * SS))


def draw_dashed_polyline(draw, pts, color, width_ss, dash_len_ss, gap_len_ss):
    for k in range(len(pts) - 1):
        x0, y0 = pts[k]
        x1, y1 = pts[k + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < 1e-3:
            continue
        # Proste próbkowanie co odcinek (punkty są gęste)
        pass
    # Ciągła parametryzacja długości łuku dla idealnie równych kresek
    cum = [0.0]
    for k in range(1, len(pts)):
        cum.append(cum[-1] + math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]))
    total = cum[-1]
    period = dash_len_ss + gap_len_ss
    pos = 0.0
    idx = 0
    while pos < total:
        end_pos = min(total, pos + dash_len_ss)
        while idx < len(cum) - 2 and cum[idx + 1] < pos:
            idx += 1
        # Interpoluj punkt początkowy
        seg_pts = []
        t0 = (pos - cum[idx]) / max(1e-6, cum[idx + 1] - cum[idx])
        seg_pts.append((
            pts[idx][0] + t0 * (pts[idx + 1][0] - pts[idx][0]),
            pts[idx][1] + t0 * (pts[idx + 1][1] - pts[idx][1]),
        ))
        j = idx
        while j + 1 < len(cum) and cum[j + 1] < end_pos:
            j += 1
            seg_pts.append(pts[j])
        if j < len(cum) - 1:
            t1 = (end_pos - cum[j]) / max(1e-6, cum[j + 1] - cum[j])
            seg_pts.append((
                pts[j][0] + t1 * (pts[j + 1][0] - pts[j][0]),
                pts[j][1] + t1 * (pts[j + 1][1] - pts[j][1]),
            ))
        if len(seg_pts) >= 2:
            draw.line(seg_pts, fill=color, width=width_ss)
        pos += period


def draw_wrapped_text(draw, text, x_ss, y_ss, max_w_ss, line_h_ss, font, color):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_w_ss or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines):
        draw.text((x_ss, y_ss + i * line_h_ss), line, font=font, fill=color)


def draw_arrow_head(draw, x_ss, y_ss, ang_scr, size_ss, color):
    tx = x_ss + math.cos(ang_scr) * size_ss
    ty = y_ss + math.sin(ang_scr) * size_ss
    lx = x_ss + math.cos(ang_scr + 2.45) * size_ss * 0.88
    ly = y_ss + math.sin(ang_scr + 2.45) * size_ss * 0.88
    rx = x_ss + math.cos(ang_scr - 2.45) * size_ss * 0.88
    ry = y_ss + math.sin(ang_scr - 2.45) * size_ss * 0.88
    draw.polygon([(tx, ty), (lx, ly), (rx, ry)], fill=color)


def precompute_guide_paths():
    s_vals = np.linspace(S1_START - 15 * D_SPACING, S1_END, 420)
    lady_pts = []
    gent_pts = []
    for s in s_vals:
        bx, by, _ = eval_base_track(s)
        dy = 0.0
        if S_TOP_START <= s <= S_TOP_END and abs(bx) <= 7.8 * D_SPACING:
            env = smoothstep(-7.9 * D_SPACING, -7.2 * D_SPACING, bx) * (
                1.0 - smoothstep(7.2 * D_SPACING, 7.9 * D_SPACING, bx)
            )
            dy = A_WEAVE * env * math.cos(2.0 * math.pi * bx / D_SPACING)
        lady_pts.append(hall_to_screen(bx, by + dy))
        gent_pts.append(hall_to_screen(-bx, by - dy))
    return lady_pts, gent_pts


GUIDE_LADY_SS, GUIDE_GENT_SS = precompute_guide_paths()


# ==============================================================================
# PRE-RENDEROWANA WARSTWA STATYCZNA (TŁO, SALA, STAŁE KARTY PANELU)
# ==============================================================================
def build_static_master():
    """Tworzy raz obraz bazowy 4K ze wszystkimi elementami nieruchomymi."""
    # Gradient tła ciemnego granatu
    y_idx = np.linspace(0.0, 1.0, H_SS, dtype=np.float32)[:, None]
    x_idx = np.linspace(0.0, 1.0, W_SS, dtype=np.float32)[None, :]
    mix = 0.5 * (x_idx + y_idx)
    bg = np.zeros((H_SS, W_SS, 3), dtype=np.uint8)
    bg[:, :, 0] = (15 + mix * 15).astype(np.uint8)
    bg[:, :, 1] = (23 + mix * 18).astype(np.uint8)
    bg[:, :, 2] = (42 + mix * 17).astype(np.uint8)
    img = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # 1. Pasek nagłówka
    draw.rounded_rectangle([S(28), S(16), S(1892), S(82)], radius=S(14),
                           fill=(30, 41, 59, 240), outline=(71, 85, 105, 160), width=S(1.2))
    draw.rounded_rectangle([S(44), S(27), S(50), S(71)], radius=S(3), fill=(225, 29, 72, 255))
    draw.rounded_rectangle([S(54), S(27), S(58), S(71)], radius=S(2), fill=(245, 158, 11, 255))

    draw.text((S(70), S(24)),
              "POLONEZ — Figura: „Łańcuch” (Mijanka wężykiem z podawaniem rąk)",
              font=get_font("serif_bold", 21.5), fill=(248, 250, 252, 255))
    draw.text((S(70), S(54)),
              "Schemat szkoleniowy z góry  •  16 par (32 osoby)  •  Opracowanie na podstawie wideo z repozytorium",
              font=get_font("sans", 13.2), fill=(148, 163, 184, 255))

    # 2. Parkiet sali balowej / gimnastycznej
    hx0, hy0, hx1, hy1 = [S(v) for v in HALL_RECT]
    draw.rounded_rectangle([hx0, hy0, hx1, hy1], radius=S(16),
                           fill=(250, 245, 234, 255), outline=(180, 160, 130, 255), width=S(2))

    # Siatka parkietu
    for gx_line in range(-560, 561, 80):
        sx, _ = hall_to_screen(gx_line, 0)
        draw.line([(sx, hy0 + S(36)), (sx, hy1 - S(36))], fill=(218, 206, 184, 110), width=S(1))
    for gy_line in range(-360, 361, 80):
        _, sy = hall_to_screen(0, gy_line)
        draw.line([(hx0 + S(36), sy), (hx1 - S(36), sy)], fill=(218, 206, 184, 110), width=S(1))

    # Koło środkowe sali (jak na sali gimnastycznej z nagrania) + oś środkowa X=0
    cx_s, cy_s = S(HALL_CX), S(HALL_CY)
    cr_s = S(108)
    draw.ellipse([cx_s - cr_s, cy_s - cr_s, cx_s + cr_s, cy_s + cr_s],
                 fill=(234, 179, 8, 30), outline=(202, 138, 4, 95), width=S(1.5))
    draw_dashed_polyline(draw, [(cx_s, hy0 + S(42)), (cx_s, hy1 - S(42))],
                         (180, 130, 40, 105), S(1.4), S(7), S(6))

    # Ścieżki prowadzące Pań i Panów
    draw_dashed_polyline(draw, GUIDE_LADY_SS, (225, 29, 72, 72), S(2.0), S(8), S(6))
    draw_dashed_polyline(draw, GUIDE_GENT_SS, (29, 78, 216, 72), S(2.0), S(8), S(6))

    # Etykiety orientacji ścian sali
    # Górna ściana
    draw.rounded_rectangle([S(HALL_CX - 228), hy0 + S(10), S(HALL_CX + 228), hy0 + S(36)],
                           radius=S(13), fill=(30, 41, 59, 225))
    draw.text((S(HALL_CX), hy0 + S(23)), "GÓRNA ŚCIANA SALI  —  LINIA SPOTKANIA I ŁAŃCUCHA",
              font=get_font("bold", 12.2), fill=(248, 250, 252, 255), anchor="mm")

    # Znacznik środka górnej ściany
    _, sy_top_ss = hall_to_screen(0.0, Y_TOP)
    mid_y = sy_top_ss + S(58)
    draw.rounded_rectangle([S(HALL_CX - 142), mid_y - S(11), S(HALL_CX + 142), mid_y + S(11)],
                           radius=S(11), fill=(254, 243, 199, 240), outline=(217, 119, 6, 180), width=S(1))
    draw.text((S(HALL_CX), mid_y), "▲ ŚRODEK: spotkanie własnej pary (Prawa)",
              font=get_font("bold", 10.8), fill=(146, 64, 14, 255), anchor="mm")

    # Dolna ściana
    draw.rounded_rectangle([S(HALL_CX - 170), hy1 - S(35), S(HALL_CX + 170), hy1 - S(10)],
                           radius=S(12.5), fill=(30, 41, 59, 225))
    draw.text((S(HALL_CX), hy1 - S(22.5)), "PRZÓD SALI  /  PUBLICZNOŚĆ",
              font=get_font("bold", 12.2), fill=(248, 250, 252, 255), anchor="mm")

    # Pionowe opisy lewej i prawej ściany
    def paste_rotated_label(txt, angle_deg, center_xy_ss):
        f_rot = get_font("bold", 11.5)
        bb = draw.textbbox((0, 0), txt, font=f_rot)
        tw, th = bb[2] - bb[0] + S(16), bb[3] - bb[1] + S(12)
        tmp = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        td = ImageDraw.Draw(tmp)
        td.text((tw // 2, th // 2), txt, font=f_rot, fill=(110, 92, 68, 235), anchor="mm")
        rot = tmp.rotate(angle_deg, expand=True, resample=Image.Resampling.BICUBIC)
        img.paste(rot, (int(center_xy_ss[0] - rot.width // 2), int(center_xy_ss[1] - rot.height // 2)), rot)

    paste_rotated_label("LEWA ŚCIANA   (Start: Kolumna Pań ↑   •   Koniec: Kolumna Panów ↓)",
                        90, (hx0 + S(17), S(HALL_CY)))
    paste_rotated_label("PRAWA ŚCIANA   (Start: Kolumna Panów ↑   •   Koniec: Kolumna Pań ↓)",
                        -90, (hx1 - S(17), S(HALL_CY)))

    # 3. Tła 4 kart w prawym panelu edukacyjnym
    for (cy0, cy1) in [(96, 256), (270, 644), (658, 872), (886, 1054)]:
        draw.rounded_rectangle([S(SIDE_X0), S(cy0), S(SIDE_X1), S(cy1)], radius=S(14),
                               fill=(248, 250, 252, 255), outline=(148, 163, 184, 150), width=S(1.2))

    # Nagłówek Karty 2 (Zbliżenie)
    draw.text((S(SIDE_X0 + 18), S(282)),
              "ZBLIŻENIE 2.2× — ŚRODEK GÓRNEJ ŚCIANY (JAK PODAWAĆ RĘCE)",
              font=get_font("bold", 13.5), fill=(15, 23, 42, 255))
    draw.text((S(SIDE_X0 + 18), S(303)),
              "Panie idą w prawo (→), Panowie idą w lewo (←)  •  P = Prawa, L = Lewa",
              font=get_font("sans", 12.0), fill=(100, 116, 139, 255))

    # Karta 3 (Ściągawka dla każdej pary — statyczna)
    draw.text((S(SIDE_X0 + 18), S(670)),
              "ZASADA PODAWANIA RĄK — ŚCIĄGAWKA DLA TANCERZY",
              font=get_font("bold", 13.6), fill=(15, 23, 42, 255))

    rules = [
        ("1, 3, 5…", (217, 119, 6, 255),
         "Numery nieparzyste (Panowie i Panie):",
         "1. spotkanie = PRAWA ręka, potem na zmianę: P – L – P – L…"),
        ("2, 4, 6…", (5, 150, 105, 255),
         "Numery parzyste (Panowie i Panie):",
         "1. spotkanie = LEWA ręka (!), potem na zmianę: L – P – L – P…"),
        ("Para N+N", (190, 18, 60, 255),
         "Spotkanie z własną parą (np. Pan 5 i Pani 5):",
         "Wypada zawsze na środku (X=0) i zawsze PRAWĄ ręką!"),
    ]
    ry = 696
    for badge_txt, bcol, r_bold, r_sub in rules:
        draw.rounded_rectangle([S(SIDE_X0 + 18), S(ry), S(SIDE_X0 + 102), S(ry + 26)],
                               radius=S(7), fill=bcol)
        draw.text((S(SIDE_X0 + 60), S(ry + 13)), badge_txt,
                  font=get_font("bold", 11.8), fill=(255, 255, 255, 255), anchor="mm")
        draw.text((S(SIDE_X0 + 112), S(ry + 2)), r_bold,
                  font=get_font("bold", 12.6), fill=(15, 23, 42, 255))
        draw.text((S(SIDE_X0 + 112), S(ry + 21)), r_sub,
                  font=get_font("sans", 11.7), fill=(71, 85, 105, 255))
        ry += 56

    # Karta 4 (Legenda statyczna)
    draw.text((S(SIDE_X0 + 18), S(896)), "LEGENDA OZNACZEŃ",
              font=get_font("bold", 13.2), fill=(15, 23, 42, 255))

    ly1, ly2 = 930, 960
    # Pani
    draw.ellipse([S(SIDE_X0 + 20), S(ly1 - 10), S(SIDE_X0 + 40), S(ly1 + 10)], fill=(225, 29, 72, 255))
    draw.text((S(SIDE_X0 + 30), S(ly1)), "1", font=get_font("bold", 10.0), fill=(255, 255, 255, 255), anchor="mm")
    draw.text((S(SIDE_X0 + 46), S(ly1)), "Pani (1–16)", font=get_font("bold", 12.2), fill=(30, 41, 59, 255), anchor="lm")

    # Pan
    draw.ellipse([S(SIDE_X0 + 154), S(ly1 - 10), S(SIDE_X0 + 174), S(ly1 + 10)], fill=(29, 78, 216, 255))
    draw.text((S(SIDE_X0 + 164), S(ly1)), "1", font=get_font("bold", 10.0), fill=(255, 255, 255, 255), anchor="mm")
    draw.text((S(SIDE_X0 + 180), S(ly1)), "Pan (1–16)", font=get_font("bold", 12.2), fill=(30, 41, 59, 255), anchor="lm")

    # Para prowadząca
    draw.ellipse([S(SIDE_X0 + 286), S(ly1 - 11), S(SIDE_X0 + 308), S(ly1 + 11)],
                 outline=(217, 119, 6, 255), width=S(2.5))
    draw.text((S(SIDE_X0 + 315), S(ly1)), "Złota obwódka = Para nr 1 (prowadząca)",
              font=get_font("sans", 11.8), fill=(51, 65, 85, 255), anchor="lm")

    # Prawa ręka
    draw.ellipse([S(SIDE_X0 + 20), S(ly2 - 9.5), S(SIDE_X0 + 39), S(ly2 + 9.5)], fill=(217, 119, 6, 255))
    draw.text((S(SIDE_X0 + 29.5), S(ly2)), "P", font=get_font("bold", 9.8), fill=(255, 255, 255, 255), anchor="mm")
    draw.text((S(SIDE_X0 + 46), S(ly2)), "Prawa ręka (mijanie prawym ramieniem)",
              font=get_font("sans", 11.6), fill=(51, 65, 85, 255), anchor="lm")

    # Lewa ręka
    draw.ellipse([S(SIDE_X0 + 316), S(ly2 - 9.5), S(SIDE_X0 + 335), S(ly2 + 9.5)], fill=(5, 150, 105, 255))
    draw.text((S(SIDE_X0 + 325.5), S(ly2)), "L", font=get_font("bold", 9.8), fill=(255, 255, 255, 255), anchor="mm")
    draw.text((S(SIDE_X0 + 342), S(ly2)), "Lewa ręka (lewym ramieniem)",
              font=get_font("sans", 11.6), fill=(51, 65, 85, 255), anchor="lm")

    return img


STATIC_MASTER = build_static_master()


# ==============================================================================
# RYSOWANIE TANCERZY I POŁĄCZEŃ RĄK NA DOWOLNYM WIDOKU
# ==============================================================================
def draw_dancers_and_links(draw, ladies, gents, tf_ss, scale=1.0, show_words=False, clip_bounds_ss=None):
    """
    Rysuje połączenia rąk oraz wszystkich 32 tancerzy na przekazanym ImageDraw (w rozdzielczości 4K SS).
    tf_ss(hx, hy) -> (x_ss, y_ss).
    """
    r_ss = R_DANCER * scale * SS

    # 1. Aktywne podania rąk
    active_links = []
    for j_idx, (lx, ly, lang, lw) in enumerate(ladies, start=1):
        if not lw and abs(ly - Y_TOP) > 35.0:
            continue
        for i_idx, (gx, gy, gang, gw) in enumerate(gents, start=1):
            if not gw and abs(gy - Y_TOP) > 35.0:
                continue
            dx = abs(gx - lx)
            if dx < 0.36 * D_SPACING and abs(0.5 * (ly + gy) - Y_TOP) < 25.0:
                strength = 1.0 - smoothstep(0.14 * D_SPACING, 0.36 * D_SPACING, dx)
                is_right = ((i_idx + j_idx) % 2 == 0)
                is_own = (i_idx == j_idx)
                active_links.append((i_idx, j_idx, lx, ly, lang, gx, gy, gang, strength, is_right, is_own))

    for (i_idx, j_idx, lx, ly, lang, gx, gy, gang, strength, is_right, is_own) in active_links:
        slx, sly = tf_ss(lx, ly)
        sgx, sgy = tf_ss(gx, gy)
        mx, my = 0.5 * (slx + sgx), 0.5 * (sly + sgy)
        if clip_bounds_ss:
            cx0, cy0, cx1, cy1 = clip_bounds_ss
            if mx < cx0 - r_ss * 3 or mx > cx1 + r_ss * 3:
                continue

        alpha = int(255 * strength)
        if is_right:
            col_main = (217, 119, 6, alpha)
            col_glow = (251, 191, 36, int(115 * strength))
            l_sh_x = lx + math.sin(lang) * (R_DANCER * 0.80)
            l_sh_y = ly - math.cos(lang) * (R_DANCER * 0.80)
            g_sh_x = gx + math.sin(gang) * (R_DANCER * 0.80)
            g_sh_y = gy - math.cos(gang) * (R_DANCER * 0.80)
        else:
            col_main = (5, 150, 105, alpha)
            col_glow = (52, 211, 153, int(115 * strength))
            l_sh_x = lx - math.sin(lang) * (R_DANCER * 0.80)
            l_sh_y = ly + math.cos(lang) * (R_DANCER * 0.80)
            g_sh_x = gx - math.sin(gang) * (R_DANCER * 0.80)
            g_sh_y = gy + math.cos(gang) * (R_DANCER * 0.80)

        sl_sh = tf_ss(l_sh_x, l_sh_y)
        sg_sh = tf_ss(g_sh_x, g_sh_y)

        if is_own:
            pr = (16.5 + 8.5 * (1.0 - strength)) * scale * SS
            draw.ellipse([mx - pr, my - pr, mx + pr, my + pr],
                         outline=(245, 158, 11, int(175 * strength)), width=max(2, int(2.4 * scale * SS)))

        draw.line([sl_sh, (mx, my), sg_sh], fill=col_glow, width=max(3, int(7.2 * scale * SS)), joint="curve")
        draw.line([sl_sh, (mx, my), sg_sh], fill=col_main, width=max(2, int(3.0 * scale * SS)), joint="curve")

    # 2. Tancerze (Panie i Panowie)
    f_num = get_font("bold", 11.2 * scale)

    def draw_one(hx, hy, ang_hall, num, is_lady):
        sx, sy = tf_ss(hx, hy)
        if clip_bounds_ss:
            cx0, cy0, cx1, cy1 = clip_bounds_ss
            if sx < cx0 - r_ss * 2 or sx > cx1 + r_ss * 2 or sy < cy0 - r_ss * 2 or sy > cy1 + r_ss * 2:
                return

        ang_scr = math.atan2(-math.sin(ang_hall), math.cos(ang_hall))

        # Cień
        sh_dx, sh_dy = 1.2 * scale * SS, 2.2 * scale * SS
        draw.ellipse([sx - r_ss + sh_dx, sy - r_ss + sh_dy, sx + r_ss + sh_dx, sy + r_ss + sh_dy],
                     fill=(15, 23, 42, 46))

        # Złota obwódka dla Pary nr 1 (prowadzącej)
        if num == 1:
            rg = r_ss + 4.0 * scale * SS
            draw.ellipse([sx - rg, sy - rg, sx + rg, sy + rg], fill=(251, 191, 36, 90))
            rg2 = r_ss + 2.5 * scale * SS
            draw.ellipse([sx - rg2, sy - rg2, sx + rg2, sy + rg2],
                         outline=(217, 119, 6, 245), width=max(2, int(2.0 * scale * SS)))

        # Dzióbek kierunku ustawienia twarzy
        ntx = sx + math.cos(ang_scr) * (r_ss + 4.4 * scale * SS)
        nty = sy + math.sin(ang_scr) * (r_ss + 4.4 * scale * SS)
        nlx = sx + math.cos(ang_scr + 0.55) * (r_ss * 0.86)
        nly = sy + math.sin(ang_scr + 0.55) * (r_ss * 0.86)
        nrx = sx + math.cos(ang_scr - 0.55) * (r_ss * 0.86)
        nry = sy + math.sin(ang_scr - 0.55) * (r_ss * 0.86)
        ncol = (159, 18, 57, 255) if is_lady else (30, 58, 138, 255)
        draw.polygon([(ntx, nty), (nlx, nly), (nrx, nry)], fill=ncol)

        # Kółko tancerza
        if is_lady:
            fill_c = (225, 29, 72, 255)
            ring_c = (255, 228, 230, 175)
            out_c = (159, 18, 57, 255)
        else:
            fill_c = (29, 78, 216, 255)
            ring_c = (219, 234, 254, 165)
            out_c = (30, 58, 138, 255)

        draw.ellipse([sx - r_ss, sy - r_ss, sx + r_ss, sy + r_ss],
                     fill=fill_c, outline=out_c, width=max(1, int(1.2 * scale * SS)))
        ri = r_ss - 2.0 * scale * SS
        if ri > 2:
            draw.ellipse([sx - ri, sy - ri, sx + ri, sy + ri],
                         outline=ring_c, width=max(1, int(1.0 * scale * SS)))

        draw.text((sx, sy), str(num), font=f_num, fill=(255, 255, 255, 255), anchor="mm")

    for j_idx, (lx, ly, lang, _) in enumerate(ladies, start=1):
        draw_one(lx, ly, lang, j_idx, True)
    for i_idx, (gx, gy, gang, _) in enumerate(gents, start=1):
        draw_one(gx, gy, gang, i_idx, False)

    # 3. Etykiety P / L na dłoniach
    for (i_idx, j_idx, lx, ly, lang, gx, gy, gang, strength, is_right, is_own) in active_links:
        if strength < 0.22:
            continue
        slx, sly = tf_ss(lx, ly)
        sgx, sgy = tf_ss(gx, gy)
        mx, my = 0.5 * (slx + sgx), 0.5 * (sly + sgy)
        if clip_bounds_ss:
            cx0, cy0, cx1, cy1 = clip_bounds_ss
            if mx < cx0 or mx > cx1:
                continue

        alpha = int(255 * min(1.0, strength * 1.25))
        bg = (217, 119, 6, alpha) if is_right else (5, 150, 105, alpha)
        if show_words:
            txt = "PRAWA" if is_right else "LEWA"
            fb = get_font("bold", 11.0)
            bb = draw.textbbox((0, 0), txt, font=fb)
            pw = (bb[2] - bb[0]) + S(14)
            ph = S(20)
            draw.rounded_rectangle([mx - pw * 0.5, my - ph * 0.5, mx + pw * 0.5, my + ph * 0.5],
                                   radius=S(9), fill=bg, outline=(255, 255, 255, alpha), width=S(1.4))
            draw.text((mx, my), txt, font=fb, fill=(255, 255, 255, alpha), anchor="mm")
        else:
            txt = "P" if is_right else "L"
            br = 7.8 * scale * SS
            draw.ellipse([mx - br, my - br, mx + br, my + br],
                         fill=bg, outline=(255, 255, 255, alpha), width=S(1.2))
            draw.text((mx, my), txt, font=get_font("bold", 9.0 * scale),
                      fill=(255, 255, 255, alpha), anchor="mm")


# Pre-kalkulacja sinusoid w oknie zbliżenia (żeby nie liczyć co klatkę)
ZOOM_W, ZOOM_H = 556, 304
ZOOM_SCALE = 2.22


def build_zoom_base():
    zw_ss, zh_ss = S(ZOOM_W), S(ZOOM_H)
    zimg = Image.new("RGB", (zw_ss, zh_ss), (250, 246, 236))
    zdraw = ImageDraw.Draw(zimg, "RGBA")
    z_cx, z_cy = zw_ss * 0.5, zh_ss * 0.5 + S(6)

    # Pas górnej ściany
    zdraw.rectangle([0, 0, zw_ss, S(26)], fill=(226, 216, 198, 255))
    zdraw.text((z_cx, S(13)), "▲ GÓRNA ŚCIANA SALI (bliżej ściany)",
               font=get_font("bold", 10.8), fill=(90, 75, 55, 255), anchor="mm")
    zdraw.text((z_cx, zh_ss - S(12)), "▼ BLIŻEJ ŚRODKA SALI",
               font=get_font("bold", 10.8), fill=(120, 105, 85, 255), anchor="mm")

    draw_dashed_polyline(zdraw, [(z_cx, S(28)), (z_cx, zh_ss - S(22))],
                         (180, 130, 40, 115), S(1.4), S(6), S(6))
    zdraw.line([(0, z_cy), (zw_ss, z_cy)], fill=(203, 213, 225, 180), width=S(1))

    zp_l, zp_g = [], []
    for x_s in np.linspace(-135.0, 135.0, 110):
        dy_s = A_WEAVE * math.cos(2.0 * math.pi * x_s / D_SPACING)
        zp_l.append((z_cx + x_s * ZOOM_SCALE * SS, z_cy - dy_s * ZOOM_SCALE * SS))
        zp_g.append((z_cx + x_s * ZOOM_SCALE * SS, z_cy + dy_s * ZOOM_SCALE * SS))
    draw_dashed_polyline(zdraw, zp_l, (225, 29, 72, 75), S(2.0), S(8), S(6))
    draw_dashed_polyline(zdraw, zp_g, (29, 78, 216, 75), S(2.0), S(8), S(6))
    return zimg


ZOOM_BASE = build_zoom_base()


# ==============================================================================
# RENDEROWANIE KLATKI
# ==============================================================================
def render_frame(t):
    frame = STATIC_MASTER.copy()
    draw = ImageDraw.Draw(frame, "RGBA")

    u = progress_u_at_time(t)
    ladies = [get_dancer_state('lady', j, u) for j in range(1, N_COUPLES + 1)]
    gents = [get_dancer_state('gent', i, u) for i in range(1, N_COUPLES + 1)]

    # 1. Licznik taktu poloneza (3/4: RAZ - dwa - trzy) w prawym górnym rogu
    takt_num = int(t / T_STEP) + 1
    total_takty = int(math.ceil(TOTAL_DURATION / T_STEP))
    beat_idx = int((t / T_STEP) * 3.0) % 3

    draw.rounded_rectangle([S(1470), S(24), S(1878), S(74)], radius=S(10), fill=(15, 23, 42, 215))
    draw.text((S(1486), S(49)), f"Metrum 3/4  •  Takt {takt_num:02d}/{total_takty}",
              font=get_font("bold", 14.0), fill=(226, 232, 240, 255), anchor="lm")

    beat_labels = ["1 (RAZ)", "2", "3"]
    bx_pos = [1728, 1798, 1841]
    bw_w = [60, 34, 34]
    for b_i in range(3):
        act = (b_i == beat_idx)
        bcol = ((245, 158, 11, 255) if b_i == 0 else (59, 130, 246, 255)) if act else (51, 65, 85, 170)
        tcol = ((15, 23, 42, 255) if b_i == 0 else (255, 255, 255, 255)) if act else (148, 163, 184, 255)
        x0_b = bx_pos[b_i]
        w_b = bw_w[b_i]
        draw.rounded_rectangle([S(x0_b), S(32), S(x0_b + w_b), S(66)], radius=S(7), fill=bcol)
        draw.text((S(x0_b + w_b * 0.5), S(49)), beat_labels[b_i],
                  font=get_font("bold", 12.0), fill=tcol, anchor="mm")

    # 2. Strzałki kierunkowe na torach bocznych oraz karta informacyjna na środku sali
    swapped = (u > (U_FIRST_MEET + U_LAST_MEET) * 0.5)
    for ay in [-200.0, 0.0, 200.0]:
        lsx, lsy = hall_to_screen(-X_WALL + 28.0, ay)
        rsx, rsy = hall_to_screen(+X_WALL - 28.0, ay)
        if not swapped:
            draw_arrow_head(draw, lsx, lsy, -math.pi / 2, S(8.5), (225, 29, 72, 135))
            draw_arrow_head(draw, rsx, rsy, -math.pi / 2, S(8.5), (29, 78, 216, 135))
        else:
            draw_arrow_head(draw, lsx, lsy, +math.pi / 2, S(8.5), (29, 78, 216, 150))
            draw_arrow_head(draw, rsx, rsy, +math.pi / 2, S(8.5), (225, 29, 72, 150))

    # Centralna ramka statusu na parkiecie
    draw.rounded_rectangle([S(HALL_CX - 245), S(HALL_CY - 45), S(HALL_CX + 245), S(HALL_CY + 45)],
                           radius=S(14), fill=(255, 255, 255, 224), outline=(203, 213, 225, 220), width=S(1.2))
    if t < T_INTRO_HOLD:
        c_head = "USTAWIENIE POCZĄTKOWE (16 PAR)"
        c_sub1 = "Lewa strona: Panie 1–16   |   Prawa strona: Panowie 1–16"
        c_sub2 = "Za chwilę obie kolumny ruszą w górę i skręcą ku sobie"
        c_col = (30, 41, 59, 255)
    elif u < U_FIRST_MEET - 16.0:
        c_head = "ETAP 1: WEJŚCIE NA GÓRNĄ LINIĘ"
        c_sub1 = "Panie skręcają w prawo →   |   ← Panowie skręcają w lewo"
        c_sub2 = "Kolumny idą naprzeciw siebie wzdłuż górnej ściany"
        c_col = (29, 78, 216, 255)
    elif u <= U_LAST_MEET + 18.0:
        k_enc = max(2, min(32, int(round((u - U_FIRST_MEET) / (D_SPACING * 0.5))) + 2))
        hand_now = "PRAWA RĘKA (P)" if (k_enc % 2 == 0) else "LEWA RĘKA (L)"
        c_head = "ETAP 2–3: ŁAŃCUCH (MIJANKA WĘŻYKIEM)"
        c_sub1 = f"Krok łańcucha {k_enc - 1}/31  •  Dominujący chwyt: {hand_now}"
        if k_enc % 2 == 0:
            c_sub2 = f"Na środku górnej ściany (X=0) spotyka się Para nr {k_enc // 2}!"
            c_col = (180, 83, 9, 255)
        else:
            c_sub2 = "Tancerze mijają się lewymi ramionami, podając LEWĄ rękę"
            c_col = (4, 120, 87, 255)
    elif t < TOTAL_DURATION - T_OUTRO_HOLD:
        c_head = "ETAP 4: ZMIANA STRON I WYJŚCIE W DÓŁ"
        c_sub1 = "Panowie (1–16) schodzą w dół LEWĄ ścianą"
        c_sub2 = "Panie (1–16) schodzą w dół PRAWĄ ścianą"
        c_col = (30, 58, 138, 255)
    else:
        c_head = "KONIEC FIGURY — STRONY ZAMIENIONE!"
        c_sub1 = "Po lewej: Kolumna Panów 1–16  |  Po prawej: Kolumna Pań 1–16"
        c_sub2 = "Na dole sali prowadzący (Para 1) mogą znów połączyć kolumny w pary"
        c_col = (22, 101, 52, 255)

    draw.text((S(HALL_CX), S(HALL_CY - 22)), c_head, font=get_font("bold", 14.2), fill=c_col, anchor="mm")
    draw.text((S(HALL_CX), S(HALL_CY + 3)), c_sub1, font=get_font("bold", 12.2), fill=(30, 41, 59, 255), anchor="mm")
    draw.text((S(HALL_CX), S(HALL_CY + 24)), c_sub2, font=get_font("sans", 11.8), fill=(71, 85, 105, 255), anchor="mm")

    # 3. Tancerze i podania rąk na głównej mapie sali
    draw_dancers_and_links(draw, ladies, gents, hall_to_screen, scale=1.0, show_words=False)

    # 4. Karta 1 w prawym panelu: Aktualny etap
    if u < U_FIRST_MEET - 16.0:
        st_badge, st_col = "ETAP 1 z 4", (29, 78, 216, 255)
        st_title = "Marsz kolumn i wejście na górną ścianę"
        st_desc = ("Kolumna Pań (lewa ściana) i Kolumna Panów (prawa ściana) idą w górę sali, "
                   "skręcają do środka na górnej ścianie i zbliżają się do siebie.")
    elif u < U_FIRST_MEET + 40.0:
        st_badge, st_col = "ETAP 2 z 4", (217, 119, 6, 255)
        st_title = "Spotkanie Pary nr 1 — Prawa ręka"
        st_desc = ("Pan 1 oraz Pani 1 spotykają się na samym środku górnej ściany, "
                   "podają sobie PRAWĄ rękę i mijają się prawymi ramionami.")
    elif u <= U_LAST_MEET + 18.0:
        st_badge, st_col = "ETAP 3 z 4", (190, 18, 60, 255)
        st_title = "Łańcuch (mijanka wężykiem)"
        st_desc = ("Wszyscy tancerze przeplatają się wężykiem, podając naprzemiennie PRAWĄ (P) "
                   "i LEWĄ (L) rękę wszystkim 16 osobom z przeciwnej kolumny.")
    else:
        st_badge, st_col = "ETAP 4 z 4", (5, 150, 105, 255)
        st_title = "Zamiana ścian i marsz w dół sali"
        st_desc = ("Po minięciu całej przeciwnej kolumny Panowie schodzą w dół LEWĄ ścianą, "
                   "a Panie PRAWĄ ścianą. Kolumny w pełni zamieniły się stronami!")

    draw.rounded_rectangle([S(SIDE_X0 + 18), S(110), S(SIDE_X0 + 114), S(134)], radius=S(12), fill=st_col)
    draw.text((S(SIDE_X0 + 66), S(122)), st_badge, font=get_font("bold", 11.4), fill=(255, 255, 255, 255), anchor="mm")
    draw.text((S(SIDE_X0 + 124), S(122)), st_title, font=get_font("bold", 15.2), fill=(15, 23, 42, 255), anchor="lm")
    draw_wrapped_text(draw, st_desc, S(SIDE_X0 + 18), S(146), S(SIDE_X1 - SIDE_X0 - 36), S(22),
                      get_font("sans", 13.4), (51, 65, 85, 255))

    # 5. Karta 2: Okno ZBLIŻENIA (osobny wycięty pod-obraz, wklejony czysto w ramkę)
    zimg = ZOOM_BASE.copy()
    zdraw = ImageDraw.Draw(zimg, "RGBA")
    zw_ss, zh_ss = zimg.size
    z_cx, z_cy = zw_ss * 0.5, zh_ss * 0.5 + S(6)

    def zoom_tf_local(hx, hy):
        return z_cx + hx * ZOOM_SCALE * SS, z_cy - (hy - Y_TOP) * ZOOM_SCALE * SS

    draw_dancers_and_links(zdraw, ladies, gents, zoom_tf_local, scale=ZOOM_SCALE,
                           show_words=True, clip_bounds_ss=(0, 0, zw_ss, zh_ss))
    if u < U_FIRST_MEET - 155.0:
        zdraw.rounded_rectangle([S(78), S(124), zw_ss - S(78), S(186)],
                                radius=S(10), fill=(255, 255, 255, 228), outline=(203, 213, 225, 220), width=S(1.2))
        zdraw.text((z_cx, S(146)), "Pani 1 (→) oraz Pan 1 (←) zbliżają się do środka",
                   font=get_font("bold", 12.2), fill=(30, 41, 59, 255), anchor="mm")
        zdraw.text((z_cx, S(168)), "Za chwilę rozpocznie się mijanka od PRAWEJ ręki",
                   font=get_font("sans", 11.2), fill=(100, 116, 139, 255), anchor="mm")
    elif u > U_LAST_MEET + 155.0:
        zdraw.rounded_rectangle([S(68), S(124), zw_ss - S(68), S(186)],
                                radius=S(10), fill=(255, 255, 255, 228), outline=(16, 185, 129, 190), width=S(1.2))
        zdraw.text((z_cx, S(146)), "Wszystkie 16 par przeszło przez łańcuch ✓",
                   font=get_font("bold", 12.4), fill=(6, 95, 70, 255), anchor="mm")
        zdraw.text((z_cx, S(168)), "Kolumny zeszły w dół po przeciwnych stronach sali",
                   font=get_font("sans", 11.2), fill=(71, 85, 105, 255), anchor="mm")

    zx0, zy0 = S(SIDE_X0 + 16), S(326)
    frame.paste(zimg, (zx0, zy0))
    draw.rounded_rectangle([zx0, zy0, zx0 + zw_ss, zy0 + zh_ss],
                           radius=S(8), outline=(180, 160, 130, 255), width=S(1.8))

    # 6. Karta 4: Pasek postępu
    bar_x0 = SIDE_X0 + 18
    bar_x1 = SIDE_X1 - 18
    bar_w = bar_x1 - bar_x0
    bar_y = 998
    prog = max(0.0, min(1.0, t / TOTAL_DURATION))

    draw.rounded_rectangle([S(bar_x0), S(bar_y), S(bar_x1), S(bar_y + 12)],
                           radius=S(6), fill=(203, 213, 225, 255))
    if prog > 0.005:
        px_end = max(bar_x0 + 12, bar_x0 + bar_w * prog)
        draw.rounded_rectangle([S(bar_x0), S(bar_y), S(px_end), S(bar_y + 12)],
                               radius=S(6), fill=(29, 78, 216, 255))

    f_prog = get_font("sans", 11.0)
    draw.text((S(bar_x0), S(bar_y + 28)), "Start", font=f_prog, fill=(100, 116, 139, 255), anchor="lm")
    draw.text((S(bar_x0 + bar_w * 0.15), S(bar_y + 28)), "Wejście", font=f_prog, fill=(100, 116, 139, 255), anchor="mm")
    draw.text((S(bar_x0 + bar_w * 0.45), S(bar_y + 28)), "Łańcuch (1–16)", font=f_prog, fill=(100, 116, 139, 255), anchor="mm")
    draw.text((S(bar_x0 + bar_w * 0.74), S(bar_y + 28)), "Zmiana ścian", font=f_prog, fill=(100, 116, 139, 255), anchor="mm")
    time_str = f"{int(t)//60}:{int(t)%60:02d} / {int(TOTAL_DURATION)//60}:{int(TOTAL_DURATION)%60:02d}"
    draw.text((S(bar_x1), S(bar_y + 28)), time_str, font=get_font("bold", 11.2), fill=(30, 41, 59, 255), anchor="rm")

    # Downsample 4K -> 1080p (cv2.INTER_AREA daje perfekcyjny 4-próbkowy antyaliasing!)
    rgb_ss = np.asarray(frame)
    rgb_1080 = cv2.resize(rgb_ss, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return rgb_1080


# ==============================================================================
# SYNTEZA AKOMPANIAMENTU RYTMICZNEGO POLONEZA (3/4, A-moll / C-dur)
# ==============================================================================
def synthesize_polonez_audio(duration_s, sr=48000):
    n_samples = int(round(duration_s * sr))
    audio = np.zeros(n_samples, dtype=np.float32)
    beat_dur = T_STEP / 3.0

    def note_freq(midi):
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    def add_piano_tone(start_t, length_s, midi_note, amp):
        i0 = int(round(start_t * sr))
        if i0 >= n_samples:
            return
        i1 = min(n_samples, i0 + int(round(length_s * sr)))
        if i1 <= i0:
            return
        tt = np.arange(i1 - i0, dtype=np.float32) / sr
        f = note_freq(midi_note)
        env = np.minimum(1.0, tt / 0.012) * np.exp(-tt / (length_s * 0.58))
        wave = (
            1.00 * np.sin(2.0 * np.pi * f * tt)
            + 0.38 * np.sin(2.0 * np.pi * 2.0 * f * tt) * np.exp(-tt * 2.2)
            + 0.14 * np.sin(2.0 * np.pi * 3.0 * f * tt) * np.exp(-tt * 3.5)
            + 0.05 * np.sin(2.0 * np.pi * 4.0 * f * tt) * np.exp(-tt * 5.0)
        )
        audio[i0:i1] += (amp * env * wave).astype(np.float32)

    chords = [
        (45, [57, 60, 64], [69, 72, 71, 69]),  # Am
        (50, [57, 62, 65], [74, 77, 76, 74]),  # Dm
        (40, [56, 59, 64], [71, 76, 74, 71]),  # E7
        (45, [57, 60, 64], [69, 72, 76, 72]),  # Am
        (48, [60, 64, 67], [76, 79, 77, 76]),  # C
        (50, [57, 62, 65], [74, 77, 76, 74]),  # Dm
        (40, [56, 59, 62], [71, 74, 72, 71]),  # E7
        (45, [57, 60, 64], [69, 71, 72, 69]),  # Am
    ]

    n_bars = int(math.ceil(duration_s / T_STEP))
    for bar in range(n_bars):
        t0 = bar * T_STEP
        bass, triad, mel = chords[bar % len(chords)]

        add_piano_tone(t0, beat_dur * 2.4, bass, 0.10)
        for m in triad:
            add_piano_tone(t0, beat_dur * 0.9, m, 0.045)
        for m in triad:
            add_piano_tone(t0 + 0.5 * beat_dur, beat_dur * 0.45, m, 0.025)
        for m in triad:
            add_piano_tone(t0 + 1.0 * beat_dur, beat_dur * 0.85, m, 0.036)
        for m in triad:
            add_piano_tone(t0 + 2.0 * beat_dur, beat_dur * 0.85, m, 0.034)

        add_piano_tone(t0, beat_dur * 0.95, mel[0], 0.055)
        add_piano_tone(t0 + 0.5 * beat_dur, beat_dur * 0.45, mel[1], 0.038)
        add_piano_tone(t0 + 1.0 * beat_dur, beat_dur * 0.95, mel[2], 0.048)
        add_piano_tone(t0 + 2.0 * beat_dur, beat_dur * 0.95, mel[3], 0.045)

    fade_in = int(0.8 * sr)
    fade_out = int(1.8 * sr)
    audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    for delay_ms, gain in [(43, 0.22), (89, 0.14), (147, 0.08)]:
        d_samp = int(delay_ms * sr / 1000)
        if d_samp < n_samples:
            audio[d_samp:] += gain * audio[:-d_samp]

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio * (0.42 / peak)

    return np.stack([audio, audio], axis=0)


# ==============================================================================
# EKSPORT DO PLIKU MP4
# ==============================================================================
def build_video(out_path):
    t_start = time.time()
    n_frames = int(round(TOTAL_DURATION * FPS))
    print(f"Generowanie wideo: {out_path} ({WIDTH}x{HEIGHT}, {FPS} fps, {n_frames} klatek, {TOTAL_DURATION:.1f}s)...")

    container = av.open(out_path, mode='w')

    v_stream = container.add_stream('libx264', rate=FPS)
    v_stream.width = WIDTH
    v_stream.height = HEIGHT
    v_stream.pix_fmt = 'yuv420p'
    v_stream.options = {'crf': '18', 'preset': 'fast', 'profile': 'high'}

    sr = 48000
    a_stream = container.add_stream('aac', rate=sr)
    a_stream.layout = 'stereo'
    a_stream.bit_rate = 160000

    for f_idx in range(n_frames):
        t = f_idx / FPS
        rgb = render_frame(t)
        v_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        for packet in v_stream.encode(v_frame):
            container.mux(packet)
        if (f_idx + 1) % 150 == 0 or f_idx == n_frames - 1:
            elapsed = time.time() - t_start
            print(f"  Klatka {f_idx + 1}/{n_frames} ({100.0*(f_idx+1)/n_frames:.0f}%) — {elapsed:.1f}s")

    for packet in v_stream.encode():
        container.mux(packet)

    stereo = synthesize_polonez_audio(TOTAL_DURATION, sr=sr)
    chunk = 1024
    pts = 0
    for idx in range(0, stereo.shape[1], chunk):
        part = np.ascontiguousarray(stereo[:, idx:idx + chunk])
        a_frame = av.AudioFrame.from_ndarray(part, format='fltp', layout='stereo')
        a_frame.sample_rate = sr
        a_frame.pts = pts
        pts += part.shape[1]
        for packet in a_stream.encode(a_frame):
            container.mux(packet)

    for packet in a_stream.encode():
        container.mux(packet)

    container.close()

    # Interleave pakietów audio/wideo oraz ustawienie atomu moov na początku (+faststart)
    tmp_fast = out_path + ".faststart.mp4"
    inp = av.open(out_path)
    out_c = av.open(tmp_fast, mode='w', options={'movflags': '+faststart'})
    v_in = inp.streams.video[0]
    a_in = inp.streams.audio[0]
    v_out = out_c.add_stream_from_template(v_in)
    a_out = out_c.add_stream_from_template(a_in)
    pkts = []
    for p in inp.demux(v_in, a_in):
        if p.dts is None:
            continue
        pkts.append((float(p.dts * p.time_base), 0 if p.stream.type == 'video' else 1, p))
    pkts.sort(key=lambda x: (x[0], x[1]))
    for _, kind, p in pkts:
        p.stream = v_out if kind == 0 else a_out
        out_c.mux(p)
    out_c.close()
    inp.close()
    os.replace(tmp_fast, out_path)

    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Gotowe! Zapisano {out_path} ({sz_mb:.2f} MB) w {time.time() - t_start:.1f}s.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="polonez_16_par_schemat.mp4")
    parser.add_argument("--preview-dir", default=None)
    args = parser.parse_args()

    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)
        sample_times = [1.0, 7.5, 12.0, 13.2, 26.4, 59.5]
        for st in sample_times:
            rgb = render_frame(st)
            p = os.path.join(args.preview_dir, f"preview_{st:05.2f}s.png")
            Image.fromarray(rgb).save(p)
            print("Zapisano podgląd:", p)
    else:
        build_video(args.out)


if __name__ == "__main__":
    main()
