#!/usr/bin/env python3
"""
clayreel.py — Motor de Reels estil plastilina per a @pepnivell.

Ús:  python3 clayreel.py episodi.json sortida.mp4

Genera un vídeo vertical 1080x1920, 30 fps, amb personatges de colors pastel
(aspecte de plastilina), targetes de dades animades i música suau generada.
Tot el contingut (personatges, edats, textos, xifres) ve del fitxer JSON.
"""
import json, math, random, subprocess, sys, os, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
FONT_DIR = "/usr/share/fonts/truetype/google-fonts"
def font(size, bold=True):
    for name in (["Poppins-Bold.ttf", "Poppins-SemiBold.ttf"] if bold else ["Poppins-Medium.ttf", "Poppins-Regular.ttf"]):
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)

RNG = np.random.default_rng(7)

# ---------------------------------------------------------------- palette
PAL = {
    "skin": (247, 210, 186), "cheek": (241, 152, 150), "cream": (240, 229, 205),
    "brown": (152, 107, 82), "sage": (152, 178, 142), "pink": (208, 143, 143),
    "teal": (112, 167, 172), "charcoal": (88, 92, 98), "mustard": (222, 184, 110),
    "lilac": (176, 160, 205), "sky": (150, 190, 220), "coral": (236, 150, 125),
    "grey": (128, 128, 130), "taupe": (140, 118, 104), "denim": (110, 135, 170),
    "shoe": (112, 76, 60), "hair_grey": (204, 204, 209), "hair_brown": (112, 76, 55),
    "hair_dark": (70, 52, 42), "hair_blond": (222, 186, 120), "ink": (70, 52, 46),
}
def col(c):
    return tuple(c) if isinstance(c, (list, tuple)) else PAL[c]

# ---------------------------------------------------------------- clay shading
def _shape_mask(w, h, kind, radius=0.35, ss=3, poly=None):
    m = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(m)
    if kind == "ellipse":
        d.ellipse([0, 0, w * ss - 1, h * ss - 1], fill=255)
    elif kind == "rrect":
        r = int(min(w, h) * radius * ss)
        d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], r, fill=255)
    elif kind == "poly":
        d.polygon([(x * ss, y * ss) for x, y in poly], fill=255)
    return m.resize((w, h), Image.LANCZOS)

def clay(w, h, color, kind="ellipse", radius=0.35, poly=None, light=(0.34, 0.26), soft=0.10):
    """Devuelve un sprite RGBA con aspecto de plastilina mate."""
    w, h = max(2, int(w)), max(2, int(h))
    mask = _shape_mask(w, h, kind, radius, poly=poly)
    if kind == "poly":
        mask = mask.filter(ImageFilter.GaussianBlur(max(1, min(w, h) * 0.04)))
        mask = mask.point(lambda v: 255 if v > 128 else int(v * 2))
    m = np.asarray(mask, dtype=np.float32) / 255.0
    blur_r = max(1.5, min(w, h) * soft)
    inner = np.asarray(mask.filter(ImageFilter.GaussianBlur(blur_r)), dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = xx / w, yy / h
    dist = np.sqrt((nx - light[0]) ** 2 + (ny - light[1]) ** 2)
    shade = 1.08 - 0.30 * dist                      # llum superior esquerra
    shade *= 0.78 + 0.22 * inner                    # vores més fosques (volum)
    spec = np.exp(-(((nx - light[0] + 0.04) ** 2) + ((ny - light[1] + 0.03) ** 2)) / 0.012) * 0.16
    base = np.array(color, dtype=np.float32)[None, None, :]
    rgb = base * shade[..., None] + (255 - base) * spec[..., None]
    rgb += RNG.normal(0, 2.2, (h, w, 1))            # gra de plastilina
    rgb = np.clip(rgb, 0, 255)
    out = np.dstack([rgb, m * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")

def put(canvas, sprite, cx, cy):
    canvas.alpha_composite(sprite, (int(cx - sprite.width / 2), int(cy - sprite.height / 2)))

def blob(canvas, cx, cy, w, h, color, kind="ellipse", radius=0.35, **kw):
    put(canvas, clay(w, h, col(color), kind, radius, **kw), cx, cy)

# ---------------------------------------------------------------- characters
def age_group(age):
    if age < 13: return "kid"
    if age < 46: return "young"
    if age < 65: return "adult"
    return "senior"

class Character:
    """Personatge frontal estil figureta. L'edat defineix proporcions, cabell i detalls."""
    def __init__(self, spec):
        self.spec = spec
        self.age = int(spec.get("age", 30))
        self.group = age_group(self.age)
        self.sex = spec.get("sex", "m")
        g = self.group
        self.height = int({"kid": 520, "young": 760, "adult": 750, "senior": 720}[g] * 0.82)
        self.head_frac = {"kid": 0.42, "young": 0.34, "adult": 0.33, "senior": 0.35}[g]
        default_hair = {"kid": "hair_blond" if self.sex == "f" else "hair_brown",
                        "young": "hair_dark" if self.sex == "f" else "hair_brown",
                        "adult": "hair_brown", "senior": "hair_grey"}[g]
        self.hair = col(spec.get("hair", default_hair))
        self.jacket = col(spec.get("jacket", "brown"))
        self.shirt = col(spec.get("shirt", "cream"))
        self.bottom = spec.get("bottom", "skirt" if self.sex == "f" else "trousers")
        self.bottom_color = col(spec.get("bottom_color", "sage" if self.bottom == "skirt" else "grey"))
        self.glasses = bool(spec.get("glasses", False))
        self.cache = {}
        self.SW, self.SH = int(self.height * 0.80), int(self.height + 40)

    def sprite(self, eyes_open=True, mouth="smile"):
        key = (eyes_open, mouth)
        if key not in self.cache:
            self.cache[key] = self._draw(eyes_open, mouth)
        return self.cache[key]

    def _draw(self, eyes_open, mouth):
        Ht = self.height
        cv = Image.new("RGBA", (self.SW, self.SH), (0, 0, 0, 0))
        cx = self.SW / 2
        hh = Ht * self.head_frac
        hw = hh * (1.0 if self.group == "kid" else 0.93)
        head_cy = 20 + hh / 2
        body_top = head_cy + hh * 0.44
        rest = Ht - (body_top - 20)
        torso_h = rest * (0.46 if self.group != "kid" else 0.50)
        legs_h = rest * 0.42
        shoe_h = rest * 0.12
        torso_w = hw * (1.02 if self.group != "kid" else 0.86)
        if self.group == "senior": torso_w *= 1.04
        skin = PAL["skin"]

        # cabell llarg (darrere)
        if self.sex == "f" and self.group in ("young", "adult"):
            blob(cv, cx, head_cy + hh * 0.22, hw * 1.12, hh * 1.05, self.hair, "rrect", 0.45)
        # cames / faldilla
        legs_top = body_top + torso_h * 0.88
        if self.bottom == "skirt":
            sk_h = legs_h * 0.62
            for s in (-1, 1):
                blob(cv, cx + s * torso_w * 0.17, legs_top + sk_h + (legs_h - sk_h) / 2, torso_w * 0.20, legs_h - sk_h + 20, skin, "rrect", 0.4)
            blob(cv, cx, legs_top + sk_h / 2, torso_w * 0.98, sk_h, self.bottom_color, "rrect", 0.22)
        else:
            for s in (-1, 1):
                blob(cv, cx + s * torso_w * 0.21, legs_top + legs_h / 2, torso_w * 0.40, legs_h + 10, self.bottom_color, "rrect", 0.22)
        # sabates
        shoe_y = legs_top + legs_h + shoe_h * 0.25
        for s in (-1, 1):
            blob(cv, cx + s * torso_w * 0.22, shoe_y, torso_w * 0.44, shoe_h * 1.05, PAL["shoe"], "rrect", 0.48)
        # tors: camisa + jaqueta oberta
        blob(cv, cx, body_top + torso_h / 2, torso_w, torso_h, self.shirt, "rrect", 0.28)
        for s in (-1, 1):
            blob(cv, cx + s * torso_w * 0.30, body_top + torso_h / 2 + 4, torso_w * 0.42, torso_h * 1.02, self.jacket, "rrect", 0.30)
        # coll
        for s in (-1, 1):
            blob(cv, cx + s * torso_w * 0.10, body_top + torso_h * 0.07, torso_w * 0.22, torso_h * 0.13, self.shirt, "ellipse")
        # botons
        for i in range(3):
            blob(cv, cx, body_top + torso_h * (0.30 + i * 0.2), 11, 11, tuple(int(v * 0.8) for v in self.shirt))
        # braços i mans
        arm_w, arm_h = torso_w * 0.24, torso_h * 0.86
        for s in (-1, 1):
            ax = cx + s * (torso_w / 2 + arm_w * 0.18)
            blob(cv, ax, body_top + arm_h / 2 + 8, arm_w, arm_h, self.jacket, "rrect", 0.48)
            blob(cv, ax, body_top + arm_h + 14, arm_w * 0.95, arm_w * 0.95, skin)
        # orelles
        for s in (-1, 1):
            blob(cv, cx + s * hw * 0.49, head_cy + hh * 0.05, hw * 0.18, hh * 0.22, skin)
        # cap
        blob(cv, cx, head_cy, hw, hh, skin, "ellipse")
        d = ImageDraw.Draw(cv)
        # galtes
        cheek = Image.new("RGBA", cv.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cheek)
        for s in (-1, 1):
            x, y, r = cx + s * hw * 0.27, head_cy + hh * 0.17, hw * 0.10
            cd.ellipse([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=PAL["cheek"] + (120,))
        cv.alpha_composite(cheek.filter(ImageFilter.GaussianBlur(hw * 0.035)))
        # ulls
        ink = PAL["ink"]
        ey = head_cy + hh * 0.04
        for s in (-1, 1):
            ex = cx + s * hw * 0.18
            if eyes_open and not (self.group == "senior" and self.sex == "f"):
                r = hw * 0.045
                d.ellipse([ex - r, ey - r * 1.15, ex + r, ey + r * 1.15], fill=ink)
                d.ellipse([ex - r * 0.25 + r * 0.3, ey - r * 0.75, ex + r * 0.25 + r * 0.3, ey - r * 0.25], fill=(255, 255, 255))
            else:
                r = hw * 0.055
                d.arc([ex - r, ey - r * 0.6, ex + r, ey + r * 0.9], 200, 340, fill=ink, width=max(3, int(hw * 0.016)))
        # celles
        brow = self.hair if self.group != "kid" else tuple(int(v * 0.8) for v in self.hair)
        for s in (-1, 1):
            bx = cx + s * hw * 0.19
            by = ey - hh * (0.11 if mouth != "sad" else 0.12)
            tilt = 4 if mouth == "sad" else 0
            bw = hw * (0.15 if self.group == "senior" else 0.11)
            thick = hh * (0.045 if self.group == "senior" else 0.022)
            d.rounded_rectangle([bx - bw / 2, by - thick / 2 - s * tilt * 0, bx + bw / 2, by + thick / 2], int(thick / 2), fill=brow)
        # nas
        blob(cv, cx, head_cy + hh * 0.13, hw * 0.11, hh * 0.08, tuple(int(v * 0.95) for v in skin))
        # bigoti (avi)
        if self.group == "senior" and self.sex == "m":
            for s in (-1, 1):
                blob(cv, cx + s * hw * 0.075, head_cy + hh * 0.205, hw * 0.17, hh * 0.075, self.hair)
        # boca
        my = head_cy + hh * (0.27 if self.group == "senior" and self.sex == "m" else 0.24)
        mw = hw * 0.15
        lw = max(3, int(hw * 0.018))
        if mouth == "smile":
            d.arc([cx - mw / 2, my - mw * 0.45, cx + mw / 2, my + mw * 0.35], 20, 160, fill=ink, width=lw)
        elif mouth == "open":
            d.ellipse([cx - mw * 0.32, my - mw * 0.15, cx + mw * 0.32, my + mw * 0.42], fill=(150, 70, 70))
        elif mouth == "grin":
            d.chord([cx - mw * 0.55, my - mw * 0.25, cx + mw * 0.55, my + mw * 0.55], 0, 180, fill=(150, 70, 70))
        elif mouth == "sad":
            d.arc([cx - mw / 2, my, cx + mw / 2, my + mw * 0.7], 200, 340, fill=ink, width=lw)
        else:
            d.line([cx - mw * 0.35, my + 4, cx + mw * 0.35, my + 4], fill=ink, width=lw)
        # cabell
        self._hair(cv, cx, head_cy, hw, hh)
        if self.glasses:
            gr = hw * 0.11
            for s in (-1, 1):
                gx = cx + s * hw * 0.18
                d.ellipse([gx - gr, ey - gr, gx + gr, ey + gr], outline=(120, 80, 60), width=max(4, int(hw * 0.02)))
            d.line([cx - hw * 0.07, ey, cx + hw * 0.07, ey], fill=(120, 80, 60), width=max(4, int(hw * 0.02)))
        return cv

    def _hair(self, cv, cx, cy, hw, hh):
        hc, g = self.hair, self.group
        if g == "senior" and self.sex == "m":
            for s in (-1, 1):
                blob(cv, cx + s * hw * 0.44, cy - hh * 0.06, hw * 0.20, hh * 0.30, hc)
            blob(cv, cx - hw * 0.05, cy - hh * 0.47, hw * 0.16, hh * 0.08, hc)
        elif g == "senior":
            pts = [(-0.38, -0.12, 0.30), (-0.25, -0.33, 0.36), (0.0, -0.40, 0.40), (0.25, -0.33, 0.36), (0.38, -0.12, 0.30)]
            for x, y, r in pts:
                blob(cv, cx + x * hw, cy + y * hh, hw * r, hh * r * 0.85, hc)
            blob(cv, cx + hw * 0.05, cy - hh * 0.58, hw * 0.30, hh * 0.22, hc)
        elif self.sex == "f":
            blob(cv, cx, cy - hh * 0.22, hw * 1.08, hh * 0.66, hc, "ellipse")
            blob(cv, cx - hw * 0.20, cy - hh * 0.13, hw * 0.58, hh * 0.24, hc, "ellipse")
            blob(cv, cx + hw * 0.24, cy - hh * 0.15, hw * 0.46, hh * 0.20, hc, "ellipse")
            if g == "kid":
                for s in (-1, 1):
                    blob(cv, cx + s * hw * 0.60, cy - hh * 0.05, hw * 0.24, hh * 0.32, hc)
        else:
            blob(cv, cx, cy - hh * 0.24, hw * 1.04, hh * 0.60, hc, "ellipse")
            blob(cv, cx + hw * 0.14, cy - hh * 0.16, hw * 0.56, hh * 0.20, hc, "ellipse")
            if g == "adult":
                for s in (-1, 1):
                    blob(cv, cx + s * hw * 0.45, cy - hh * 0.12, hw * 0.14, hh * 0.22, (170, 160, 155))

# ---------------------------------------------------------------- helpers
def ease_out_back(t):
    c1, c3 = 1.70158, 2.70158
    t = min(max(t, 0), 1)
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
def ease_in_out(t):
    t = min(max(t, 0), 1)
    return t * t * (3 - 2 * t)

def background(top, bottom):
    top, bottom = np.array(col(top), np.float32), np.array(col(bottom), np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    horizon = 0.70
    t = np.clip(y / horizon, 0, 1) ** 1.3
    img = top * (1 - t) + bottom * t
    floor = np.clip((y - horizon) / (1 - horizon), 0, 1)
    img = img * (1 - 0.04 * floor)
    xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :, None]
    vign = 1 - 0.10 * (xx ** 2) - 0.06 * ((y * 2 - 1) ** 2)
    img = np.clip(img * vign + RNG.normal(0, 1.5, (H, W, 1)), 0, 255)
    img = np.broadcast_to(img, (H, W, 3)).astype(np.uint8)
    return Image.fromarray(img, "RGB").convert("RGBA")

def wrap(text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if fnt.getlength(t) <= maxw: cur = t
        else:
            if cur: lines.append(cur)
            cur = wd
    if cur: lines.append(cur)
    return lines

def text_block(lines_spec, maxw):
    """lines_spec: [(text, size, color, bold)] -> RGBA image of stacked centered text."""
    rendered = []
    for text, size, colr, bold in lines_spec:
        f = font(size, bold)
        for ln in wrap(text, f, maxw):
            bb = f.getbbox(ln)
            rendered.append((ln, f, colr, bb[3] - bb[1] + int(size * 0.28), bb))
    hgt = sum(r[3] for r in rendered)
    wid = int(max(r[1].getlength(r[0]) for r in rendered)) + 10
    img = Image.new("RGBA", (wid, hgt + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = 0
    for ln, f, colr, lh, bb in rendered:
        x = (wid - f.getlength(ln)) / 2
        d.text((x, y - bb[1]), ln, font=f, fill=col(colr))
        y += lh
    return img

def card(content, color=(255, 250, 244), pad=(60, 44)):
    cw, ch = content.width + pad[0] * 2, content.height + pad[1] * 2
    base = clay(cw, ch, col(color), "rrect", 0.22, soft=0.05)
    shadow = Image.new("RGBA", (cw + 80, ch + 80), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([40, 52, cw + 40, ch + 52], int(min(cw, ch) * 0.22), fill=(60, 40, 30, 60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    shadow.alpha_composite(base, (40, 40))
    shadow.alpha_composite(content, (40 + pad[0], 40 + pad[1]))
    return shadow

def arrow(direction, color):
    w, h = 118, 118
    if direction == "up":
        poly = [(w * 0.5, 4), (w - 4, h * 0.55), (w * 0.68, h * 0.55), (w * 0.68, h - 4), (w * 0.32, h - 4), (w * 0.32, h * 0.55), (4, h * 0.55)]
    else:
        poly = [(w * 0.5, h - 4), (w - 4, h * 0.45), (w * 0.68, h * 0.45), (w * 0.68, 4), (w * 0.32, 4), (w * 0.32, h * 0.45), (4, h * 0.45)]
    return clay(w, h, col(color), "poly", poly=poly)

def scaled(img, s):
    if s <= 0.01: return None
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)

def fmt_num(v, decimals):
    s = f"{v:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

# ---------------------------------------------------------------- audio
def make_audio(path, dur, pops):
    sr = 44100
    n = int(sr * dur)
    out = np.zeros(n, np.float32)
    def pluck(freq, t0, amp=0.18, length=1.6):
        i0 = int(t0 * sr); L = int(length * sr)
        if i0 >= n: return
        period = int(sr / freq)
        buf = RNG.uniform(-1, 1, period).astype(np.float32)
        y = np.zeros(L, np.float32)
        for i in range(L):
            y[i] = buf[i % period]
            buf[i % period] = 0.996 * 0.5 * (buf[i % period] + buf[(i + 1) % period])
        seg = y[: n - i0] * amp
        out[i0: i0 + len(seg)] += seg
    chords = [[261.6, 329.6, 392.0], [196.0, 246.9, 392.0], [220.0, 261.6, 329.6], [174.6, 220.0, 349.2]]
    beat = 60 / 104
    t, k = 0.0, 0
    while t < dur - 0.5:
        ch = chords[(k // 4) % 4]
        for j, f in enumerate(ch):
            pluck(f, t + j * 0.018, 0.10 if k % 2 else 0.14)
        if k % 2 == 0:
            pluck(ch[0] / 2, t, 0.12, 1.2)
        t += beat / 2 if k % 4 in (1, 3) else beat / 2
        k += 1
    for p in pops:
        i0 = int(p * sr); L = int(0.12 * sr)
        tt = np.arange(L) / sr
        tone = np.sin(2 * np.pi * (500 + 900 * tt / 0.12) * tt) * np.exp(-tt * 30) * 0.35
        out[i0: i0 + L] += tone[: max(0, n - i0)]
    fade = int(1.2 * sr)
    out[-fade:] *= np.linspace(1, 0, fade)
    out[: int(0.05 * sr)] *= np.linspace(0, 1, int(0.05 * sr))
    out = np.clip(out / max(1e-6, np.abs(out).max()) * 0.7, -1, 1)
    data = (out * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(data.tobytes())

# ---------------------------------------------------------------- render
def render(spec, out_path):
    dur = float(spec.get("duration", 20))
    nframes = int(dur * FPS)
    bg = background(tuple(spec.get("bg_top", [207, 216, 234])), tuple(spec.get("bg_bottom", [240, 238, 243])))
    chars = [Character(c) for c in spec["characters"]]
    n = len(chars)
    floor_y = 1440
    slots = [W / 2] if n == 1 else [W * (0.33 + 0.34 * i / (n - 1)) for i in range(n)] if n == 2 else [W * (0.22 + 0.56 * i / (n - 1)) for i in range(n)]
    scenes = spec["scenes"]
    accent_up, accent_down = (226, 128, 112), (122, 176, 140)

    # pre-render overlays
    for sc in scenes:
        kind = sc["type"]
        if kind == "title":
            content = text_block([(sc.get("kicker", ""), 40, (150, 120, 110), False)] * bool(sc.get("kicker")) + [(sc["text"], 78, "ink", True)], 820)
            sc["_img"] = card(content)
        elif kind == "stat":
            sc["_label"] = text_block([(sc["label"], 40, (130, 110, 100), False)], 820)
            sc["_sub"] = text_block([(sc["sub"], 36, (130, 110, 100), False)], 820) if sc.get("sub") else None
            sc["_arrow"] = arrow(sc["arrow"], accent_up if sc["arrow"] == "up" else accent_down) if sc.get("arrow") else None
        elif kind == "bubble":
            content = text_block([(sc["text"], 46, "ink", True)], 520)
            sc["_img"] = card(content, (255, 255, 255), (40, 30))
        elif kind == "outro":
            content = text_block([(sc.get("text", "Segueix per entendre el mercat"), 50, "ink", True),
                                  (sc.get("handle", "@pepnivell"), 64, (196, 120, 104), True)], 820)
            sc["_img"] = card(content)
    src = spec.get("source", "Font: idealista")
    src_img = text_block([(src, 30, (120, 110, 105), False)], 900)

    pops = [sc["t0"] + 0.05 for sc in scenes if sc["type"] != "bubble"]
    tmp_audio = out_path + ".wav"
    make_audio(tmp_audio, dur, pops)

    proc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", tmp_audio, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)

    def char_state(i, t):
        mood, speaking = "idle", False
        for sc in scenes:
            if sc["t0"] <= t < sc["t1"]:
                m = sc.get("moods")
                if m: mood = m[i] if isinstance(m, list) else m
                if sc["type"] == "bubble" and sc.get("speaker") == i: speaking = True
        return mood, speaking

    for f in range(nframes):
        t = f / FPS
        frame = bg.copy()
        # personatges
        for i, ch in enumerate(chars):
            mood, speaking = char_state(i, t)
            x = slots[i]
            dy, rot, sq = 0.0, 0.0, 1.0
            ph = t * 2 * math.pi + i * 1.7
            enter = spec.get("enter_time", 1.6)
            if t < enter:
                p = ease_in_out(t / enter)
                start = -ch.SW if x < W / 2 else W + ch.SW
                x = start + (x - start) * p
                dy = -abs(math.sin(t * 9)) * 26
                rot = math.sin(t * 9) * 4
            elif mood == "happy":
                hop = abs(math.sin((t - enter) * 6 + i))
                dy = -hop * 70
                sq = 1 + (0.05 if hop < 0.15 else -0.02)
            elif mood == "worried":
                rot = math.sin(t * 14 + i) * 1.5
                sq = 0.98
            else:
                sq = 1 + math.sin(ph * 0.8) * 0.012
                rot = math.sin(ph * 0.5) * 1.2
            blink = ((t + i * 0.9) % 3.1) < 0.12
            mouth = {"happy": "grin", "worried": "sad", "idle": "smile", "neutral": "flat"}.get(mood, "smile")
            if speaking: mouth = "open" if int(t * 7) % 2 == 0 else "smile"
            spr = ch.sprite(not blink, mouth)
            if sq != 1.0:
                spr = spr.resize((int(spr.width * (2 - sq)), int(spr.height * sq)), Image.BILINEAR)
            if abs(rot) > 0.05:
                spr = spr.rotate(rot, resample=Image.BILINEAR, center=(spr.width / 2, spr.height), expand=False)
            # ombra al terra
            sw = ch.SW * 0.7 * (1 - min(0.4, -dy / 250))
            sh_img = Image.new("RGBA", (int(sw) + 60, 90), (0, 0, 0, 0))
            ImageDraw.Draw(sh_img).ellipse([30, 30, 30 + sw, 60], fill=(70, 55, 60, int(70 * (1 - min(0.5, -dy / 200)))))
            sh_img = sh_img.filter(ImageFilter.GaussianBlur(10))
            frame.alpha_composite(sh_img, (int(x - sh_img.width / 2), int(floor_y - 45)))
            frame.alpha_composite(spr, (int(x - spr.width / 2), int(floor_y - spr.height + dy + 12)))
        # superposicions
        for sc in scenes:
            t0, t1 = sc["t0"], sc["t1"]
            if not (t0 <= t < t1): continue
            pin = ease_out_back((t - t0) / 0.38)
            pout = 1 - ease_in_out((t - (t1 - 0.25)) / 0.25) if t > t1 - 0.25 else 1
            s = pin * pout
            kind = sc["type"]
            if kind in ("title", "outro"):
                im = scaled(sc["_img"], s)
                if im: put(frame, im, W / 2, sc.get("y", 450))
            elif kind == "bubble":
                im = scaled(sc["_img"], s)
                if im:
                    sx = slots[sc["speaker"]]
                    bx = min(max(sx, im.width / 2 + 30), W - im.width / 2 - 30)
                    head_top = floor_y - chars[sc["speaker"]].height
                    put(frame, im, bx, head_top - im.height / 2 + 30)
            elif kind == "stat":
                p = ease_in_out((t - t0 - 0.2) / sc.get("count", 1.2))
                val = sc["value"] * p
                vtxt = sc.get("prefix", "") + fmt_num(val, sc.get("decimals", 0)) + sc.get("suffix", "")
                vimg = text_block([(vtxt, 112, "ink", True)], 900)
                parts = [sc["_label"], vimg] + ([sc["_sub"]] if sc["_sub"] is not None else [])
                cw = max(p_.width for p_ in parts) + (170 if sc["_arrow"] else 0)
                chh = sum(p_.height for p_ in parts) + 20
                content = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
                y = 0
                for p_ in parts:
                    content.alpha_composite(p_, (int((cw - p_.width) / 2 + (0 if not sc["_arrow"] or p_ is not vimg else 80)), y))
                    if p_ is vimg and sc["_arrow"] is not None:
                        wob = int(math.sin(t * 6) * 6)
                        ax = int((cw - p_.width) / 2 + 80 - 170)
                        content.alpha_composite(sc["_arrow"], (max(0, ax), y + 18 + wob))
                    y += p_.height + 10
                im = scaled(card(content), s)
                if im: put(frame, im, W / 2, sc.get("y", 450))
        put(frame, src_img, W / 2, 1505)
        proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close()
    proc.wait()
    os.remove(tmp_audio)
    if proc.returncode != 0:
        raise SystemExit("ffmpeg ha fallat")

if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    render(spec, sys.argv[2])
    print("OK", sys.argv[2])
