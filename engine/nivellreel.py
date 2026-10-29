#!/usr/bin/env python3
"""
nivellreel.py — Reels de @pepnivell amb la imatge de Pep i el robot de nivell.

Ús:  python3 nivellreel.py episodi.json sortida.mp4 [--scene cartoon|real|estand|botiga] [--seed N]

- Les dues figures es retallen de la imatge original i entren cadascuna pel seu
  costat de la pantalla fins a quedar juntes al centre.
- Només 2 dades per vídeo + una frase de cada presentador + conclusió.
- Cada vídeo té un estil diferent (fons, targetes, càmera, entrada de targetes,
  música, gestos del robot), triat a partir de l'id de l'episodi o --seed.
- To seriós: moviments suaus, el cap de Pep no es deforma mai.
"""
import json, math, os, random, subprocess, sys, wave, hashlib
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
FONT_DIR = "/usr/share/fonts/truetype/google-fonts"
NAVY, RED = (28, 36, 62), (214, 36, 44)

def font(size, weight="Bold"):
    for name in (f"Poppins-{weight}.ttf", "Poppins-Bold.ttf"):
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)

def ease(t):
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)
def ease_out(t):
    t = min(max(t, 0.0), 1.0); return 1 - (1 - t) ** 3
def env(t, t0, t1, r=0.4):
    if t < t0 or t > t1: return 0.0
    return min(ease((t - t0) / r), ease((t1 - t) / r))
def smooth(e0, e1, x):
    v = np.clip((x - e0) / (e1 - e0), 0, 1); return v * v * (3 - 2 * v)

# ------------------------------------------------------------------ estils (variació per vídeo)
BGS = ["studio", "warm", "cool", "navytop", "brand"]
CARDS = ["white", "navy", "bar"]
CAMS = ["in", "out", "drift"]
CARD_IN = ["up", "left", "scale"]
MUSIC = [
    {"prog": [[220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 261.6, 329.6], [196.0, 246.9, 293.7]], "bar": 2.5, "inst": "piano"},
    {"prog": [[261.6, 329.6, 392.0], [220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 246.9, 293.7]], "bar": 2.2, "inst": "pluck"},
    {"prog": [[146.8, 174.6, 220.0], [116.5, 146.8, 174.6], [130.8, 164.8, 196.0], [110.0, 138.6, 164.8]], "bar": 2.8, "inst": "pad"},
    {"prog": [[196.0, 246.9, 293.7], [164.8, 196.0, 246.9], [130.8, 164.8, 196.0], [146.8, 185.0, 220.0]], "bar": 2.0, "inst": "piano"},
    {"prog": [[174.6, 220.0, 261.6], [196.0, 233.1, 293.7], [220.0, 261.6, 329.6], [196.0, 246.9, 293.7]], "bar": 2.4, "inst": "pluck"},
]

def pick_style(seed):
    r = random.Random(seed)
    bg = r.choice(BGS)
    card = r.choice(["white", "bar"] if bg == "navytop" else CARDS)
    return {"bg": bg, "card": card, "cam": r.choice(CAMS), "card_in": r.choice(CARD_IN),
            "music": r.randrange(len(MUSIC)), "transpose": r.choice([0.89, 0.94, 1.0, 1.06, 1.12]),
            "robot_first": r.random() < 0.5, "stagger": r.uniform(0.0, 0.5), "enter_dur": r.uniform(1.4, 1.9),
            "extra_look": r.random() < 0.6, "tilt_dir": r.choice([-1, 1]), "seed": seed}

def make_bg(style):
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :, None]
    b = style["bg"]
    def grad(top, bot, p=1.0):
        t = yy ** p
        return np.array(top, np.float32) * (1 - t) + np.array(bot, np.float32) * t
    if b == "studio": img = grad((246, 247, 250), (252, 252, 253))
    elif b == "warm": img = grad((251, 247, 241), (240, 233, 224), 1.4)
    elif b == "cool": img = grad((236, 241, 249), (250, 251, 253), 0.8)
    elif b == "navytop":
        t = smooth(0.18, 0.46, yy)
        img = np.array(NAVY, np.float32) * (1 - t) + np.array((248, 249, 251), np.float32) * t
    else:  # brand: blanc amb una taca vermella molt suau a la cantonada
        img = grad((250, 250, 251), (244, 244, 246))
        d = np.sqrt((xx - 0.95) ** 2 + ((yy * 2 - 1) / 1.78 + 0.95) ** 2)
        img = img + (np.array(RED, np.float32) - img) * (0.10 * np.exp(-(d ** 2) / 0.10))
    img = np.broadcast_to(img, (H, W, 3)).copy()
    # terra: lleugera ombra radial on es posen els personatges
    fl = np.exp(-(((xx[..., 0]) / 0.75) ** 2 + ((yy[..., 0] - 0.78) / 0.07) ** 2))
    img *= (1 - 0.035 * fl[..., None])
    if b == "cool":
        dots = np.zeros((H, W), np.float32)
        dots[::36, ::36] = 1
        dots = cv2.GaussianBlur(dots, (0, 0), 1.2) * 7
        img -= dots[..., None] * 4
    img += np.random.default_rng(1).normal(0, 1.0, (H, W, 1))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

# ------------------------------------------------------------------ figures
def draw_eyes(img, e, P, S, expr, blink, talk, look):
    """Dibuixa els ulls de llum del robot sobre img (BGR o BGRA, in-place).
    e: config d'ulls; P: funció coord. imatge original -> coord. de img; S: escala."""
    k = 4
    if e.get("style", "arc") == "bar":
        c = P(e["c"]) + np.array([9 * S * look, 2 * S * look], np.float32)
        ang = math.radians(e.get("angle", 0)); ca, sa = math.cos(ang), math.sin(ang)
        half = e["w"] * S; th0 = e.get("th", 6) * S
        pad = int(half * 1.9)
        bx0, by0 = int(c[0] - pad), int(c[1] - pad * 0.8)
        bw, bh = int(pad * 2), int(pad * 1.6)
        layer = np.zeros((bh * k, bw * k), np.float32)
        def pt(lx, ly):
            return (int((c[0] + lx * ca - ly * sa - bx0) * k), int((c[1] + lx * sa + ly * ca - by0) * k))
        L = half * (0.55 if blink else 1.0)
        if talk > 0: L = half * (0.55 + 0.45 * talk)
        th = th0 * (0.45 if blink else 1.0)
        if expr == "worried" and not blink:
            cv2.line(layer, pt(-L, -th * 0.5), pt(L * 0.0, th * 0.6), 1.0, max(2, int(th * k)), cv2.LINE_AA)
            cv2.line(layer, pt(0, th * 0.6), pt(L, -th * 0.5), 1.0, max(2, int(th * k)), cv2.LINE_AA)
        else:
            cv2.line(layer, pt(-L, 0), pt(L, 0), 1.0, max(2, int(th * k)), cv2.LINE_AA)
        glow_s, core_col = 3.0, np.array([255, 240, 140], np.float32)
        blue = np.array([255, 215, 60], np.float32)
    else:
        Lc, Rc = P(e["L"]), P(e["R"])
        shift = np.array([10 * S * look, 3 * S * look], np.float32)
        Lc, Rc = Lc + shift, Rc + shift
        ang = math.atan2(Rc[1] - Lc[1], Rc[0] - Lc[0]); ca, sa = math.cos(ang), math.sin(ang)
        mid = (Lc + Rc) / 2
        pad = int(e["w"] * 3.2 * S)
        bx0, by0 = int(mid[0] - pad * 1.5), int(mid[1] - pad)
        bw, bh = int(pad * 3), int(pad * 2.4)
        layer = np.zeros((bh * k, bw * k), np.float32)
        def pt(lx, ly, base):
            return (int((base[0] + lx * ca - ly * sa - bx0) * k), int((base[1] + lx * sa + ly * ca - by0) * k))
        ew = e["w"] * S; th = max(2, int(e["w"] * 0.26 * S * k))
        for side, cen in ((-1, Lc), (1, Rc)):
            if blink: pts = [pt(-ew, 2, cen), pt(ew, 2, cen)]
            elif expr == "worried":
                lift = 0.28 * ew * (-side)
                pts = [pt(-ew, -lift, cen), pt(ew, lift, cen)] if side < 0 else [pt(-ew, lift, cen), pt(ew, -lift, cen)]
            else:
                hgt = ew * (0.62 if expr == "happy" else 0.48)
                pts = [pt(ew * math.cos(q), -hgt * math.sin(q) + 0.2 * ew, cen) for q in np.linspace(math.pi, 0, 26)]
            cv2.polylines(layer, [np.array(pts, np.int32)], False, 1.0, th, cv2.LINE_AA)
        if talk > 0:
            mw = (0.35 + 0.65 * talk) * ew * 0.9
            mc = mid + np.array([-sa, ca], np.float32) * ew * 2.0
            cv2.line(layer, pt(-mw, 0, mc), pt(mw, 0, mc), 1.0, max(2, int(th * 0.85)), cv2.LINE_AA)
        glow_s, core_col = 5.0, np.array([255, 245, 225], np.float32)
        blue = np.array([255, 190, 70], np.float32)
    h, w = img.shape[:2]
    if bx0 < 0 or by0 < 0 or bx0 + bw > w or by0 + bh > h: return
    core = cv2.resize(layer, (bw, bh), interpolation=cv2.INTER_AREA)
    glow = cv2.GaussianBlur(core, (0, 0), glow_s * S) * 1.6 + cv2.GaussianBlur(core, (0, 0), 1.8 * S) * 0.9
    sub = img[by0:by0 + bh, bx0:bx0 + bw, :3].astype(np.float32)
    sub = sub + glow[..., None] * blue * 0.55
    sub = sub * (1 - core[..., None]) + core[..., None] * (core_col * 0.6 + blue * 0.4)
    img[by0:by0 + bh, bx0:bx0 + bw, :3] = np.clip(sub, 0, 255).astype(np.uint8)

def remove_eyes(img, e):
    """Esborra els ulls/la barra de llum original (inpainting)."""
    roi = img.astype(np.int32)
    b, r = roi[..., 0], roi[..., 2]
    glow = ((b > 55) & (b > r + 22)).astype(np.uint8)
    keep = np.zeros_like(glow)
    if e.get("style", "arc") == "bar":
        cv2.ellipse(keep, (int(e["c"][0]), int(e["c"][1])), (int(e["w"] * 1.3), int(e.get("th", 6) * 2.2)), e.get("angle", 0), 0, 360, 1, -1)
    else:
        ang = math.degrees(math.atan2(e["R"][1] - e["L"][1], e["R"][0] - e["L"][0]))
        for c in (e["L"], e["R"]):
            cv2.ellipse(keep, (int(c[0]), int(c[1])), (int(e["w"] * 1.45), int(e["w"] * 1.05)), ang, 0, 360, 1, -1)
    m = cv2.dilate((glow & keep) * 255, np.ones((7, 7), np.uint8))
    out = img.copy()
    out[..., :3] = cv2.inpaint(np.ascontiguousarray(img[..., :3]), m, 6, cv2.INPAINT_TELEA)
    return out

class Hosts:
    """Escenari 'cutout': Pep i el robot retallats, entren cadascun pel seu costat."""
    kind = "cutout"
    def __init__(self, rig_path):
        cfg = self.cfg = json.load(open(rig_path))
        base = os.path.dirname(os.path.abspath(rig_path))
        raw = cv2.imread(os.path.join(base, cfg["image"]), cv2.IMREAD_UNCHANGED)
        if cfg.get("alpha_source"):
            img = np.ascontiguousarray(raw[..., :3]); fg = (raw[..., 3] > 40).astype(np.uint8) * 255
            alpha_src = raw[..., 3]
        else:
            img = raw[..., :3] if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            img = np.ascontiguousarray(img); fg = self._segment(img, cfg); alpha_src = None
        h, w = img.shape[:2]
        img = remove_eyes(img, cfg["robot"]["eyes"])
        yy, xx = np.mgrid[0:h, 0:w]
        split = np.interp(yy, [p[1] for p in cfg["split"]], [p[0] for p in cfg["split"]])
        pep = (xx >= split)
        band = np.zeros((h, w), bool)
        if "navy_band" in cfg:
            x0, y0, x1, y1 = cfg["navy_band"]
            bb, gg, rr = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
            navy = (bb > rr + 8) & (bb < 140) & (gg < 120)
            band = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
            pep = np.where(band, navy, pep)
        fgm = fg > 0
        if "floor_cut" in cfg:
            ycut = np.full(w, cfg.get("floor_default", h), np.int32)
            for a_, b_, yv in cfg["floor_cut"]: ycut[a_:b_] = yv
            fgm &= yy < ycut[None, :]
            light = (img.min(axis=2) > 200)
            fgm &= ~(band & light & pep)
        self.S = S = cfg["scale"]
        layers = {}
        for name, m in (("robot", fgm & ~pep), ("pep", fgm & pep)):
            m8 = (m * 255).astype(np.uint8)
            if alpha_src is None:
                m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                n, lab, st, _ = cv2.connectedComponentsWithStats(m8)
                keep = np.zeros_like(m8)
                for i in range(1, n):
                    if st[i, 4] > 1500: keep[lab == i] = 255
                keep = cv2.erode(keep, np.ones((3, 3), np.uint8))
                a = cv2.GaussianBlur(keep, (0, 0), 0.9)
            else:
                keep = m8; a = np.where(m, alpha_src, 0).astype(np.uint8)
            ys, xs = np.nonzero(keep)
            bx0, by0 = max(0, xs.min() - 6), max(0, ys.min() - 6)
            bx1, by1 = min(w, xs.max() + 7), min(h, ys.max() + 7)
            rgba = np.dstack([img[by0:by1, bx0:bx1], a[by0:by1, bx0:bx1]])
            rgba = cv2.resize(rgba, (int((bx1 - bx0) * S), int((by1 - by0) * S)), interpolation=cv2.INTER_AREA)
            layers[name] = {"bgra": rgba, "x0": bx0 * S, "y0": by0 * S}
        self.layers = layers
        self.group_x0, self.group_x1 = cfg["group_x"][0] * S, cfg["group_x"][1] * S
        self.feet_y = cfg["feet_y"] * S
        L = layers["robot"]; lh, lw = L["bgra"].shape[:2]
        ly, lx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        self.rX, self.rY = lx, ly
        hc = self.rp(cfg["robot"]["head"]["c"]); hr = cfg["robot"]["head"]["r"] * S
        d = np.sqrt((lx - hc[0]) ** 2 + (ly - hc[1]) ** 2)
        self.head_mask = (1 - smooth(hr, hr * 1.55, d)).astype(np.float32)
        self.head_pivot = self.rp(cfg["robot"]["head"]["pivot"])

    def rp(self, p):
        L = self.layers["robot"]
        return np.array([p[0] * self.S - L["x0"], p[1] * self.S - L["y0"]], np.float32)

    @staticmethod
    def _segment(img, cfg):
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
        seeds += [tuple(s) for s in cfg.get("bg_seeds", []) if img[s[1], s[0]].min() > 240]
        for s in seeds: cv2.floodFill(img.copy(), mask, s, (0, 0, 0), (24,) * 3, (24,) * 3, flags)
        fg = (mask[1:-1, 1:-1] == 0).astype(np.uint8) * 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        keep = np.zeros_like(fg)
        for i in range(1, n):
            if st[i, 4] > 3000: keep[lab == i] = 255
        inv = 255 - keep
        n2, l2, s2, _ = cv2.connectedComponentsWithStats(inv)
        for i in range(1, n2):
            if s2[i, 4] < 1500: keep[l2 == i] = 255
        return keep

    def robot_frame(self, expr, blink, talk, look, tilt):
        bgra = self.layers["robot"]["bgra"].copy()
        draw_eyes(bgra, self.cfg["robot"]["eyes"], self.rp, self.S, expr, blink, talk, look)
        a = tilt + look * math.radians(3.5)
        if abs(a) > 1e-5 or look > 0:
            ca, sa = math.cos(a), math.sin(a)
            px, py = self.head_pivot
            rx, ry = self.rX - px, self.rY - py
            tx = look * 4 * self.S
            dx = (ca * rx - sa * ry + px + tx - self.rX) * self.head_mask
            dy = (sa * rx + ca * ry + py - self.rY) * self.head_mask
            bgra = cv2.remap(bgra, self.rX - dx, self.rY - dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return bgra

class PhotoStage:
    """Escenari 'photo': foto real amb fons, zoom lent de càmera i ulls del robot animats."""
    kind = "photo"
    def __init__(self, rig_path):
        cfg = self.cfg = json.load(open(rig_path))
        base = os.path.dirname(os.path.abspath(rig_path))
        img = cv2.imread(os.path.join(base, cfg["image"]), cv2.IMREAD_COLOR)
        self.img = remove_eyes(img, cfg["robot"]["eyes"])
        self.h, self.w = img.shape[:2]
        self.box_w = cfg.get("box_w", W)
        self.box_h = int(self.box_w * self.h / self.w)
        self.box_x = (W - self.box_w) // 2
        self.box_y = int(cfg.get("box_cy", 1000) - self.box_h / 2)
        self.focus = np.array(cfg.get("focus", [self.w / 2, self.h / 2]), np.float32)
        # fons: la mateixa foto, ampliada, desenfocada i aclarida
        sc = max(W / self.w, H / self.h) * 1.1
        bg = cv2.resize(img, (int(self.w * sc), int(self.h * sc)), interpolation=cv2.INTER_LINEAR)
        oy, ox = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
        bg = cv2.GaussianBlur(bg[oy:oy + H, ox:ox + W], (0, 0), 38)
        bg = (bg.astype(np.float32) * 0.42 + 255 * 0.58).astype(np.uint8)
        self.bg = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("RGBA")
        r = 34
        mk = Image.new("L", (self.box_w, self.box_h), 0)
        ImageDraw.Draw(mk).rounded_rectangle([0, 0, self.box_w - 1, self.box_h - 1], r, fill=255)
        self.mask = mk
        sh = Image.new("RGBA", (self.box_w + 120, self.box_h + 120), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([60, 74, self.box_w + 60, self.box_h + 74], r, fill=(20, 24, 40, 80))
        self.shadow = sh.filter(ImageFilter.GaussianBlur(24))

    def transform(self, t, dur, zoom_in):
        p = ease(t / dur)
        z = 1.0 + 0.10 * (p if zoom_in else 1 - p)
        s = self.box_w / self.w * z
        c = np.array([self.w / 2, self.h / 2], np.float32) * (1 - (z - 1) * 4) + self.focus * ((z - 1) * 4)
        # assegura que la finestra no surt de la foto
        half = np.array([self.box_w / s / 2, self.box_h / s / 2], np.float32)
        c = np.minimum(np.maximum(c, half), np.array([self.w, self.h], np.float32) - half)
        tx = self.box_w / 2 - c[0] * s; ty = self.box_h / 2 - c[1] * s
        return s, tx, ty

    def frame(self, t, dur, zoom_in, expr, blink, talk, look):
        s, tx, ty = self.transform(t, dur, zoom_in)
        img = self.img.copy()
        draw_eyes(img, self.cfg["robot"]["eyes"], lambda p: np.array(p, np.float32), 1.0, expr, blink, talk, look)
        M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
        out = cv2.warpAffine(img, M, (self.box_w, self.box_h), flags=cv2.INTER_LANCZOS4 if s > 1.5 else cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if s > 1.5:
            blur = cv2.GaussianBlur(out, (0, 0), 1.2); out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
        pil.putalpha(self.mask)
        return pil, (s, tx, ty)

    def to_frame(self, p, tr):
        s, tx, ty = tr
        return self.box_x + p[0] * s + tx, self.box_y + p[1] * s + ty

def bgra_to_pil(a):
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA), "RGBA")


# ------------------------------------------------------------------ targetes
def text_lines(text, f, maxw):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if f.getlength(t) <= maxw: cur = t
        else:
            if cur: lines.append(cur)
            cur = wd
    if cur: lines.append(cur)
    return lines

def block(rows, maxw, align="center"):
    """rows: [(text, size, color, weight)] -> RGBA"""
    out = []
    for text, size, colr, weight in rows:
        f = font(size, weight)
        for ln in text_lines(text, f, maxw):
            bb = f.getbbox(ln); out.append((ln, f, colr, bb, int(size * 1.22)))
    wd = int(max(f.getlength(ln) for ln, f, *_ in out)) + 8
    ht = sum(o[4] for o in out) + 6
    im = Image.new("RGBA", (wd, ht), (0, 0, 0, 0)); d = ImageDraw.Draw(im); y = 0
    for ln, f, colr, bb, lh in out:
        x = (wd - f.getlength(ln)) / 2 if align == "center" else 0
        d.text((x, y + (lh - (bb[3] - bb[1])) / 2 - bb[1]), ln, font=f, fill=colr)
        y += lh
    return im

def panel(content, style, pad=(56, 40)):
    cw, ch = content.width + 2 * pad[0], content.height + 2 * pad[1]
    if style == "bar":
        im = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        ImageDraw.Draw(im).rounded_rectangle([0, pad[1] * 0.6, 10, ch - pad[1] * 0.6], 5, fill=RED + (255,))
        im.alpha_composite(content, (pad[0], pad[1]))
        return im
    fill = (255, 255, 255, 250) if style == "white" else NAVY + (246,)
    sh = Image.new("RGBA", (cw + 80, ch + 80), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([40, 52, cw + 40, ch + 52], 34, fill=(20, 24, 40, 55))
    sh = sh.filter(ImageFilter.GaussianBlur(20))
    box = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    ImageDraw.Draw(box).rounded_rectangle([0, 0, cw - 1, ch - 1], 34, fill=fill)
    sh.alpha_composite(box, (40, 40)); sh.alpha_composite(content, (40 + pad[0], 40 + pad[1]))
    return sh

def colors_for(style):
    on_dark = style["card"] == "navy" or (style["card"] == "bar" and style["bg"] == "navytop")
    if on_dark: return {"ink": (255, 255, 255), "muted": (190, 198, 220), "accent": (255, 120, 112)}
    return {"ink": NAVY, "muted": (96, 104, 124), "accent": RED}

def arrow_img(up, size=86):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    s = size
    pts = [(s * .5, s * .06), (s * .94, s * .52), (s * .66, s * .52), (s * .66, s * .94), (s * .34, s * .94), (s * .34, s * .52), (s * .06, s * .52)]
    if not up: pts = [(x, s - y) for x, y in pts]
    d.polygon(pts, fill=((214, 36, 44) if up else (46, 150, 96)) + (255,))
    return im

def fade(im, a):
    if a >= 0.999: return im
    im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * a))); return im

def fmt(v, dec):
    s = f"{v:,.{dec}f}"; return s.replace(",", "X").replace(".", ",").replace("X", ".")

# ------------------------------------------------------------------ música
def music(path, dur, style):
    sr = 44100; n = int(sr * dur); out = np.zeros(n, np.float32)
    M = MUSIC[style["music"]]; tr = style["transpose"]
    def note(freq, t0, length, amp, kind):
        i0 = int(t0 * sr); L = min(int(length * sr), n - i0)
        if L <= 0: return
        tt = np.arange(L) / sr
        if kind == "piano":
            e = np.exp(-tt * 1.6) * (1 - np.exp(-tt * 200))
            w = np.sin(2*np.pi*freq*tt) + 0.35*np.sin(4*np.pi*freq*tt) + 0.12*np.sin(6*np.pi*freq*tt)
        elif kind == "pluck":
            e = np.exp(-tt * 3.2) * (1 - np.exp(-tt * 300))
            w = np.sin(2*np.pi*freq*tt) + 0.5*np.sin(4*np.pi*freq*tt) * np.exp(-tt * 6)
        else:
            e = np.minimum(1, tt / 0.9) * np.minimum(1, (length - tt) / 0.9)
            w = np.sin(2*np.pi*freq*tt) + 0.5*np.sin(2*np.pi*freq*1.003*tt) + 0.25*np.sin(4*np.pi*freq*tt)
        out[i0:i0 + L] += (w * e * amp).astype(np.float32)
    t, k, bar = 0.0, 0, M["bar"]
    while t < dur:
        ch = [f * tr for f in M["prog"][k % 4]]
        for f in ch: note(f, t, bar + 0.4, 0.045, "pad")
        note(ch[0] / 2, t, bar, 0.07, "pad")
        if M["inst"] != "pad":
            steps = 4 if M["inst"] == "piano" else 8
            seq = [ch[0], ch[1], ch[2], ch[1] * 2, ch[2], ch[1], ch[0] * 2, ch[2]][:steps]
            for j, f in enumerate(seq): note(f * 2, t + j * bar / steps, 2.0, 0.055, M["inst"])
        else:
            note(ch[2] * 2, t + bar * 0.5, bar, 0.03, "piano")
        t += bar; k += 1
    fl = int(1.5 * sr); out[-fl:] *= np.linspace(1, 0, fl); out[:int(0.6 * sr)] *= np.linspace(0, 1, int(0.6 * sr))
    out = out / max(1e-6, np.abs(out).max()) * 0.6
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((out * 32767).astype(np.int16).tobytes())

# ------------------------------------------------------------------ guió -> línia de temps
def timeline(ep, style):
    r = random.Random(style["seed"] + 7)
    T = []
    T.append({"t0": 0.2, "t1": 3.3, "type": "title"})
    t = 3.3
    for i, st in enumerate(ep["stats"][:2]):
        d = 5.0 if i == 0 else 4.8
        T.append({"t0": t, "t1": t + d, "type": "stat", "i": i})
        t += d
    T.append({"t0": t, "t1": t + 3.2, "type": "takeaway"}); t += 3.2
    T.append({"t0": t, "t1": 20.0, "type": "outro"})
    lines = []
    if ep.get("pep_line"):
        s = T[1]; lines.append({"t0": s["t0"] + 1.6, "t1": s["t1"] - 0.2, "who": "pep", "text": ep["pep_line"]})
    if ep.get("robot_line"):
        s = T[2]; lines.append({"t0": s["t0"] + 1.6, "t1": s["t1"] - 0.2, "who": "robot", "text": ep["robot_line"]})
    return T, lines

SCENES = ["cartoon", "real", "estand", "botiga"]

def render(ep, out_path, scenes_dir=None, seed=None):
    if seed is None: seed = int(hashlib.md5(ep["id"].encode()).hexdigest()[:8], 16)
    scenes_dir = scenes_dir or os.path.join(HERE, "scenes")
    style = pick_style(seed)
    scene_name = ep.get("scene") or random.Random(seed + 3).choice(SCENES)
    rig_path = os.path.join(scenes_dir, scene_name + ".json")
    kind = json.load(open(rig_path)).get("kind", "cutout")
    if kind == "photo" and style["card"] == "bar": style["card"] = "white"
    print("escenari:", scene_name, "| estil:", {k: v for k, v in style.items() if k != "seed"}, flush=True)
    stage = PhotoStage(rig_path) if kind == "photo" else Hosts(rig_path)
    bg = stage.bg if kind == "photo" else make_bg(style)
    C = colors_for(style)
    T, lines = timeline(ep, style)
    dur = 20.0; nfr = int(dur * FPS)
    floor_y = 1470
    card_y = 330 if style["bg"] != "navytop" or kind == "photo" else 300
    if kind == "photo": card_y = min(330, stage.box_y - 190)
    maxw = 860

    for s in T:
        if s["type"] == "title":
            s["img"] = panel(block([(ep.get("kicker", ""), 38, C["accent"], "SemiBold"), (ep["title"], 74, C["ink"], "Bold")], maxw), style["card"])
        elif s["type"] == "takeaway":
            s["img"] = panel(block([(ep.get("takeaway_label", "En resum"), 36, C["accent"], "SemiBold"), (ep["takeaway"], 58, C["ink"], "Bold")], maxw), style["card"])
        elif s["type"] == "outro":
            s["img"] = panel(block([(ep.get("outro", "Dades del mercat immobiliari, cada 3 dies"), 44, C["ink"], "SemiBold"), ("@pepnivell", 68, C["accent"], "Bold")], maxw), style["card"])
        else:
            st = ep["stats"][s["i"]]
            s["label"] = block([(st["label"], 40, C["muted"], "Medium")], maxw)
            s["sub"] = block([(st["sub"], 36, C["muted"], "Medium")], maxw) if st.get("sub") else None
            s["arrow"] = arrow_img(st["trend"] == "up") if st.get("trend") in ("up", "down") else None
    for ln in lines:
        ln["img"] = panel(block([(ln["text"], 40, NAVY, "SemiBold")], 300), "white", (30, 22))
    src = block([(ep.get("source", "Font: idealista"), 28, (90, 96, 114), "Medium")], 940)
    src_y = 1560 if kind != "photo" else min(1600, stage.box_y + stage.box_h + 50)
    logo = None
    lp = ep.get("logo", "")
    if lp and os.path.exists(lp):
        logo = Image.open(lp).convert("RGBA")
        lw = 470; logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
    logo_t0 = [s for s in T if s["type"] == "outro"][0]["t0"]

    if kind == "cutout":
        gx = (W - (stage.group_x1 - stage.group_x0)) / 2 - stage.group_x0
        gy = floor_y - stage.feet_y
        shadows = {}
        for name in ("robot", "pep"):
            L = stage.layers[name]; w_ = L["bgra"].shape[1]
            sh = Image.new("RGBA", (int(w_ * 1.1), 120), (0, 0, 0, 0))
            ImageDraw.Draw(sh).ellipse([w_ * 0.08, 40, w_ * 1.02, 80], fill=(30, 30, 45, 70))
            shadows[name] = sh.filter(ImageFilter.GaussianBlur(14))
        pep_layer = bgra_to_pil(stage.layers["pep"]["bgra"])
        ed = style["enter_dur"]
        first, second = ("robot", "pep") if style["robot_first"] else ("pep", "robot")
        start = {first: 0.15, second: 0.15 + style["stagger"]}
    zoom_in = style["cam"] != "out"

    tmp = out_path + ".wav"; music(tmp, dur, style)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                             "-i", "-", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                             "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    sentiment = {i: st.get("sentiment", "neutral") for i, st in enumerate(ep["stats"])}
    for f in range(nfr):
        t = f / FPS
        scene = bg.copy()
        expr, talk, look = "smile", 0.0, 0.0
        for s in T:
            if s["type"] == "stat" and s["t0"] <= t < s["t1"] and sentiment.get(s["i"]) == "negative": expr = "worried"
        for ln in lines:
            if ln["who"] == "robot":
                e = env(t, ln["t0"], ln["t1"], 0.15)
                if e > 0: talk = e * (0.3 + 0.7 * abs(math.sin(t * 15)) * abs(math.sin(t * 5.7 + 1)))
            else:
                look = max(look, env(t, ln["t0"] - 0.2, ln["t1"], 0.45))
        if style["extra_look"]: look = max(look, env(t, 17.2, 20.0, 0.5))
        blink = (t % 3.4) < 0.13 and t > 0.8
        power = ease(t / 0.6)

        if kind == "photo":
            a_in = ease(t / 0.7)
            photo, tr = stage.frame(t, dur, zoom_in, expr, blink or power < 0.5, talk, look)
            yoff = (1 - a_in) * 40
            scene.alpha_composite(fade(stage.shadow, a_in), (stage.box_x - 60, int(stage.box_y - 60 + yoff)))
            scene.alpha_composite(fade(photo, a_in), (stage.box_x, int(stage.box_y + yoff)))
            def bubble_pos(who):
                return stage.to_frame(stage.cfg[who]["bubble"], tr)
        else:
            tilt = style["tilt_dir"] * math.radians(1.6) * math.sin(2 * math.pi * t / 6.5)
            robot = bgra_to_pil(stage.robot_frame(expr, blink, talk, look, tilt))
            for name, side in (("robot", -1), ("pep", 1)):
                p = ease_out((t - start[name]) / ed)
                dx = side * (1 - p) * 820
                bob = -abs(math.sin((t - start[name]) * math.pi * 3.0)) * 7 * (1 - p) if p < 1 else 0.0
                L = stage.layers[name]
                x = gx + L["x0"] + dx; y = gy + L["y0"] + bob
                if name == "pep":
                    br = 1 + 0.0035 * math.sin(2 * math.pi * t / 3.9 + 1.3)
                    lw_, lh_ = pep_layer.size
                    layer = pep_layer.resize((lw_, int(lh_ * br)), Image.BICUBIC)
                    y -= (layer.height - lh_)
                else:
                    layer = robot
                sh = shadows[name]
                scene.alpha_composite(sh, (int(x + layer.width / 2 - sh.width / 2), int(floor_y - 75)))
                scene.alpha_composite(layer, (int(x), int(y)))
            cam = style["cam"]; z = 1.0; px = 0.0
            if cam == "in": z = 1.0 + 0.05 * ease(t / dur)
            elif cam == "out": z = 1.05 - 0.05 * ease(t / dur)
            else: z = 1.025; px = 14 * math.sin(2 * math.pi * t / dur)
            if abs(z - 1) > 1e-4 or px:
                zw, zh = int(W * z), int(H * z)
                big = scene.resize((zw, zh), Image.BILINEAR)
                ox = int((zw - W) / 2 + px); oy = int((zh - H) * 0.72)
                scene = big.crop((ox, oy, ox + W, oy + H))
            def bubble_pos(who):
                b = stage.cfg[who]["bubble"]
                return gx + b[0] * stage.S, gy + b[1] * stage.S

        for s in T:
            if not (s["t0"] <= t < s["t1"]): continue
            a_in = ease((t - s["t0"]) / 0.5); a_out = 1 - ease((t - (s["t1"] - 0.35)) / 0.35) if t > s["t1"] - 0.35 else 1
            a = a_in * a_out; ci = style["card_in"]
            offx = -(1 - a_in) * 60 if ci == "left" else 0
            offy = (1 - a_in) * 30 if ci == "up" else 0
            sc_ = 0.94 + 0.06 * a_in if ci == "scale" else 1.0
            if s["type"] == "stat":
                st = ep["stats"][s["i"]]
                p = ease((t - s["t0"] - 0.25) / 1.3)
                v = block([(st.get("prefix", "") + fmt(st["value"] * p, st.get("decimals", 0)) + st.get("suffix", ""), 110, C["ink"], "Bold")], 900)
                parts = [s["label"], v] + ([s["sub"]] if s["sub"] is not None else [])
                aw = 120 if s["arrow"] is not None else 0
                cw = max(q.width for q in parts) + aw
                content = Image.new("RGBA", (cw, sum(q.height for q in parts) + 16), (0, 0, 0, 0)); y = 0
                for q in parts:
                    off = aw // 2 if q is v else 0
                    content.alpha_composite(q, (int((cw - q.width) / 2 + off), y))
                    if q is v and aw: content.alpha_composite(s["arrow"], (int((cw - q.width) / 2 + off - aw), y + int(q.height / 2 - 43)))
                    y += q.height + 8
                img = panel(content, style["card"])
            else:
                img = s["img"]
            if sc_ != 1.0: img = img.resize((int(img.width * sc_), int(img.height * sc_)), Image.BICUBIC)
            img = fade(img, a)
            scene.alpha_composite(img, (int(W / 2 - img.width / 2 + offx), int(card_y - img.height / 2 + offy)))
        for ln in lines:
            if not (ln["t0"] <= t < ln["t1"]): continue
            a = ease((t - ln["t0"]) / 0.4) * (1 - ease((t - (ln["t1"] - 0.3)) / 0.3) if t > ln["t1"] - 0.3 else 1)
            img = fade(ln["img"], a)
            bx, by = bubble_pos(ln["who"])
            bx = min(max(bx, img.width / 2 - 20), W - img.width / 2 + 20)
            scene.alpha_composite(img, (int(bx - img.width / 2), int(by - img.height / 2 + (1 - a) * 12)))
        if logo is not None and t >= logo_t0:
            a = ease((t - logo_t0) / 0.6)
            scene.alpha_composite(fade(logo, a), (int(W / 2 - logo.width / 2), int(src_y - 15 + (1 - a) * 18)))
        elif ep.get("show_source", False):
            scene.alpha_composite(src, (int(W / 2 - src.width / 2), src_y))
        proc.stdin.write(scene.convert("RGB").tobytes())
    proc.stdin.close(); proc.wait(); os.remove(tmp)
    if proc.returncode != 0: raise SystemExit("ffmpeg ha fallat")
    return style

if __name__ == "__main__":
    args = sys.argv[1:]
    seed = None
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    scene = None
    if "--scene" in args:
        i = args.index("--scene"); scene = args[i + 1]; del args[i:i + 2]
    ep = json.load(open(args[0], encoding="utf-8"))
    if scene: ep["scene"] = scene
    render(ep, args[1], seed=seed)
    print("OK", args[1])
