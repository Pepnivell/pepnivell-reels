#!/usr/bin/env python3
"""
realreel.py — Reels amb la imatge real de Pep i el robot de nivell, animada.

Ús:  python3 realreel.py episodi.json sortida.mp4 [rig.json]

No es renderitza res en 3D: es treballa sobre la foto original i se li dona vida
amb deformacions suaus (respiració, cap que assenteix/nega/s'inclina, polzes que
fan "pump", mà a l'espatlla), ulls de llum del robot redibuixats i animats
(parpelleig, expressions, "boca" quan parla), zoom de càmera lent, targetes de
dades, bafarades i música.
"""
import json, math, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clayreel import (W, H, FPS, PAL, text_block, card, arrow, scaled, put, fmt_num, make_audio,
                      ease_out_back, ease_in_out, clay)

def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def env(t, t0, t1, ramp=0.35):
    """Envolupant suau 0→1→0 dins [t0,t1]."""
    if t < t0 or t > t1: return 0.0
    a = min(1.0, (t - t0) / ramp); b = min(1.0, (t1 - t) / ramp)
    a = a * a * (3 - 2 * a); b = b * b * (3 - 2 * b)
    return min(a, b)

class Rig:
    def __init__(self, rig_path):
        self.cfg = json.load(open(rig_path))
        base = os.path.dirname(os.path.abspath(rig_path))
        img = cv2.imread(os.path.join(base, self.cfg["image"]), cv2.IMREAD_COLOR)
        img = self._remove_eyes(img)
        x0, y0, x1, y1 = self.cfg["crop"]
        self.S = S = self.cfg["scale"]
        crop = img[y0:y1, x0:x1]
        self.base = cv2.resize(crop, (int((x1 - x0) * S), int((y1 - y0) * S)), interpolation=cv2.INTER_AREA)
        self.h, self.w = self.base.shape[:2]
        self.ox = (W - self.w) // 2
        self.oy = self.cfg["place_y"]
        self.bg_color = tuple(int(v) for v in img[5, 5][::-1])
        self.P = lambda p: np.array([(p[0] - x0) * S, (p[1] - y0) * S], np.float32)
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.X, self.Y = xx, yy
        sx0, sx1 = [(v - x0) * S for v in self.cfg["split_x"]]
        robot_side = 1 - smooth(sx0, sx1, xx)
        self.feet = (self.cfg["feet_y"] - y0) * S
        vert = smooth(self.feet + 5, self.feet - 60, yy)
        self.body = {"robot": robot_side * vert, "pep": (1 - robot_side) * vert}
        self.masks = {}
        for who in ("robot", "pep"):
            for part in ("head", "thumb", "pat"):
                if part in self.cfg[who]:
                    r = self.cfg[who][part]
                    c = self.P(r["c"]); rad = r["r"] * S
                    d = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
                    self.masks[(who, part)] = (1 - smooth(rad, rad * 1.6, d)).astype(np.float32)

    def _remove_eyes(self, img):
        """Esborra els ulls de llum originals del visor (inpainting) per redibuixar-los animats."""
        e = self.cfg["robot"]["eyes"]
        x0, y0, x1, y1 = e["visor_box"]
        roi = img.astype(np.int32)
        b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
        glow = ((b > 55) & (b > r + 22)).astype(np.uint8)
        keep = np.zeros_like(glow)
        for c in (e["L"], e["R"]):
            cv2.ellipse(keep, (int(c[0]), int(c[1])), (34, 24), 18, 0, 360, 1, -1)
        mask = (glow & keep) * 255
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        return cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)

    def eyes_layer(self, img, expr, blink, talk, bounce, look=0.0):
        """Dibuixa els ulls del robot (i la 'boca' quan parla) amb resplendor blava."""
        e = self.cfg["robot"]["eyes"]
        S = self.S
        L, R = self.P(e["L"]), self.P(e["R"])
        L = L + np.array([10 * S * look, 3 * S * look], np.float32); R = R + np.array([10 * S * look, 3 * S * look], np.float32)
        ang = math.atan2(R[1] - L[1], R[0] - L[0])
        ca, sa = math.cos(ang), math.sin(ang)
        mid = (L + R) / 2
        k = 4  # supermostreig
        pad = int(70 * S)
        bx0, by0 = int(mid[0] - pad * 1.4), int(mid[1] - pad)
        bw, bh = int(pad * 2.8), int(pad * 2.2)
        layer = np.zeros((bh * k, bw * k), np.float32)
        def pt(lx, ly, base):
            x = base[0] + (lx * ca - ly * sa); y = base[1] + (lx * sa + ly * ca) - bounce
            return (int((x - bx0) * k), int((y - by0) * k))
        ew, th = 17 * S, max(2, int(5.5 * S * k))
        for side, cen in ((-1, L), (1, R)):
            pts = []
            if blink:
                pts = [pt(-ew, 1, cen), pt(ew, 1, cen)]
            elif expr == "worried":
                lift = 7 * S * (-side)
                pts = [pt(-ew, -lift, cen), pt(ew, lift, cen)] if side < 0 else [pt(-ew, lift, cen), pt(ew, -lift, cen)]
            elif expr == "surprised":
                c = pt(0, 0, cen); cv2.circle(layer, c, int(9 * S * k), 1.0, th, cv2.LINE_AA); continue
            else:
                hgt = (13 if expr == "happy" else 10) * S
                pts = [pt(ew * math.cos(a), -hgt * math.sin(a) + 4 * S, cen) for a in np.linspace(math.pi, 0, 24)]
            cv2.polylines(layer, [np.array(pts, np.int32)], False, 1.0, th, cv2.LINE_AA)
        if talk > 0:
            mw = (8 + 14 * talk) * S
            mc = mid + np.array([-sa, ca]) * 38 * S
            cv2.line(layer, pt(-mw, 0, mc), pt(mw, 0, mc), 1.0, max(2, int(4.5 * S * k)), cv2.LINE_AA)
        core = cv2.resize(layer, (bw, bh), interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(core, (0, 0), 6 * S) * 1.6 + cv2.GaussianBlur(core, (0, 0), 2.2 * S) * 0.9
        sub = img[by0:by0 + bh, bx0:bx0 + bw].astype(np.float32)
        blue = np.array([255, 190, 70], np.float32)       # BGR
        white = np.array([255, 245, 225], np.float32)
        sub = sub + glow[..., None] * blue * 0.55
        sub = sub * (1 - core[..., None]) + core[..., None] * (white * 0.6 + blue * 0.4)
        img[by0:by0 + bh, bx0:bx0 + bw] = np.clip(sub, 0, 255).astype(np.uint8)
        return img

    def warp(self, img, moves):
        """moves: llista de (mask, pivot(px), angle(rad), tx, ty). Mapa invers aproximat per a moviments petits."""
        dx = np.zeros((self.h, self.w), np.float32); dy = np.zeros_like(dx)
        for m, piv, a, tx, ty in moves:
            if abs(a) < 1e-5 and abs(tx) < 1e-3 and abs(ty) < 1e-3: continue
            ca, sa = math.cos(a), math.sin(a)
            rx, ry = self.X - piv[0], self.Y - piv[1]
            fx = ca * rx - sa * ry + piv[0] + tx - self.X
            fy = sa * rx + ca * ry + piv[1] + ty - self.Y
            dx += m * fx; dy += m * fy
        return cv2.remap(img, self.X - dx, self.Y - dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def make_audio_calm(path, dur):
    """Fons musical sobri: acords de piano suaus amb coixí (pad), sense percussió."""
    import wave
    sr = 44100; n = int(sr * dur); out = np.zeros(n, np.float32)
    rng = np.random.default_rng(3)
    def note(freq, t0, length, amp, kind="piano"):
        i0 = int(t0 * sr); L = min(int(length * sr), n - i0)
        if L <= 0: return
        tt = np.arange(L) / sr
        if kind == "piano":
            env = np.exp(-tt * 1.6) * (1 - np.exp(-tt * 200))
            w = np.sin(2*np.pi*freq*tt) + 0.35*np.sin(4*np.pi*freq*tt) + 0.12*np.sin(6*np.pi*freq*tt)
        else:
            env = np.minimum(1, tt / 0.9) * np.minimum(1, (length - tt) / 0.9)
            w = np.sin(2*np.pi*freq*tt) + 0.5*np.sin(2*np.pi*freq*1.003*tt) + 0.25*np.sin(4*np.pi*freq*tt)
        out[i0:i0+L] += (w * env * amp).astype(np.float32)
    # Am - F - C - G
    prog = [[220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 261.6, 329.6], [196.0, 246.9, 293.7]]
    bars = 2.5; t = 0.0; k = 0
    while t < dur:
        ch = prog[k % 4]
        for f in ch: note(f, t, bars + 0.4, 0.05, "pad")
        note(ch[0] / 2, t, bars, 0.07, "pad")
        for j, f in enumerate([ch[0], ch[1], ch[2], ch[1] * 2]):
            note(f * 2, t + j * bars / 4, 2.2, 0.06)
        t += bars; k += 1
    fade = int(1.5 * sr); out[-fade:] *= np.linspace(1, 0, fade)
    out[:int(0.6 * sr)] *= np.linspace(0, 1, int(0.6 * sr))
    out = out / max(1e-6, np.abs(out).max()) * 0.6
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((out * 32767).astype(np.int16).tobytes())

def faded(im, a):
    if a >= 0.999: return im
    im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * a))); return im

def render(spec, out_path, rig_path):
    rig = Rig(rig_path)
    dur = float(spec.get("duration", 20)); nframes = int(dur * FPS)
    if spec.get("ink"): PAL["ink"] = tuple(spec["ink"])
    accent = tuple(spec.get("accent", (214, 36, 44)))
    muted = tuple(spec.get("muted", (96, 104, 124)))
    scenes = spec["scenes"]; actions = spec.get("actions", [])
    card_col = tuple(spec.get("card", (255, 255, 255)))
    bgc = rig.bg_color

    for sc in scenes:
        k = sc["type"]
        if k == "title":
            parts = ([(sc["kicker"], 40, accent, True)] if sc.get("kicker") else []) + [(sc["text"], 80, "ink", True)]
            sc["_img"] = card(text_block(parts, 860), card_col)
        elif k == "stat":
            sc["_label"] = text_block([(sc["label"], 42, muted, False)], 860)
            sc["_sub"] = text_block([(sc["sub"], 36, muted, False)], 860) if sc.get("sub") else None
            sc["_arrow"] = arrow(sc["arrow"], (226, 92, 84) if sc["arrow"] == "up" else (86, 170, 120)) if sc.get("arrow") else None
        elif k == "bubble":
            sc["_img"] = card(text_block([(sc["text"], 42, "ink", True)], 290), (255, 255, 255), (34, 24))
        elif k == "outro":
            sc["_img"] = card(text_block([(sc.get("text", ""), 48, "ink", True), (sc.get("handle", "@pepnivell"), 68, accent, True)], 860), card_col)
    src_img = text_block([(spec.get("source", "Font: idealista"), 30, muted, False)], 940)

    def mood_at(t):
        m = "idle"
        for sc in scenes:
            if sc["t0"] <= t < sc["t1"] and sc.get("moods"):
                m = sc["moods"] if isinstance(sc["moods"], str) else sc["moods"][0]
        return m

    pops = [sc["t0"] + 0.05 for sc in scenes if sc["type"] != "bubble"]
    tmp_audio = out_path + ".wav"
    serious = spec.get("tone", "serious") == "serious"
    if serious: make_audio_calm(tmp_audio, dur)
    else: make_audio(tmp_audio, dur, pops)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-i", tmp_audio, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", out_path],
                            stdin=subprocess.PIPE)
    bg_frame = Image.new("RGBA", (W, H), bgc + (255,))
    cfg = rig.cfg
    heads = {w: (rig.masks[(w, "head")], rig.P(cfg[w]["head"]["pivot"])) for w in ("robot", "pep")}
    thumbs = {w: (rig.masks[(w, "thumb")], rig.P(cfg[w]["thumb"]["pivot"])) for w in ("robot", "pep")}
    S = rig.S
    for f in range(nframes):
        t = f / FPS
        mood = mood_at(t)
        # ---- ulls del robot
        blink = (t % 3.3) < 0.13 and t > 0.6
        expr = {"worried": "worried", "happy": "happy"}.get(mood, "smile")
        talk = 0.0; bounce = 0.0; look = 0.0
        for a in actions:
            e = env(t, a["t0"], a["t1"], 0.15)
            if a["who"] == "robot" and a["act"] == "talk" and e > 0:
                talk = e * (0.35 + 0.65 * abs(math.sin(t * 17)) * abs(math.sin(t * 6.3 + 1)))
            if a["who"] == "robot" and a["act"] == "look_at": look = max(look, env(t, a["t0"], a["t1"], 0.45))
            if a["who"] == "robot" and a["act"] == "eyes" and e > 0.5:
                expr = a.get("expr", expr)
        if mood == "happy" and not serious: bounce = abs(math.sin(t * 7)) * 3 * S
        power = min(1.0, t / 0.5)
        img = rig.eyes_layer(rig.base.copy(), expr, blink or power < 0.5, talk, bounce, look)
        # ---- moviments
        moves = []
        for who, ph in (("robot", 0.0), ("pep", 1.3)):
            br = math.sin(2 * math.pi * t / 3.8 + ph)
            moves.append((rig.body[who], (rig.w / 2, rig.feet), 0.0, 0.0, 0.0))
            # respiració: estirament vertical suau respecte dels peus
            m, piv = heads[who]
            idle_a = math.radians(0.9) * math.sin(2 * math.pi * t / 5.1 + ph * 2)
            a = idle_a; tx = 0.0; ty = -1.2 * S * br
            for act in actions:
                if act["who"] != who: continue
                e = env(t, act["t0"], act["t1"])
                if e <= 0: continue
                k = act["act"]
                if k == "nod": ty += e * act.get("amp", 1.0) * 6 * S * math.sin(2 * math.pi * act.get("hz", 1.8) * (t - act["t0"]))
                elif k == "shake":
                    tx += e * 5 * S * math.sin(2 * math.pi * 2.2 * (t - act["t0"]))
                    a += e * math.radians(2.2) * math.sin(2 * math.pi * 2.2 * (t - act["t0"]))
                elif k == "tilt": a += e * math.radians(act.get("deg", 4))
                elif k == "look_up": a += e * math.radians(act.get("deg", -3)); ty -= e * 4 * S
                elif k == "look_at": a += e * math.radians(act.get("deg", 4)); tx += e * 4 * S
                elif k == "talk" and who == "pep": ty += e * 2.2 * S * abs(math.sin(2 * math.pi * 2.6 * (t - act["t0"])))
            if who in spec.get("static_heads", []): a = tx = ty = 0.0
            moves.append((m, piv, a, tx, ty))
            # cos: respiració
            bm = rig.body[who]
            bmask = bm * smooth(rig.feet, rig.feet * 0.35, rig.Y)
            if who in spec.get("static_heads", []): bmask = bmask * (1 - m)
            moves.append((bmask, (rig.w / 2, rig.feet), 0.0, 0.0, -2.0 * S * br))
            # polze
            tm, tp = thumbs[who]
            ta, tty = 0.0, 0.0
            for act in actions:
                if act["who"] in (who, "both") and act["act"] == "thumbs":
                    e = env(t, act["t0"], act["t1"])
                    tty -= e * 10 * S * abs(math.sin(2 * math.pi * 1.7 * (t - act["t0"]) + ph))
                    ta += e * math.radians(3) * math.sin(2 * math.pi * 1.7 * (t - act["t0"]) + ph) * (-1 if who == "robot" else 1)
            moves.append((tm, tp, ta, 0.0, tty))
            # tots dos: bot suau
            for act in actions:
                if act["who"] in (who, "both") and act["act"] == "bounce":
                    e = env(t, act["t0"], act["t1"])
                    moves.append((rig.body[who] * smooth(rig.feet, rig.feet - 120 * S, rig.Y), (0, 0), 0.0, 0.0,
                                  -e * 7 * S * abs(math.sin(2 * math.pi * 1.6 * (t - act["t0"]) + ph))))
        for act in actions:
            if act["act"] == "pat":
                e = env(t, act["t0"], act["t1"])
                moves.append((rig.masks[("robot", "pat")], (0, 0), 0.0, 0.0, e * 5 * S * math.sin(2 * math.pi * 3 * (t - act["t0"]))))
        img = rig.warp(img, moves)
        # ---- càmera: zoom lent amb petites empentes als canvis d'escena
        z = 1.0 + 0.045 * ease_in_out(t / dur)
        for sc in scenes:
            if sc["type"] == "stat": z += 0.012 * math.exp(-((t - sc["t0"] - 0.35) ** 2) / 0.08)
        frame = bg_frame.copy()
        person = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if abs(z - 1) > 1e-4:
            pw, ph_ = int(rig.w * z), int(rig.h * z)
            person = person.resize((pw, ph_), Image.BICUBIC)
            cx, cy = rig.ox + rig.w / 2, rig.oy + rig.h * 0.35
            frame.paste(person, (int(cx - pw / 2), int(cy - rig.h * 0.35 * z)))
        else:
            frame.paste(person, (rig.ox, rig.oy))
        # suavitza les vores del retall amb el fons
        # ---- targetes
        for sc in scenes:
            t0, t1 = sc["t0"], sc["t1"]
            if not (t0 <= t < t1): continue
            if serious:
                a_in = ease_in_out((t - t0) / 0.45); a_out = (1 - ease_in_out((t - (t1 - 0.3)) / 0.3)) if t > t1 - 0.3 else 1
                alpha = a_in * a_out; s = 1.0; slide = (1 - a_in) * 26
            else:
                s = ease_out_back((t - t0) / 0.38) * ((1 - ease_in_out((t - (t1 - 0.25)) / 0.25)) if t > t1 - 0.25 else 1)
                alpha = 1.0; slide = 0.0
            k = sc["type"]
            if k in ("title", "outro"):
                im = scaled(sc["_img"], s)
                if im: put(frame, faded(im, alpha), W / 2, sc.get("y", 310) + slide)
            elif k == "bubble":
                im = scaled(sc["_img"], s)
                if im:
                    who = sc.get("who", "robot" if sc.get("speaker", 0) == 0 else "pep")
                    hc = rig.P(cfg[who]["head"]["c"])
                    hx, hy = rig.ox + hc[0], rig.oy + hc[1]
                    side = -1 if who == "robot" else 1
                    bx = hx + side * (im.width / 2 + 75 * S)
                    bx = min(max(bx, im.width / 2 - 20), W - im.width / 2 + 20)
                    by = sc.get("y", hy - 100 * S)
                    d = ImageDraw.Draw(frame)
                    put(frame, faded(im, alpha), bx, by + slide)
            elif k == "stat":
                p = ease_in_out((t - t0 - 0.2) / sc.get("count", 1.2))
                vtxt = sc.get("prefix", "") + fmt_num(sc["value"] * p, sc.get("decimals", 0)) + sc.get("suffix", "")
                vimg = text_block([(vtxt, 112, "ink", True)], 900)
                parts = [sc["_label"], vimg] + ([sc["_sub"]] if sc["_sub"] is not None else [])
                aw = 150 if sc["_arrow"] is not None else 0
                cw = max(pp.width for pp in parts) + aw
                content = Image.new("RGBA", (cw, sum(pp.height for pp in parts) + 20), (0, 0, 0, 0))
                y = 0
                for pp in parts:
                    off = aw // 2 if pp is vimg else 0
                    content.alpha_composite(pp, (int((cw - pp.width) / 2 + off), y))
                    if pp is vimg and aw:
                        content.alpha_composite(sc["_arrow"], (max(0, int((cw - pp.width) / 2 + off - aw)), y + 18 + (0 if serious else int(math.sin(t * 6) * 6))))
                    y += pp.height + 10
                im = scaled(card(content, card_col), s)
                if im: put(frame, faded(im, alpha), W / 2, sc.get("y", 310) + slide)
        put(frame, src_img, W / 2, 1585)
        proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close(); proc.wait(); os.remove(tmp_audio)
    if proc.returncode != 0: raise SystemExit("ffmpeg ha fallat")

if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    rig_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "rig.json")
    render(spec, sys.argv[2], rig_path)
    print("OK", sys.argv[2])
