#!/usr/bin/env python3
"""Motor v3 de reels @pepnivell: motion graphics dinàmics (HTML animat filmat amb Playwright).
Ús: python3 engine/v3/reel.py episodis/<id>.json videos/<id>.mp4 [--frames 2,6,11,18 --outdir DIR]
"""
import json, os, subprocess, sys, wave
import numpy as np
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
W, H, FPS, DUR = 1080, 1920, 30, 20.0
BPM = 120
CUTS = [0, 2.5, 7.5, 12.5, 16.5, DUR]   # talls sobre el ritme
SR = 44100

def build_html(ep):
    timing = [[CUTS[i], CUTS[i + 1]] for i in range(5)]
    data = dict(ep["v3"], timing=timing, kicker=ep["kicker"], source=ep["source"], outro=ep["outro"])
    html = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    bg = {"botiga": "botiga.jpg", "estand": "estand.png"}.get(ep.get("scene"), "estand.png")
    html = html.replace("BGIMG", "file://" + os.path.join(ENGINE, bg))
    html = html.replace("PEPIMG", "file://" + os.path.join(HERE, "pep_cut.png"))
    html = html.replace("EPJSON", json.dumps(data, ensure_ascii=False))
    return html

def music(path):
    n = int(DUR * SR); t = np.arange(n) / SR; out = np.zeros(n)
    beat = 60 / BPM
    def add(sig, t0):
        i = int(t0 * SR); j = min(n, i + len(sig))
        if i < n: out[i:j] += sig[: j - i]
    rng = np.random.default_rng(7)
    # kick
    kl = int(0.35 * SR); kt = np.arange(kl) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-kt * 30)) / SR) * np.exp(-kt * 9)
    # hat
    hl = int(0.05 * SR); hat = rng.standard_normal(hl); hat = np.diff(hat, prepend=0) * np.exp(-np.arange(hl) / SR * 80)
    roots = [55.0, 43.65, 65.41, 49.0]   # A F C G
    for b in range(int(DUR / beat)):
        tb = b * beat
        if tb < 0.0: continue
        add(kick * 0.9, tb)
        add(hat * 0.18, tb + beat / 2)
        r = roots[(b // 4) % 4]
        for e in range(2):  # baix en corxeres
            bl = int(beat / 2 * SR); bt = np.arange(bl) / SR
            s = (np.sign(np.sin(2 * np.pi * r * bt)) * 0.3 + np.sin(2 * np.pi * r * 2 * bt)) * np.exp(-bt * 6) * 0.18
            add(s, tb + e * beat / 2)
        if b % 2 == 0:  # pad/pluck
            pl = int(beat * SR); pt = np.arange(pl) / SR
            chord = sum(np.sin(2 * np.pi * r * 4 * m * pt) for m in (1, 1.26, 1.5))
            add(chord * np.exp(-pt * 4) * 0.05, tb)
    # whoosh + impacte a cada tall
    for c in CUTS[1:-1]:
        wl = int(0.45 * SR); wt = np.arange(wl) / SR
        noise = rng.standard_normal(wl)
        k = np.linspace(0.02, 0.5, wl); f = np.zeros(wl); acc = 0
        for i in range(wl): acc += k[i] * (noise[i] - acc); f[i] = acc
        add(f * (wt / wt[-1]) ** 2 * 0.5, c - 0.45)
        il = int(0.8 * SR); it = np.arange(il) / SR
        add(np.sin(2 * np.pi * 38 * it) * np.exp(-it * 5) * 0.6, c)
    out[-int(0.6 * SR):] *= np.linspace(1, 0, int(0.6 * SR))
    out = np.tanh(out * 1.3); out /= np.max(np.abs(out)) * 1.12
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((out * 32767).astype(np.int16).tobytes())

def render(ep_path, out_path, stills=None, outdir=None):
    ep = json.load(open(ep_path, encoding="utf-8"))
    html_path = out_path + ".html"
    open(html_path, "w", encoding="utf-8").write(build_html(ep))
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto("file://" + html_path); pg.evaluate("document.fonts.ready")
        if stills:
            for s in stills:
                pg.evaluate(f"render({s})")
                pg.screenshot(path=os.path.join(outdir, f"still_{s}.jpg"), type="jpeg", quality=90)
            b.close(); os.remove(html_path); return
        wav = out_path + ".wav"; music(wav)
        tmp = out_path + ".tmp.mp4"
        ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS),
                               "-c:v", "mjpeg", "-i", "-", "-i", wav, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", tmp],
                              stdin=subprocess.PIPE)
        for i in range(int(DUR * FPS)):
            pg.evaluate(f"render({i / FPS})")
            ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
        ff.stdin.close(); rc = ff.wait(); b.close()
        os.remove(wav); os.remove(html_path)
        if rc != 0: sys.exit("ffmpeg ha fallat")
        os.replace(tmp, out_path)

if __name__ == "__main__":
    a = sys.argv[1:]
    stills = outdir = None
    if "--frames" in a:
        stills = [float(x) for x in a[a.index("--frames") + 1].split(",")]; outdir = a[a.index("--outdir") + 1]
    render(a[0], a[1], stills, outdir)
