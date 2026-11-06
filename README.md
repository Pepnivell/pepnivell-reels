# pepnivell-reels

Reels automàtics de mercat immobiliari per a @pepnivell (Instagram) i @pep.nivell (TikTok).

## Motor actual: v3 (motion graphics dinàmics)
```
python3 engine/v3/reel.py episodis/<id>.json videos/<id>.mp4
```
Comprovar fotogrames sense renderitzar el vídeo (molt ràpid):
```
python3 engine/v3/reel.py episodis/<id>.json /tmp/x.mp4 --frames 2,6,11,18 --outdir <carpeta>
```
(cal playwright amb chromium, numpy, opencv i ffmpeg). Vídeo de 20 s, 1080x1920, 30 fps, amb música i efectes generats.

### Estructura del vídeo (talls al ritme, 120 BPM)
| temps | escena | camp |
|---|---|---|
| 0–2,5 s | **Ganxo**: pregunta o frase forta, paraula a paraula | `v3.hook` |
| 2,5–7,5 s | **Dada 1**: número gran que compta fins al valor + xip amb context | `v3.stat1` |
| 7,5–12,5 s | **Comparació**: 2 barres que creixen (mateixa unitat) | `v3.compare` |
| 12,5–16,5 s | **Pep** (foto real, engine/v3/pep_cut.png) amb la seva frase en subtítols | `v3.pep` |
| 16,5–20 s | **Conclusió** + botó "Segueix @pepnivell" + outro | `v3.take`, `v3.cta` |

### Format de l'episodi (`episodis/<id>.json`) — model: `engine/v3/exemple.json`
- `id` (= nom del fitxer), `"engine": "v3"`, `scene` (`estand` o `botiga`: foto desenfocada de fons del ganxo i la dada 1; alterna respecte de l'últim episodi),
  `source` ("Font: idealista · <informe>, <mes any>"), `kicker` (MAJÚSCULES automàtiques, ≤ 34 caràcters), `title` (≤ 35, per a TikTok), `outro`.
- `v3.hook`: `text` (≤ 40 caràcters) i `marks` (una entrada per paraula: `null`, `"hly"` = ressaltat groc, `"hl"` = ressaltat vermell). Ressalta com a molt 2–3 paraules seguides.
- `v3.stat1`: `label` (≤ 34), `value`, `decimals`, `prefix` opcional, `suffix` (p. ex. `" €/m²"`), `sub` (≤ 28).
- `v3.compare`: `label` (≤ 30) i `items` = EXACTAMENT 2 elements `{name (≤ 12), value, decimals, prefix, suffix, hl}`; `hl: true` = barra vermella (la protagonista).
  Han de ser dues dades verificades de la mateixa mètrica (p. ex. Catalunya vs Espanya, Barcelona vs Madrid, ara vs fa un any).
- `v3.pep`: `text` (≤ 38) i `marks`. `v3.take`: `text` (≤ 50) i `marks` (ressalta només 1–2 paraules finals). `v3.cta`: "Segueix @pepnivell".

## Motor antic (v2)
`python3 engine/nivellreel.py episodis/<id>.json videos/<id>.mp4` — escenaris `cartoon`, `real`, `estand`, `botiga`. Ja no s'utilitza.

## Altres fitxers
- `episodis/<id>-text.txt` — text del post d'Instagram; `episodis/<id>-tiktok.txt` — versió curta per a TikTok.
- `videos/` — vídeos finals. Metricool els agafa de `https://raw.githubusercontent.com/pepnivell/pepnivell-reels/main/videos/<id>.mp4`.

To: seriós i proper. Font de les dades: idealista. Mai no s'inventa cap xifra.
