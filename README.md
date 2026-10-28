# pepnivell-reels

Reels automàtics de mercat immobiliari per a @pepnivell (Instagram).

- `engine/realreel.py` — anima la imatge real (Pep + robot de nivell) i hi afegeix dades, bafarades i música.
  Ús: `python3 engine/realreel.py episodis/<episodi>.json videos/<episodi>.mp4 engine/rig.json`
- `engine/rig.json` — punts de control de la imatge (caps, polzes, ulls del robot).
- `episodis/` — guió i dades de cada vídeo (font: idealista) i el text del post.
- `videos/` — vídeos finals; Metricool els agafa des de l'enllaç "raw" de GitHub.
