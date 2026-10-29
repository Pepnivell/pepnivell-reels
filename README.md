# pepnivell-reels

Reels automàtics de mercat immobiliari per a @pepnivell (Instagram), amb Pep i el robot de nivell.

## Com es fa un vídeo
```
python3 engine/nivellreel.py episodis/<id>.json videos/<id>.mp4 engine/rig2.json
```
- `engine/nivellreel.py` — motor: retalla Pep i el robot de `hosts2.png`, els fa entrar cadascun pel seu costat,
  anima el robot (ulls, mirada, "parla"), mostra 2 dades + frase de resum + tancament, i hi posa música.
  L'estil (fons, targetes, càmera, música, ordre d'entrada) canvia a cada vídeo segons l'`id`.
- `engine/rig2.json` — punts de control de la imatge.
- `episodis/<id>.json` — guió: `title`, `kicker`, `stats` (màx. 2), `pep_line`, `robot_line`, `takeaway`, `outro`, `source`.
- `episodis/<id>-text.txt` — text del post d'Instagram.
- `videos/` — vídeos finals. Metricool els agafa de `https://raw.githubusercontent.com/pepnivell/pepnivell-reels/main/videos/<id>.mp4`.

To: seriós. El cap de Pep no es deforma. Font de les dades: idealista.
