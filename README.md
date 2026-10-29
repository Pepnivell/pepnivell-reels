# pepnivell-reels

Reels automàtics de mercat immobiliari per a @pepnivell (Instagram), amb Pep i el robot de nivell.

## Com es fa un vídeo
```
python3 engine/nivellreel.py episodis/<id>.json videos/<id>.mp4
```
(cal numpy, opencv, pillow i ffmpeg)

## Escenaris (camp `"scene"` de l'episodi)
| scene     | imatge               | com es veu |
|-----------|----------------------|------------|
| `cartoon` | engine/hosts2.png    | Pep i el robot il·lustrats; entren cadascun pel seu costat i es queden al centre |
| `real`    | engine/full.png      | foto real de cos sencer amb els braços plegats; entren pels costats |
| `estand`  | engine/estand.png    | selfie real a l'estand de fusta de nivell; foto amb zoom lent |
| `botiga`  | engine/botiga.jpg    | selfie real davant la botiga de nivell; foto amb zoom lent |

**Regla de rotació:** tria un escenari diferent del de l'últim episodi publicat (mira el camp `scene` dels JSON a `episodis/`).
Si no s'indica `scene`, se'n tria un a l'atzar. A tots els escenaris el robot està animat (ulls de llum, mirada, "parla");
el cap de Pep no es deforma mai.

L'estil (fons, targetes, càmera, música, ordre d'entrada) també canvia a cada vídeo segons l'`id`.

## Format de l'episodi (`episodis/<id>.json`)
`id`, `scene`, `source`, `kicker`, `title`, `stats` (exactament 2), `pep_line`, `robot_line`, `takeaway`, `outro`,
`"logo": ""`, `"show_source": true`. Mira l'últim episodi com a model.

- `episodis/<id>-text.txt` — text del post d'Instagram.
- `videos/` — vídeos finals. Metricool els agafa de `https://raw.githubusercontent.com/pepnivell/pepnivell-reels/main/videos/<id>.mp4`.

To: seriós. Font de les dades: idealista.
