# Référence et validité des modes Outcoloring

Ce document résume les recherches effectuées sur les formules d’outcoloring pour fractales (Mandelbrot, Julia, etc.) et vérifie la cohérence des modes proposés dans le menu.

## Sources utilisées

- **Wikipedia** : [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set), [Plotting algorithms for the Mandelbrot set](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set)
- **Linas Vepstas** : [Renormalizing the Mandelbrot Escape](https://linas.org/art-gallery/escape/escape.html) (formule de lissage continu)
- **Fractint / XaoS** : noms et familles de modes (Iter+Real, Biomorphs, etc.)

---

## Modes implémentés et validité

### 1. **Iter** (itération discrète)
- **Formule** : valeur = `iteration / iter_max`.
- **Référence** : algorithme d’échappement classique (escape time). Couleur = palette[iteration].
- **Validité** : ✅ Standard, documenté partout.

---

### 2. **Smooth** (coloration continue / normalised iteration count)
- **Formule** (code) :  
  `nu = log(log(|z|)/log(2)) / log(2)`  
  `smooth = (iteration + 1 - nu) / iter_max`
- **Référence** :  
  - Linas Vepstas : *μ = n + 1 − log(log|Z(n)|) / log(2)* (log naturel).  
  - Wikipedia (Continuous coloring) : *ν(z) = n − log_P(log|z_n|/log(N))* avec P=2.
- **Note** : Notre terme en double log utilise `ln(ln|z|/ln(2))/ln(2)`, ce qui revient à la même famille (continuité du gradient, décalage de normalisation possible). Comportement visuel cohérent avec la littérature.
- **Validité** : ✅ Formule standard de lissage continu.

---

### 3. **Iter+Real**, **Iter+Imag**, **Iter+Real/Imag**, **Iter+All**
- **Formule** : base = iteration/max, puis ajout d’une contribution issue de Re(z), Im(z) ou atan2(Re,Im), normalisée (ex. /10, *0.2).
- **Référence** : variantes courantes (type XaoS/Fractint) pour casser les bandes et ajouter de la variation à partir de la valeur finale de z.
- **Validité** : ✅ Variantes connues, usage artistique répandu.

---

### 4. **Binary Decomposition**
- **Formule** : même base que Smooth (ou itération), puis **inversion RGB** lorsque Im(z) &lt; 0 à l’échappement.
- **Référence** : la “binary decomposition” de l’extérieur du Mandelbrot consiste à séparer les points selon un critère binaire (souvent le signe de Im(z) à l’échappement), donnant des régions alternées (arbres de Hubbard, rayons externes).
- **Validité** : ✅ Correspondance avec la pratique classique (signe de Im(z) + inversion de couleur).

---

### 5. **Biomorphs**
- **Formule** (code) : si |Re(z)| &lt; 10 et |Im(z)| &lt; 10 → itération discrète ; sinon → smooth.
- **Référence** : *Pickover biomorphs* : on détecte une “frontière” quand |Re(z)| ou |Im(z)| dépasse un seuil (souvent 10), ce qui donne des formes organiques.
- **Validité** : ✅ Comportement typique des biomorphs (seuil sur Re/Im).

---

### 6. **Potential**
- **Formule** (code) : `potential = ln(|z|) / 2^iteration`, puis normalisation (ex. * 1000, clamp [0,1]).
- **Référence** : Wikipedia (Continuous coloring) donne le potentiel de Douady–Hubbard :  
  *φ(z) = lim log|z_n| / P^n* (P=2 pour Mandelbrot).  
  Donc log|z|/2^n est bien une forme de potentiel renormalisé.
- **Validité** : ✅ Cohérent avec la notion de potentiel (extérieur du Mandelbrot).

---

### 7. **Color Decomposition**
- **Formule** (code) : base = smooth, puis mélange avec l’angle (arg(z)) normalisé (ex. 70 % smooth, 30 % angle).
- **Référence** : “Color decomposition” ou “argument-based coloring” : utilisation de l’argument de z pour moduler la teinte (décomposition en angle + module/iteration).
- **Validité** : ✅ Variante courante (angle pour la couleur).

---

### 8. **Orbit Traps**
- **Formule** (code) : valeur basée sur la distance minimale à un piège d’orbite (point, ligne, cercle, etc.), mélangée avec l’itération.
- **Référence** : Wikipedia cite *Orbit trap* comme technique de rendu ; Fractint et autres utilisent des traps (point, ligne, cercle) pour colorier selon la proximité de l’orbite au piège.
- **Validité** : ✅ Correspond à la définition usuelle des orbit traps.

---

### 9. **Wings**
- **Formule** (code) : moyenne de |sinh(z)| sur les points de l’orbite, normalisée, puis mélange avec l’itération.
- **Référence** : “Wings” fait plutôt référence aux structures en “ailes” (wakes) près du Mandelbrot (bulbes, antennes). Ici c’est une coloration **artistique** basée sur sinh(orbite), pas une formule standard unique.
- **Validité** : ⚠️ Acceptable comme mode créatif ; pas une formule de référence unique dans la littérature.

---

### 10. **Distance**
- **Formule** (code) : si distance estimée disponible : normalisation en log (ex. -ln(d)/10), mélange avec itération.
- **Référence** : Wikipedia (Plotting algorithms) : *Distance estimation* (exterior/interior) ; coloration par distance à la frontière du Mandelbrot/Julia est courante.
- **Validité** : ✅ Utilisation standard de la distance estimée pour la couleur.

---

### 11. **Distance AO** (Ambient Occlusion)
- **Formule** (code) : facteur du type `1 - exp(-k/distance)`, puis mélange avec itération.
- **Référence** : AO en rendu 3D/fractal : assombrir les zones proches de la frontière (petite distance) pour donner du relief. Formule typique d’occlusion.
- **Validité** : ✅ Formule d’AO classique appliquée à la distance estimée.

---

### 12. **Distance 3D**
- **Formule** (code) : combinaison distance + angle de z pour simuler un éclairage (ex. lumière en haut-gauche), puis mélange avec itération.
- **Référence** : rendu “3D” ou “shaded” par distance + direction (normal/angle) pour effet de relief.
- **Validité** : ✅ Approche courante pour un rendu type relief 3D.

---

## Résumé

| Mode              | Validité | Remarque                                      |
|-------------------|----------|-----------------------------------------------|
| Iter              | ✅       | Standard escape time                          |
| Smooth            | ✅       | Formule standard de lissage continu           |
| Iter+Real/Imag/…   | ✅       | Variantes XaoS/Fractint                       |
| Binary Decomp     | ✅       | Signe Im(z) + inversion couleur               |
| Biomorphs         | ✅       | Seuils Pickover Re/Im                         |
| Potential         | ✅       | Potentiel type Douady–Hubbard                 |
| Color Decomp      | ✅       | Angle + smooth                                |
| Orbit Traps       | ✅       | Distance aux traps                            |
| Wings             | ⚠️       | Créatif (sinh orbite), pas une référence unique |
| Distance          | ✅       | Distance estimation                           |
| Distance AO       | ✅       | AO sur distance                               |
| Distance 3D       | ✅       | Shading par distance + angle                  |

**Conclusion** : Les modes du menu correspondent à des méthodes documentées (Wikipedia, Linas, Fractint/XaoS) ou à des extensions courantes (distance, AO, 3D). Le seul mode “artiste” sans formule de référence unique est **Wings** ; il reste cohérent comme option de coloration basée sur l’orbite.
