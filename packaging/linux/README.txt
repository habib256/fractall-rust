fractall — explorateur de fractales (Linux AppImage, x86_64)
============================================================

Moteur de rendu de fractales à profondeur arbitraire : perturbation + BLA
(portage de Fraktaler-3), orbite de référence GMP/MPFR, moteur bytecode unifié
CPU/GPU, 33 types (Mandelbrot, Julia, Burning Ship, Tricorn, Celtic, Buffalo,
Multibrot, hybrides multi-phase, Buddhabrot/Nebulabrot, Lyapunov…), 27 palettes
et 15 modes de colorisation.


Lancer
------
  chmod +x fractall-*-x86_64.AppImage
  ./fractall-*-x86_64.AppImage

L'AppImage embarque les TROIS binaires de la distribution. Par défaut elle
ouvre la GUI ; les outils console s'appellent par leur nom :

  ./fractall-*-x86_64.AppImage fractall-cli --type 3 --output out.png
  ./fractall-*-x86_64.AppImage fractall-quality suite

ou via un lien symbolique, qui rend l'appel transparent :

  ln -s fractall-0.8.2-x86_64.AppImage fractall-cli
  ./fractall-cli --type 3 --zoom 1e20 --iterations 5000 --output deep.png

Sur une machine sans FUSE (conteneur, serveur minimal), l'AppImage sait se
déballer elle-même :

  ./fractall-*-x86_64.AppImage --appimage-extract-and-run


Prérequis système
-----------------
Bâtie sous Ubuntu 18.04 (glibc 2.27) : tourne sur toute distribution plus
récente. Restent à la charge du système la glibc et la pile graphique — X11 /
Wayland / xkbcommon / libGL / Vulkan sont chargés à chaud (dlopen) et l'ABI du
pilote graphique est par nature celle de la machine hôte : l'empaqueter
casserait l'accélération. Tout desktop Linux les fournit.

Le rendu GPU (wgpu : Vulkan, sinon OpenGL) est optionnel — sans pilote
utilisable, tout le moteur tourne sur CPU.


Où sont écrits les fichiers
---------------------------
L'AppImage est en lecture seule et n'installe rien. Les seules écritures :
  - les images / maps / projets vidéo là où VOUS les demandez ;
  - ~/.config/fractall/wisdom.toml, si vous lancez le benchmark machine
    (`fractall-cli --wisdom-bench`) qui calibre l'arbitrage CPU/GPU.


Prise en main rapide (GUI)
--------------------------
  molette          zoom avant/arrière ancré au curseur
  clic gauche      glisser = sélection rectangle de zoom
  clic droit       zoom arrière
  clic milieu      panoramique
  C / R            cycler palette / répétition de couleur
  J                bascule Julia (aperçu au survol)
  S                capture PNG (les paramètres exacts sont écrits dans le PNG —
                   le glisser-déposer d'une image dans la fenêtre restaure la vue)
  F1…F12           changer de type de fractale
  0                réinitialiser la vue


Sources et documentation : https://github.com/habib256/fractall-rust
