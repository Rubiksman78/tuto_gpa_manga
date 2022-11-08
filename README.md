#   GPA Manga

Projet pour générer des mangas en couleur à partir de leurs versions en Noir et Blanc.

## Organisation

- networks:
    - neural_networks.py : architecture du générateur et du discriminateur
    - pix2pix.py : modèle Pix2Pix (train_step)
    - cyclegan.py : modèle CycleGAN (train_step)
- scripts:
    - utils.py : fonctions utilitaire (plot,log..etc)
    - pdf_to_images.py : convertir des mangas pdf en dossier d'images
    - process_data.py : création d'un dataset de paires A/B pour TF
- ressources:
    articles, liens utiles


