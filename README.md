# Categorisation Automatique des Offres d'Emploi
Ce dépôt contient des scripts permettant d'entraîner un modèle de catégorisation automatique d'offres d'emploi et de le tester. Il est conçu pour fonctionner avec des notebooks Google Colaboratory.

## Fichiers du dépôt
1. **nlp_test.py**: Script pour tester le modèle de catégorisation automatique. Il charge le modèle pré-entraîné, le vectoriseur, nettoie et transforme les données d'entrée, et prédit les catégories pour un ensemble donné d'offres d'emploi.
   - Lien vers le notebook d'origine: [Colab Notebook](https://colab.research.google.com/drive/1bdG98FOx0O4AyxCIwJccKoRIljSV4us3)
   
2. **nlp_train.py**: Script pour entraîner le modèle de catégorisation automatique. Il charge les données, les nettoie, divise l'ensemble de données en train/test, transforme les textes avec TF-IDF, entraîne un modèle de régression logistique, évalue le modèle et sauvegarde le modèle et le vectoriseur.
   - Lien vers le notebook d'origine: [Colab Notebook](https://colab.research.google.com/drive/1VgoODJPicoZ5YxjWPyGGAMzdR5eg4FcU)

## Utilisation
1. Assurez-vous d'avoir accès à Google Colaboratory.
2. Ouvrez les notebooks fournis ci-dessus.
3. Exécutez les scripts dans l'environnement Google Colab.
4. Les fichiers nécessaires au fonctionnement des scripts doivent être présents dans le répertoire `/content/drive/MyDrive/Rocket4Sales/`.

## Librairies nécessaires
- pandas
- re
- nltk
- sklearn
- joblib
- google.colab

## Auteur
Adrien Alvarez
