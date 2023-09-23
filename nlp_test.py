#Import des librairies
import pandas as pd
from joblib import load
from google.colab import drive
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

#Import des fonctions
def clean_text(text):
    # Conversion en chaîne si ce n'est pas déjà le cas
    text = str(text)
    # Suppression des caractères non alphabétiques
    text = re.sub('[^a-zA-Z\s]', '', text)
    # Conversion en minuscules
    text = text.lower().strip()
    # Suppression des stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('french')])
    return text

#Import du modèle et du vectoriseur
model = load('/content/drive/MyDrive/Rocket4Sales/moodel_rocket_nlp.joblib')
vectorizer = load('/content/drive/MyDrive/Rocket4Sales/vectorizer_rocket_nlp.joblib')

#Import des données
xl = pd.ExcelFile('/content/Alternance Data scientist - Cas 1.xlsx')
predict_df = xl.parse('Linkedin job URL', usecols=['jobTitle', 'JObTitle Correspondance'])

#Nettoyage des données
predict_df.rename(columns={'jobTitle': 'Job', 'JObTitle Correspondance': 'Job référent'}, inplace=True)
predict_df['Job'] = predict_df['Job'].apply(clean_text)
predict_df.dropna(inplace=True)

# Transformation des données textuelles en utilisant le vectoriseur TF-IDF chargé
X_predict_tfidf = vectorizer.transform(predict_df['Job'])

# Prédiction en utilisant le modèle chargé
predict_df['Job référent'] = model.predict(X_predict_tfidf)

#Connection à Google Drive
drive.mount('/content/drive')

#Sauvegarde du DataFrame mis à jour
predict_df.to_csv('/content/drive/MyDrive/Rocket4Sales/predict_df_updated.csv', index=False)
