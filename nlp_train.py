#Import des librairies
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

#Import des fonctions
def clean_text(text):
    # Suppression des caractères non alphabétiques
    text = re.sub('[^a-zA-Z\s]', '', text)
    # Conversion en minuscules
    text = text.lower().strip()
    # Suppression des stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('french')])
    return text

#Import des données
Excel = pd.ExcelFile('/content/Alternance Data scientist - Cas 1.xlsx')
train_test_df = Excel.parse('JobTitle Correspondance', usecols=['Job', 'Job référent'], skiprows=1)

#Nettoyage des données
train_test_df['Job'] = train_test_df['Job'].apply(clean_text)
train_test_df.dropna(inplace=True)

#Séparation des données en entrées (X) et sorties (y)
X = train_test_df['Job']
y = train_test_df['Job référent']

#Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Transformation des données textuelles en utilisant TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Création du modèle de régression logistique
model = LogisticRegression()

#Entraînement du modèle
model.fit(X_train_tfidf, y_train)

#Prédiction sur l'ensemble de test
y_pred = model.predict(X_test_tfidf)

#Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#Connection à Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Sauvegarde du modèle
dump(model, '/content/drive/MyDrive/Rocket4Sales/moodel_rocket_nlp.joblib')

#Sauvegarde du vectoriseur
dump(vectorizer, '/content/drive/MyDrive/Rocket4Sales/vectorizer_rocket_nlp.joblib')
