###### Call library #####
import streamlit as st
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import requests
import numpy as np
import re, string
import nltk
from tqdm.auto import tqdm

nltk.download('popular')
nltk.download('stopwords')
tqdm.pandas()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import chain

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from googletrans import Translator
import matplotlib.pyplot as plt

##### Code #####
#### Function ####
def del_word(string_awal, kata_hapus):
    string_hasil = string_awal.replace(kata_hapus, '')
    return string_hasil

def cleaning(text):
    # Menghapus tag HTML
    text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))

    # Mengubah seluruh teks menjadi huruf kecil
    text = text.lower()

    # Menghapus spasi pada teks
    text = text.strip()

    # Menghapus Tanda Baca, karakter spesial, and spasi ganda
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub("â", "", text)

    # Menghapus Nomor
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Mengubah text yang berisi 'nan' dengan whitespace agar nantinya dapat dihapus
    text = re.sub('nan', '', text)

    return text
#### UI ####
st.header("UAS PPW")
st.subheader("Crawling berita dari Detik.com")
var_input = st.text_input("Masukan judul berita yang ingin dicrawling : ")
if st.button("Crawl") == True :
    pencarian = var_input
    cari = pencarian.replace(' ','+')
    tglAwal = '11/11/2023'
    tglAkhir = '11/12/2023'
    header={
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    }
    berita_list = []
    
    for halaman in range(10):
        url = f'https://www.detik.com/search/searchall?query={cari}&sortby=time&fromdatex={tglAwal}&todatex={tglAkhir}&page={halaman}'
        
        # Mengambil data dari detik.com
        req = requests.get(url,header)
        soup = BeautifulSoup(req.text, 'lxml')
        listberita = soup.find('div', class_='list media_rows list-berita')
        artikel = listberita.find_all('article')
    
        for x in artikel:
            url2 = x.find('a')['href']
            judul = x.find('a').find('h2').text
            
            # Mengambil data dari setiap konten
            urlkonten = requests.get(url2, header)
            soupkonten = BeautifulSoup(urlkonten.text, 'lxml')
            
            # Mencoba mengekstrak kategori dari struktur HTML
            kategori_element = soupkonten.find('span', class_='detail__label')
            kategori = kategori_element.text.strip() if kategori_element else 'Tidak Diketahui'
            
            konten = soupkonten.find_all('div', class_='detail__body-text itp_bodycontent')
    
            for x in konten:
                isi = x.find_all('p')
                y = [y.text for y in isi]
                fixkonten = ''.join(y).replace('\n','').replace('ADVERTISEMENT','').replace('SCROLL TO RESUME CONTENT','')
                
                # Menambahkan berita ke dalam daftar dengan kategori
                berita_list.append([judul, fixkonten,kategori])

    filter_berita = []

    for i in range(len(berita_list)):
        if berita_list[i][2] != 'detikNews':
            berita_list[i][2] = del_word(berita_list[i][2],"detik")
            filter_berita.append(berita_list[i])

    frame_berita = pd.DataFrame(filter_berita, columns =['Judul','Isi','Kategori'])
    st.text("Output :")
    st.dataframe(frame_berita)

    st.subheader("Preprocessing")
    df = frame_berita
    df['clean'] = df['Isi'].apply(lambda x: cleaning(x))
    df['tokenize'] = df['clean'].apply(lambda x: word_tokenize(x))
    stop_words = set(chain(stopwords.words('indonesian')))
    df['remove_stopword'] = df['tokenize'].apply(lambda x: [w for w in x if not w in stop_words])

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df['steming'] = df['remove_stopword'].progress_apply(lambda x: stemmer.stem(' '.join(x)).split(' '))
    df['Isi_terbaru'] = df['steming'].apply(lambda tokens: ' '.join(tokens))
    
    st.dataframe(df)
    st.subheader("Modeling")

    # Misal dataframe Anda bernama df
    X = df['Isi_terbaru'].values
    y = df['Kategori'].values
    
    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X)
    X_test_tfidf = vectorizer.transform(X)

    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_tfidf, y)

    y_pred = svm_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y, y_pred)
    st.text(f'Accuracy: {accuracy:.2f}')
    
    st.text('\nClassification Report:')
    st.text(classification_report(y, y_pred))

    st.subheader("Diagram")
    # Membuat diagram batang
    report_dict = classification_report(y, y_pred, output_dict=True)
    labels = list(report_dict.keys())[:-3]  # Mengambil label kelas (excludes avg/total row)
    precision = [report_dict[label]['precision'] for label in labels]
    recall = [report_dict[label]['recall'] for label in labels]
    f1_score = [report_dict[label]['f1-score'] for label in labels]
    support = [report_dict[label]['support'] for label in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.2
    index = range(len(labels))
    
    bar1 = ax.bar(index, precision, bar_width, label='Precision')
    bar2 = ax.bar([i + bar_width for i in index], recall, bar_width, label='Recall')
    bar3 = ax.bar([i + 2 * bar_width for i in index], f1_score, bar_width, label='F1-Score')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1-Score')
    ax.set_xticks([i + bar_width for i in index])
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Menambahkan nilai support di atas setiap bar
    for i, v in enumerate(support):
        ax.text(i + bar_width / 2, v + 5, str(v), ha='center', va='bottom')
    
    st.pyplot(fig)

    texts = X

    # Terjemahkan teks dari Bahasa Indonesia ke Bahasa Inggris
    translator = Translator()
    translated_texts = [translator.translate(text, src='id', dest='en').text for text in texts]
    
    text_list = []
    
    for l in range(len(X)):
        text_list.append("text {}".format(l+1))
    
    # Analisis sentimen untuk setiap teks yang sudah diterjemahkan
    sentiments = []
    for text in translated_texts:
        analysis = TextBlob(text)
        sentiment = analysis.sentiment.polarity
        sentiments.append(sentiment)
    
    # Menampilkan hasil analisis sentimen
    for i, sentiment in enumerate(sentiments):
        print(f'Translated Text {i+1}: {sentiment:.2f}')
    
    # Membuat diagram batang untuk sentimen
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(translated_texts)), sentiments, color=['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sentiments])
    ax.set_xticks(range(len(translated_texts)))
    ax.set_xticklabels(text_list, rotation=45, ha='right')
    ax.set_ylabel('Sentiment Polarity')
    ax.set_title('Sentiment Analysis (Translated)')
    
    # Menambahkan label di atas setiap bar
    for bar, sentiment in zip(bars, sentiments):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{sentiment:.2f}', ha='center', va='bottom' if sentiment >= 0 else 'top', color='white' if sentiment != 0 else 'black')
    
    st.pyplot(fig)
