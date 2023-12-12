###### Call library #####
import streamlit as st
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
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
#### UI ####
st.header("UAS PPW")
st.subheader("Crawling berita dari Detik.com")
var_input = st.input("Masukan judul berita yang ingin dicrawling : ")
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
    st.dataframe(frame_berita)
