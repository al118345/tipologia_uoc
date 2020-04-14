import tweepy
import csv #Import csv
# import libraries
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd
from datetime import date
import os
from autenticate import get_auth

'''
Este metodo se encarga de acceder a las web   'https://covid19.isciii.es'
para obtener los infectados por comunidades autónomas

Return dataframe con las siguiente información 
                   0       1                       2   ...        16     17      
CCAA        Andalucía  Aragón  Principado de Asturias  ...  La Rioja  Ceuta  
12/04/2020      10006    4070                    1892  ...      3279     93       

'''
def get_info_web_covid_comunidaddes():
    driver = webdriver.Chrome('chromedriver.exe')
    urlpage = 'https://covid19.isciii.es'
    # get web page
    driver.get(urlpage)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    ccaa = []
    casos = []
    today = date.today()
    # Find datatable
    for row in soup.find_all('table')[1].findAll('tr'):
        if (row.find_all('td') != []):
            ccaa.append(row.findAll('td')[0].text)
            casos.append(row.findAll('td')[1].text)

    df = pd.DataFrame(list(zip(ccaa, casos)),
                      columns=['CCAA', today.strftime("%d/%m/%Y")])
    return df.T




'''
Este metodo se encarga de acceder a las web 'https://www.worldometers.info/
coronavirus/#nav-today'
para obtener los infectados por paises

Devuelver un datafreme con la siguiente información

                 0           1                  2    ...     225     226   
País            World  \nEurope\n  \nNorth America\n  ...  Total:  Total:    
12/04/2020  1,812,970     874,633            574,487  ...  14,478     721 
Muertes       112,255      75,816             22,753  ...     768      13    

'''
def info_web_covid_mundo() :
    urlpage2 = 'https://www.worldometers.info/coronavirus/#nav-today'
    driver = webdriver.Chrome('chromedriver.exe')
    today = date.today()
    # get web page
    driver.get(urlpage2)
    # execute script to scroll down the page
    driver.execute_script(
        "window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
    # Click buton to get todays data
    dot = driver.find_element_by_id('nav-today-tab')
    dot.click

    soup2 = BeautifulSoup(driver.page_source, 'html.parser')
    country = []
    casosCountry = []
    muertesCountry = []

    # Find datatable
    for row in soup2.find_all('table')[0].findAll('tr'):
        if (row.find_all('td') != []):
            country.append(row.findAll('td')[0].text)
            casosCountry.append(row.findAll('td')[1].text)
            muertesCountry.append(row.findAll('td')[3].text)
    df2 = pd.DataFrame(list(zip(country, casosCountry, muertesCountry)),
                       columns=['País', today.strftime("%d/%m/%Y"), 'Muertes'])
    lista_paises = ['World', 'USA', 'Spain', 'Italy', 'France', 'Germany', 'UK',
                    'China', 'Iran', 'Turkey', 'Brazil', 'Russia', 'Japan',
                    'Morocco', 'Brazil']
    df2 = df2[df2['País'].isin(lista_paises)]
    return df2.T


'''
Código base para la obtención de los tweets. 
'''
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
            csvFile = open('result.csv', 'a', newline='')
            csvWriter = csv.writer(csvFile)
            if status is not False and status.text is not None:
                try:
                    texto = status.extended_tweet["full_text"]
                except AttributeError:
                    texto = status.text
                print(texto)
                info1 = get_info_web_covid_comunidaddes().iloc[1].tolist()
                info2 = info_web_covid_mundo()
                casos = info2.iloc[1].tolist()
                muertos = info2.iloc[2].tolist()
                linea = [status.created_at,
                         status.id, texto, status.source, status.truncated,
                         status.in_reply_to_status_id, status.in_reply_to_user_id,
                         status.in_reply_to_screen_name, status.geo, status.coordinates,
                         status.place,status.contributors, status.lang, status.retweeted]
                linea = linea + info1 + casos+muertos
                csvWriter.writerow(linea)
            print("Almacenamos Tweet")
            csvFile.close()
            print("fin")

    def on_error(self, status_code):
        print(status_code)
        return False



if __name__ == '__main__':
    print("===== Captador de tweets =====")
    # Get an API item using tweepy
    auth = get_auth()  # Retrieve an auth object using the function 'get_auth' above
    api = tweepy.API(auth)  # Build an API object.

    # Connect to the stream
    myStreamListener = MyStreamListener()
    while True:
        try:
            if os.path.isfile(
                    'result.csv'):
               print('Preparado el fichero')
            else:
                print('El no archivo existe.');
                csvFile = open('result.csv', 'w', newline='')
                csvWriter = csv.writer(csvFile)
                cabecera=['Fecha_creación','Id','Texto','Fuente','Truncado'
                    ,'Respuesta_al_tweet','Respuesta_al_usuario_id'
                    ,'Respuesta_al_usuario_nombre'
                    ,'Localización'
                    ,'Coordenadas'
                    ,'Ciudad'
                    ,'Contribuciones'
                    ,'Idioma'
                    ,'Retweeted'
                    ,'Andalucía'
                    ,'Aragón'
                    ,'Principado de Asturias'
                    ,'Islas Baleares'
                    ,'Canarias'
                    ,'Cantabria'
                    ,'Castilla y León'
                    ,'Castilla  La Mancha'
                    ,'Cataluña'
                    ,'Galicia'
                    ,'C.Valenciana'
                    ,'Extremadura'
                    ,'Comunidad de Madrid'
                    ,'Región de Murcia'
                    ,'Comunidad Foral de Navarra'
                    ,'País Vasco'
                    ,'La Rioja'
                    ,'Ceuta'
                    ,'Melilla','World_cases', 'USA_cases', 'Spain_cases',
                          'Italy_cases', 'France_cases',
                          'Germany_cases', 'UK_cases','China_cases', 'Iran_cases',
                          'Turkey_cases', 'Brasil_cases',
                          'Russia_cases', 'Japan_cases','Moroco_cases',
                'World_dead', 'USA_dead', 'Spain_dead', 'Italy_dead'
                    , 'France_dead','Germany_dead', 'UK_dead',
                          'China_dead', 'Iran_dead', 'Turkey_dead',
                          'Brasil_dead',
                         'Russia_dead', 'Japan_dead', 'Moroco_dead',

                ]
                csvWriter.writerow(cabecera)
                csvFile.close()
                print("Creación de la cabecera")
            myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
            print(">> Listening to tweets about #coronavirus en castellano:")
            myStream.filter(track=['Coronavirus'], languages=['es'])
        except:
            continue
    # End
    print("Terminado")



