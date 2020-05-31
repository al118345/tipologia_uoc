import pandas as pd
import numpy as np
from textblob import TextBlob
import time
import datetime
import matplotlib.pyplot as plt
'''
Param path  to file y name of file
'''
def read_csv(path,filename):
    df = pd.read_csv(path+filename)
    print(df.head(50))
    print(df.info())
    return df


def write_csv(path,filename,df):
    df.to_csv(path+filename, encoding='utf-8')


'''
Param Dataframe
'''
def clean_data(df):
    #Se eliminan las columnas que no son útiles
    df = df.drop(['Respuesta_al_tweet','Fuente','Truncado','Respuesta_al_usuario_id','Respuesta_al_usuario_nombre','Localización','Ciudad','Contribuciones','Coordenadas','Id','Idioma','Retweeted'], axis=1)
    #Se procesan las fechas
    df = clean_dates(df)
    #df = clean_numerical(df)
    return df

def clean_dates(df):
    # Get a Series object containing the data type objects of each column of Dataframe.
    # Index of series is column name.
    dataTypeSeries = df.dtypes
   # print('Data type of each column of Dataframe :')
   # print(dataTypeSeries)

    #transformo el campo Fecha_creación a datatime
    df['Fecha_creación'] = pd.to_datetime(df['Fecha_creación'],
                                             errors='coerce')
    df = df.dropna(subset=['Fecha_creación'])
    #print(df.dtypes)

    #creo la columna mes para las agrupaciones
    df['mes'] = pd.DatetimeIndex(df['Fecha_creación']).month
    #print(df.head())

    # creo la columna dia para las agrupaciones
    df['dia'] = pd.DatetimeIndex(df['Fecha_creación']).day
    #print(df.head())

    # creo la columna dia para las agrupaciones
    df['hora'] = pd.DatetimeIndex(df['Fecha_creación']).hour
    #print(df.head())
    return df

def clean_numerical(df):
    numericalData = df.iloc[:,2:]
    
    #Check if there is any null
    print(numericalData.isnull().values.any())
    

def new_pandas_agrupado(df):
    aggregation = ["mes","dia","hora"]
    grouped = df.groupby(aggregation).hora.agg('count').to_frame('total').reset_index()
    #print('He obtenido que el mes 4 dia 27 a las 13 es la hora más popular')
    print(grouped.iloc[grouped['total'].idxmax()] )

    #aggregation = ["mes", "dia" ]
    #grouped2 = df.groupby(aggregation).dia.agg('count').to_frame(
    #    'total').reset_index()
    #print('He obtenido que el mes 4 dia 14 es el dia con más tweets.')
   # print(grouped2.iloc[grouped2['total'].idxmax()])
    return grouped


def add_sentiment(df):
    popularidad_list = []
    popularidad_list_text = []

    number = df.shape[0]
    print(number)


    for i in df['Texto']:
        analisis = TextBlob(i)
        language = analisis.detect_language()

        if language == 'en':
            analysis_ready = analisis
        else:
            analysis_ready = analisis.translate(to='en')

        if analysis_ready.sentiment.polarity > 0:
            popularidad_list_text.append('positive')
        elif analysis_ready.sentiment.polarity == 0:
            popularidad_list_text.append('neutral')
        else:
            popularidad_list_text.append('negative')
        time.sleep(1.2)
        analysis_ready = analysis_ready.sentiment
        popularidad = analysis_ready.polarity
        popularidad_list.append(popularidad)
        print(number)
        number=number-1
    df['sentimiento'] = popularidad_list
    df['sentimiento_texto'] = popularidad_list_text
    return df


def GraficarDatos(numeros_list, popularidad_list):
    axes = plt.gca()
    axes.set_ylim([-1, 2])

    plt.scatter(numeros_list, popularidad_list)
    popularidadPromedio = (sum(popularidad_list)) / (len(popularidad_list))
    popularidadPromedio = "{0:.0f}%".format(popularidadPromedio * 100)
    plt.text(0, 1.25,
             "Sentimiento promedio:  " + str(popularidadPromedio) + "\n" ,
             fontsize=12,
             bbox=dict(facecolor='none',
                       edgecolor='black',
                       boxstyle='square, pad = 1'))

    plt.title("Sentimientos sobre coronavirus en castellano en twitter")
    plt.xlabel("Numero de tweets")
    plt.ylabel("Sentimiento")
    plt.show()


if __name__ == '__main__':
    _path_file = '../files/'
    _file_name = 'result.csv'
    _file_name_result = 'analizado.csv'
    df = read_csv(_path_file,_file_name)
    df=clean_data(df)
    agrupado=new_pandas_agrupado(df)
    #sentimiento = add_sentiment(df[:1000])
    sentimiento = add_sentiment(df)

    write_csv(_path_file,_file_name_result,sentimiento)
    GraficarDatos(list(range(1, sentimiento.shape[0]+1)) , sentimiento['sentimiento'].values.tolist())



