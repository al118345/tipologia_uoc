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
    #print(df.head(50))
    #print(df.info())
    return df


def write_csv(path,filename,df):
    df.to_csv(path+filename, encoding='utf-8')

def to_latex(df):
    number = 0
    voy_por = 0
    for i in df.columns:
        if number == 4:
            df1 = df.iloc[:, voy_por:voy_por + number]
            voy_por = voy_por + number  # Remember that Python does not slice inclusive of the ending index.
            number = 0
            print(df1[:5].to_latex(index=False))
        else:
            number = number + 1
    df1 = df.iloc[:, voy_por:voy_por + number]

    print(df1[:5].to_latex(index=False))



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
    
    df['mes-dia'] = pd.DatetimeIndex(df['Fecha_creación']).strftime("%") +'-'+pd.DatetimeIndex(df['Fecha_creación']).strftime("%d")

    return df

def check_Empty(df):
    numericalData = df
    #Check if there is any null
    print(numericalData.isnull().values.any())

def new_pandas_agrupado(df):
    aggregation = ["mes","dia","hora"]
    grouped = df.groupby(aggregation).hora.agg('count').to_frame('total').reset_index()
    #print('He obtenido que el mes 4 dia 27 a las 13 es la hora más popular')
    print(grouped.iloc[grouped['total'].idxmax()] )

    aggregation = ["mes", "dia" ]
    grouped2 = df.groupby(aggregation).dia.agg('count').to_frame(
        'total').reset_index()
    #print('He obtenido que el mes 4 dia 14 es el dia con más tweets.')
   # print(grouped2.iloc[grouped2['total'].idxmax()])
    return grouped2


def add_sentiment(df):
    popularidad_list = []
    popularidad_list_text = []

    number = df.shape[0]
    print(number)
    for i in df['Texto']:
        analisis = TextBlob(i)
        try:
            language = analisis.detect_language()

            if language == 'en':
                analysis_ready = analisis
            else:
                if language == 'en':
                    analysis_ready = analisis
                else:
                    try:
                        analysis_ready = analisis.translate(to='en')
                    except:
                        analysis_ready = analisis
        except:
            analysis_ready = analisis

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


def GraficarDatosSentimientos(numeros_list, popularidad_list):
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



def GraficarDatosSentimientos_lista(df):
    axes = plt.gca()
    axes.set_ylim([-1, 2])
    df['mes-dia'] = pd.DatetimeIndex(df['Fecha_creación']).strftime(
        "%m") + '-' + pd.DatetimeIndex(df['Fecha_creación']).strftime("%d")
    aggregation = ['mes-dia']

    grouped2 = df.groupby(aggregation).sentimiento.agg('mean').to_frame(
        'mediana').reset_index()

    plt.scatter(grouped2['mes-dia'].values.tolist(), grouped2["mediana"].values.tolist())
    popularidadPromedio = (sum(grouped2["mediana"].values.tolist())) / (len(grouped2["mediana"].values.tolist()))
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




def show_evoluction_pandemic_ccaa(df):
    # creo la columna mes para las agrupaciones
    df['mes-dia'] = pd.DatetimeIndex(df['Fecha_creación']).strftime(
        "%m") + '-' + pd.DatetimeIndex(df['Fecha_creación']).strftime("%d")

    df['hora'] = pd.DatetimeIndex(df['Fecha_creación']).hour

    numericalData = df.iloc[:, 2:-1]
    numericalData = numericalData.drop_duplicates(keep='last', subset='mes-dia')
    ccaa = []
    ranges = []
    for i in range(0, 19):
        ranges.append(i)
    ranges.append(-1)

    dates = []
    for i in numericalData['mes-dia']:
        dates.append(str(i))

    ccaa = numericalData.iloc[:, ranges]
    countries = ccaa.iloc[:, :-1].columns
    for i, country in enumerate(countries):
        plt.xlabel('Fecha')
        plt.ylabel('Casos')
        plt.title('Cases by date in %s' % country)
        plt.plot(ccaa['mes-dia'], ccaa[countries[i]])
        plt.xticks(rotation='vertical')
        plt.show()

def show_evoluction_pandemic_world_cases(df):
    # creo la columna mes para las agrupaciones
    df['mes-dia'] = pd.DatetimeIndex(df['Fecha_creación']).strftime(
        "%m") + '-' + pd.DatetimeIndex(df['Fecha_creación']).strftime("%d")

    df['hora'] = pd.DatetimeIndex(df['Fecha_creación']).hour
    ccaa = []
    ranges = []
    for i in range(0, 19):
        ranges.append(i)
    ranges.append(-1)

    numericalData = df.iloc[:, 2:-1]
    numericalData = numericalData.drop_duplicates(keep='last', subset='mes-dia')

    world = numericalData.iloc[:,19:33]

    countries = world.columns
    n_rows = len(countries)-1 // 2 + 1
    for i, country in enumerate(countries):
        plt.xlabel('Fecha')
        plt.ylabel('Casos')
        plt.title('Cases by date in %s'% country)
        plt.plot(ccaa['mes-dia'],world[countries[i]])
        plt.xticks(rotation='vertical')
        plt.show()

def show_evoluction_pandemic_world_dead(df):
    df['mes-dia'] = pd.DatetimeIndex(df['Fecha_creación']).strftime(
        "%m") + '-' + pd.DatetimeIndex(df['Fecha_creación']).strftime("%d")

    ccaa = []
    ranges = []
    for i in range(0, 19):
        ranges.append(i)
    ranges.append(-1)

    numericalData = df.iloc[:, 2:-1]
    numericalData = numericalData.drop_duplicates(keep='last', subset='mes-dia')


#Se muestra la evolución a nivel mundial delas MUERTES
    worldDead = numericalData.iloc[:,33:-1]

    countries = worldDead.columns
    n_rows = len(countries)-1 // 2 + 1
    for i, country in enumerate(countries):
        plt.xlabel('Fecha')
        plt.ylabel('Muertes')
        plt.title('Cases by date in %s'% country)
        plt.plot(ccaa['mes-dia'],worldDead[countries[i]])
        plt.xticks(rotation='vertical')
        plt.show()


def GraficarDatosSentimientos_boxplot(sentimiento):
    sentimiento['mes-dia'] = pd.DatetimeIndex(sentimiento['Fecha_creación']).strftime(
        "%m") + '-' + pd.DatetimeIndex(sentimiento['Fecha_creación']).strftime("%d")
    sentimiento.boxplot(by='mes-dia',
                           column=['sentimiento'],
                           grid=False)
    plt.xticks(rotation=90)
    plt.savefig('box_dias.png')
    plt.show()
    sentimiento.boxplot(
                        column=['sentimiento']
                        )
    plt.savefig('box_completo.png')
    #plt.show()


def word_cloud_save(sentimiento):
    import nltk
    import re
    from wordcloud import WordCloud, STOPWORDS

    # Este proceso puede hacerse antes de forma manual, descargar las stopwords de la librerñia nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words_sp = set(stopwords.words('spanish'))
    stop_words_en = set(stopwords.words('english'))

    stop_words = stop_words_sp | stop_words_en

    from nltk import tokenize
    import matplotlib
    matplotlib.style.use('ggplot')
    pd.options.mode.chained_assignment = None





    # Remover URLs, RTs, y t tter handles
    for i in range(len(sentimiento['Texto'])):
        sentimiento['Texto'][i] = " ".join([word for word in sentimiento['Texto'][i].split()
                                      if  'http' not in word and '@' not in word and '<' not in word and 'RT' not in word])

    # Monitorear que se removieron las menciones y URLs


    # Remover puntuación, se agregan símbolos del español
    sentimiento['Texto'] = sentimiento['Texto'].apply(lambda x: re.sub('[¡!@#$:).;,¿?&]', '', x.lower()))
    sentimiento['Texto'] = sentimiento['Texto'].apply(lambda x: re.sub('  ', ' ', x))
    stoplist = set(stopwords.words("spanish"))

    sentimiento['Texto'] = sentimiento['Texto'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stoplist)]))



    # Crear la imagen con las palabras más frecuentes

    wordcloud = WordCloud(background_color="white", stopwords=stop_words, random_state=2016).generate(
        " ".join([i for i in sentimiento['Texto']]))
    # Preparar la figura
    plt.figure(num=400, figsize=(20, 10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("COVID")
    plt.savefig('words.png')
    plt.show()


def agrupacion_sentimiento_dia_mes_cant(sentimiento):
    sentimiento['mes-dia'] = pd.DatetimeIndex(
        sentimiento['Fecha_creación']).strftime(
        "%m") + '-' + pd.DatetimeIndex(sentimiento['Fecha_creación']).strftime(
        "%d")
    aggregation = ['mes-dia']
    agrupado= sentimiento.groupby(aggregation).sentimiento.agg('mean').to_frame(
        'sentimiento').reset_index()



    agrupado2= sentimiento.groupby(['mes-dia']).World_dead.agg('max').to_frame(
        'muertes_mundo')
    agrupado3 = sentimiento.groupby(['mes-dia']).Spain_dead.agg(
        'max').to_frame(
        'muertes_españa')

    aux=  pd.merge(agrupado, agrupado2, on='mes-dia')
    aux=  pd.merge(aux, agrupado3, on='mes-dia')
    return aux


from scipy import stats
from statistics import mean,stdev
def check_Welch():
    totalCases = []
    totalDead = []
    for j in numericalData['World_cases']:
            totalCases.append(int(str(j).replace(',','')))

    for j in numericalData['World_dead']:
            totalDead.append(int(str(j).replace(',','')))
    
    t_score = stats.ttest_ind_from_stats(mean1=mean(totalCases), std1=stdev(totalCases), nobs1=2661, \
                                   mean2=mean(totalDead), std2=stdev(totalDead), nobs2=2661, \
                                   equal_var=False)
    print(t_score)
    
from scipy.stats import levene 

def check_Variance():
    totalCases = []
    totalDead = []
    for j in numericalData['World_cases']:
            totalCases.append(int(str(j).replace(',','')))

    for j in numericalData['World_dead']:
            totalDead.append(int(str(j).replace(',','')))

    levene(totalCases,totalDead)
    
    
    
def check_Outliers(df):
    countries = df.columns
    for i, country in enumerate(countries):
        fig = plt.figure()
        plt.title(country)    
        intData = []
        for j in numericalData[countries[i]]:
            intData.append(int(str(j).replace(',','')))
        plt.boxplot(intData)
        plt.show()
        
        
def check_Normality(df):
    countries = numericalData.iloc[:,:-1].columns
    for i, country in enumerate(countries):
        fig = plt.figure()
        plt.title(country)    
        intData = []
        for j in numericalData[countries[i]]:
            intData.append(int(str(j).replace(',','')))
        plt.hist(intData)
        plt.show()
        
import statsmodels.api as sm
     
def plot_QQ(df):    
    countries = numericalData.iloc[:,:-1].columns
    for i, country in enumerate(countries):
  
        intData = []
        for j in numericalData[countries[i]]:
            intData.append(int(str(j).replace(',','')))
        fig = sm.qqplot(np.stack(intData), fit=True,line='45')
        plt.title(country)    
        plt.show()



def plot_corr(df):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    import matplotlib.pyplot as plt
    df = df.drop(df.columns[0], axis=1)
    df['sentimiento'] =  pd.to_numeric(df['sentimiento'])
    df["muertes_mundo"] = df["muertes_mundo"].str.replace(',', ".")

    df["muertes_españa"] = df["muertes_españa"].str.replace(',', ".")

    df['muertes_mundo'] =  pd.to_numeric(df['muertes_mundo'])

    df['muertes_españa'] =  pd.to_numeric(df['muertes_españa'])

    plt.matshow(df.corr())
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()
    plt.savefig('cor.png')

    plt.show()

    from pandas.plotting import scatter_matrix

    scatter_matrix(df, figsize=(6, 6))
    plt.savefig('scatter.png')

    plt.show()

    import seaborn as sns

    #-- Matriz de correlación
    df_corr = df.corr()
    mask = np.triu(df_corr, k=1)
    sns_plot = sns.heatmap(df_corr, cmap= 'YlGnBu', annot=True, fmt=".2f", mask=mask )
    figure = sns_plot.get_figure()
    figure.savefig('cor.png', dpi=400)


if __name__ == '__main__':
    _path_file = '../files/'
    _file_name = 'result.csv'
    #read_csv(_path_file,_file_name)
    _file_name_result = 'analizado.csv'

    _file_name = _file_name_result

    #sentimiento = add_sentiment(df[:1000])
    sentimiento= read_csv(_path_file,_file_name_result)
    sentimiento = sentimiento.drop(sentimiento.columns[0], axis=1)
    sentimieento_agrupado=agrupacion_sentimiento_dia_mes_cant(sentimiento)
    #plot_corr(sentimieento_agrupado)
    #GraficarDatosSentimientos(list(range(1, sentimiento.shape[0]+1)) , sentimiento['sentimiento'].values.tolist())
    #GraficarDatosSentimientos_lista(sentimiento)
    GraficarDatosSentimientos_boxplot(sentimiento)
    #word_cloud_save(sentimiento)

    #sentimiento_agrupado = new_pandas_agrupado(sentimiento)

    #show_evoluction_pandemic_world_dead(sentimiento)





