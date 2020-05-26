import pandas as pd
import numpy as np

'''
Param path  to file y name of file
'''
def read_csv(path,filename):
    df = pd.read_csv(path+filename)
    print(df.head(50))
    print(df.info())
    return df

'''
Param Dataframe
'''
def clean_data(df):

    print("El primer punto es borrar la columnas que no tienen ningún valor o no"
          " están correctamente rellenadas")
    df.drop(['Respuesta_al_tweet','Respuesta_al_usuario_id',
             'Respuesta_al_usuario_nombre','Localización','Ciudad',
             'Contribuciones','Coordenadas'], axis=1)

    # Get a Series object containing the data type objects of each column of Dataframe.
    # Index of series is column name.
    dataTypeSeries = df.dtypes
    print('Data type of each column of Dataframe :')
    print(dataTypeSeries)

    #transformo el campo Fecha_creación a datatime
    df['Fecha_creación'] = pd.to_datetime(df['Fecha_creación'],
                                             errors='coerce')
    df = df.dropna(subset=['Fecha_creación'])
    print(df.dtypes)

    #creo la columna mes para las agrupaciones
    df['mes'] = pd.DatetimeIndex(df['Fecha_creación']).month
    print(df.head())

    # creo la columna dia para las agrupaciones
    df['dia'] = pd.DatetimeIndex(df['Fecha_creación']).day
    print(df.head())

    # creo la columna dia para las agrupaciones
    df['hora'] = pd.DatetimeIndex(df['Fecha_creación']).hour
    print(df.head())
    return df



def new_pandas_agrupado(df):
    aggregation = ["mes","dia","hora"]
    grouped = df.groupby(aggregation).hora.agg('count').to_frame('total').reset_index()
    print('He obtenido que el mes 4 dia 27 a las 13 es la hora más popular')
    print(grouped.iloc[grouped['total'].idxmax()] )

    aggregation = ["mes", "dia", ]
    grouped2 = df.groupby(aggregation).dia.agg('count').to_frame(
        'total').reset_index()
    print('He obtenido que el mes 4 dia 14 es el dia con más tweets.')
    print(grouped.iloc[grouped['total'].idxmax()])





if __name__ == '__main__':
    _path_file = '../files/'
    _file_name = 'result.csv'
    df = read_csv(_path_file,_file_name)
    df=clean_data(df)
    agrupado=new_pandas_agrupado(df)


