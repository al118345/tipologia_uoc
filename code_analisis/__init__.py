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





if __name__ == '__main__':
    _path_file = '../files/'
    _file_name = 'result.csv'
    df = read_csv(_path_file,_file_name)
    clean_data(df)

