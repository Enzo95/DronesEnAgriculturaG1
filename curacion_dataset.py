import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.get_cachedir()
import matplotlib.pyplot as plt
import seaborn as sns
import os
import chardet

def curacion_dataset(dfa, dfc):

    # Se detectó que hay claves NO únicas, se define resetear
    dfa = dfa.reset_index()
    # Chequeamos si hay filas repetidas
    print('Cantidad de filas repetidas en dataset drone (con el mismo contenido): ',len(dfa[dfa.duplicated()]))
    # Chequeamos si hay filas repetidas
    print('\nCantidad de filas repetidas en dataset clima (con el mismo contenido): ', len(dfc[dfc.duplicated()]))
    
    # Le asiganamos el tipo correcto a las variables de fechas
    print('\nTransformacion de columnas fechas a datatime')
    dfa['Fecha de madurez']=dfa['Fecha de madurez'].replace(regex={'2-Dec': '2/12/2018'}) #correjimo a mano para que no haya errores luego
    #convertimos las fechas a datetime
    dfa['Fecha de madurez']=pd.to_datetime(dfa['Fecha de madurez'])
    dfa['Fecha de espigaz?']=pd.to_datetime(dfa['Fecha de espigaz?'])

    # Corregimos nombres de columnas en dataset dfa
    print('\nEliminacion de caracteres raros de las columnas del dataset Dron')
    dfa.columns[~dfa.columns.str.match(r'^(\w+)$')]
    dfa.columns = dfa.columns.str.replace('ID ', 'ID')
    dfa.columns = dfa.columns.str.replace(' ', '_')
    dfa.columns = dfa.columns.str.replace('', '')
    dfa.columns = dfa.columns.str.replace('(', '')
    dfa.columns = dfa.columns.str.replace(')', '')
    dfa.columns = dfa.columns.str.replace('%', 'porcentaje')
    dfa.columns = dfa.columns.str.replace('?', '')
    dfa.columns = dfa.columns.str.replace('Fecha_de_espigaz', 'Fecha_de_espigazon')
    dfa.columns[~dfa.columns.str.match(r'^(\w+)$')]

    # Corregimos nombres de columnas en dataset dfc
    print('Eliminacion de caracteres raros de las columnas del dataset Clima')
    dfc.columns[~dfc.columns.str.match(r'^(\w+)$')]
    dfc.columns = dfc.columns.str.replace(' ', '_')
    dfc.columns = dfc.columns.str.replace('[', '')
    dfc.columns = dfc.columns.str.replace(']', '')
    dfc.columns = dfc.columns.str.replace('/', '_')
    dfc.columns = dfc.columns.str.replace('Tem.', 'Temp')
    dfc.columns = dfc.columns.str.replace('Raf.', 'Rafaga')
    dfc.columns = dfc.columns.str.replace('%', 'porcentaje')
    dfc.columns = dfc.columns.str.replace('°C', 'centigrados')
    dfc.columns = dfc.columns.str.replace('Inten.', 'Inten')
    dfc.columns[~dfc.columns.str.match(r'^(\w+)$')]
    
    
    print('\nChequeo de valores nulos')
    valores_faltantes = pd.DataFrame([dfa.isnull().sum(),
                                   dfa.isnull().sum()/len(dfa)]).transpose().rename(
    columns = {0:'Cantidad_NaN',1:'Porcentaje_Nan_s_Total'})

    valores_faltantes.loc[valores_faltantes['Cantidad_NaN']>0].style.format({'Porcentaje_Nan_s_Total':"{:.2%}"})

    valores_faltantes = pd.DataFrame([dfc.isnull().sum(),
                                       dfc.isnull().sum()/len(dfc)]).transpose().rename(
        columns = {0:'Cantidad_NaN',1:'Porcentaje_Nan_s_Total'})

    valores_faltantes.loc[valores_faltantes['Cantidad_NaN']>0].style.format({'Porcentaje_Nan_s_Total':"{:.2%}"})
    
    
    # Replace 'Manchas_foliares' column values different to 'cero' to value 'otros'
    print('\nAplicando codificacion de variables')
    dfa['Manchas_Foliares'] = dfa['Manchas_Foliares'].replace(['diez', 'ocho/uno', 'ocho/dos', 'ocho/diez'], 'otros')
    # Replace 'Fusariosis' column values different to 'cero' to value 'otros'
    dfa['Fusariosis'] = dfa['Fusariosis'].replace(['uno/uno', 'dos/uno'], 'otros')
    
    # Ciclos; Asignar el valor 0 al ciclo corto y valor 1 al ciclo largo
    dfa['Ciclos'] = dfa['Ciclos'].replace('CC', 0)
    dfa['Ciclos'] = dfa['Ciclos'].replace('CL', 1)

    # Conjunto de datos; Asignar 1 correspondiente a la 1° fecha de siembra, o 0 en caso contrario.
    dfa['Conjunto_de_datos'] = dfa['Conjunto_de_datos'].replace('2daSiembra', 0)
    dfa['Conjunto_de_datos'] = dfa['Conjunto_de_datos'].replace('1er Siembra', 1)
    
    print('\nAplicando tests de integridad')
    test_integridad(dfa)
    
    # Corregimos las inconsistencias en las fechas de madurez
    print('\nCorreccion de las inconsistencias en las fechas de madurez\n')
    dfa['Fecha_de_madurez'] = dfa['Fecha_de_madurez'].mask(dfa['Fecha_de_madurez'].dt.year == 2015, 
                                dfa['Fecha_de_madurez'] + pd.offsets.DateOffset(year=2018))
    dfa['Fecha_de_madurez'] = dfa['Fecha_de_madurez'].mask(dfa['Fecha_de_madurez'].dt.year == 2016, 
                                dfa['Fecha_de_madurez'] + pd.offsets.DateOffset(year=2018))
    dfa['Fecha_de_madurez'].value_counts()  #revisamos que las fechas esten ok

    # Corregimos las inconsistencias en las fechas de espigazon
    print('Correccion de las inconsistencias en las fechas de espigazon\n')
    dfa['Fecha_de_espigazon'] = dfa['Fecha_de_espigazon'].mask(dfa['Fecha_de_espigazon'].dt.year == 2015, 
                                dfa['Fecha_de_espigazon'] + pd.offsets.DateOffset(year=2018))
    dfa['Fecha_de_espigazon'] = dfa['Fecha_de_espigazon'].mask(dfa['Fecha_de_espigazon'].dt.year == 2016, 
                                dfa['Fecha_de_espigazon'] + pd.offsets.DateOffset(year=2018))
    dfa['Fecha_de_espigazon'].value_counts() #revisamos que las fechas esten ok

    # Corregimos las inconsistencias en la variable Variedad
    print('Correccion de inconsistencias en variable Variedad\n')
    dfa.Variedad.value_counts()
    dfa['Variedad'] = dfa.Variedad.replace('Variedad_U ','Variedad_U')

    # Imputamos valores nulos de la variable dias_entre_fechas y convertimos a int
    print('Correccion de inconsistencias en variable dias_entre_fechas\n')
    dfa['dias_entre_fechas'] = pd.to_numeric(dfa['dias_entre_fechas'],
                                             errors='coerce').fillna(pd.to_numeric(
                                                 dfa['dias_entre_fechas'],
                                                 errors='coerce').mean())

    # Verificamos que las inconsistencias se hayan solucionado
    print('\nAplicando tests de integridad')
    test_integridad(dfa)

    outliers_data = []
    num_features = dfa.columns[(dfa.dtypes != 'object') & (dfa.dtypes != 'datetime64[ns]')]
    num_features_with_outliers = num_features.copy()
    
    for col in num_features:
        dfa_without_outliers = dfa.copy()
        dfa_without_outliers = clean_outliers(dfa_without_outliers, col)
        outliers_quantity = len(dfa) - len(dfa_without_outliers)
        outliers_percentage = str(round(outliers_quantity * 100 / len(dfa), 2)) + '%'
        if outliers_quantity > 0:
            outliers_data.append([outliers_quantity,outliers_percentage])
        else:
            num_features_with_outliers = num_features_with_outliers.drop(col)


    print("Variables con outliers dataset drone\n")
    df_otliers = pd.DataFrame(outliers_data, num_features_with_outliers, ['Cantidad_Outliers','Porcentaje_Outliers'])
    df_otliers.sort_values('Cantidad_Outliers', ascending=False)
    print(df_otliers)
    
    # Eliminamos todos los outliers del dataset con datos agronómicos
    print("\nEliminado Variables con outliers dataset drone\n")
   
    num_features = dfa.columns[(dfa.dtypes != 'object') & (dfa.dtypes != 'datetime64[ns]')]
    for col in num_features:
        dfa = clean_outliers(dfa, col)
        
    outliers_data = []
    num_features = dfc.columns[dfc.dtypes != 'object']
    num_features_with_outliers = num_features.copy()
    for col in num_features:
        dfc_without_outliers = dfc.copy()
        dfc_without_outliers = clean_outliers(dfc_without_outliers, col)
        outliers_quantity = len(dfc) - len(dfc_without_outliers)
        outliers_percentage = str(round(outliers_quantity * 100 / len(dfa), 2)) + '%'
        if outliers_quantity > 0:
            outliers_data.append([outliers_quantity,outliers_percentage])
        else:
            num_features_with_outliers = num_features_with_outliers.drop(col)
    
    print("Variables con outliers dataset Clima\n")
    df_otliers = pd.DataFrame(outliers_data, num_features_with_outliers, ['Cantidad_Outliers','Porcentaje_Outliers'])
    df_otliers.sort_values('Cantidad_Outliers', ascending=False)
    print(df_otliers)
    
    # Eliminamos todos los outliers del dataset con datos climáticos
    print("\nEliminado Variables con outliers dataset drone\n")
    num_features = dfc.columns[dfc.dtypes != 'object']
    for col in num_features:
        dfc = clean_outliers(dfc, col)
    
    #ordenamos variables por rendimiento
    dfa=dfa.sort_values(by=['RDTO'])
    
    # Removemos columnas irrelevantes o dependientes al target Rendimiento
    print("Eliminado variables irrelevantes o dependientes\n")
    dfa = dfa.drop(['p_grano', 'Peso_hecto','Peso_de_1000_granos','hum','Aspecto_','Altura','Vuelco','PAJUST'], axis=1)
    
    dfa=pd.get_dummies(dfa, columns=['Fusariosis','Manchas_Foliares'], drop_first=True) #agrupadas en 0 y 1


    dfc=dfc.reset_index()
    dfc['Fecha']=pd.to_datetime(dfc.Fecha)
    dfc_2=dfc.iloc[:, [0,1,4,7,12,13,15,18]] #trabajamos con los promedios
    dfc_2.head()
    
    print("Realizando unifacion de ambos dataset\n")
    dron_final=merged_inner = pd.merge(left=dfa,right=dfc_2, left_on='Fecha_de_espigazon', right_on='Fecha')

    return dron_final
    

def test_Cant_siembras(df):
    if len(df['Conjunto_de_datos'].value_counts()) <= 2:
        return 0
    else:
        return 1

def test_consist_fecha_madurez(df):
    if len(df[df['Fecha_de_madurez'].dt.year != 2018]) == 0:
      return 0
    else:
      return 1

def test_consist_fecha_espigazon(df):
    if len(df[df['Fecha_de_espigazon'].dt.year != 2018]) == 0:
      return 0
    else:
      return 1

def test_cant_repet(df):
      if df.Rep.isin([1,2,3]).all():
        return 0
      else:
        return 1

def test_cant_genotipos(df):
      if len(df.Variedad.value_counts()) == 22:
        return 0
      else:
        return 1

def test_cant_dias_dif(df):
      if pd.to_numeric(df['dias_entre_fechas'],errors='coerce').isnull().any():
        return 1
      else:
        return 0

def test_enfermedad_rendimiento(df):
      if len(df[((df['Roya_porcentaje'] > 1) | 
                (df['Carbon_porcentaje'] == 10) | (df['Adversidades'] >= 5)) &
                (df['RDTO'] > df['RDTO'].mean())]) == 0:
        return 0
      else:
        return 1

def test_integridad(df):
    """
    Ejecuta uno por uno los tests e informa el resultado
    """
        
    print('Test Cantidad de Siembras: %s' % 
          ('ERROR' if test_Cant_siembras(df) else 'EXITOSO'))
    print('Test Fecha de Madurez: %s' % 
          ('ERROR' if test_consist_fecha_madurez(df) else 'EXITOSO'))
    print('Test Fecha de Espigazon: %s' % 
          ('ERROR' if test_consist_fecha_espigazon(df) else 'EXITOSO'))
    print('Test Cantidad de Repeticiones: %s' % 
          ('ERROR' if test_cant_repet(df) else 'EXITOSO'))
    print('Test Cantidad de Genotipos: %s' % 
          ('ERROR' if test_cant_genotipos(df) else 'EXITOSO'))
    print('Test Cantidad de dias entre Madurez y Espigazon: %s' % 
          ('ERROR' if test_cant_dias_dif(df) else 'EXITOSO'))
    print('Test relacion Enfermedad vs Rendimiento: %s' % 
          ('ERROR' if test_enfermedad_rendimiento(df) else 'EXITOSO'))
        
    return
    
def clean_outliers(dataset, column_name):
    """Returns dataset removing the outlier rows from column @column_name."""
    interesting_col = dataset[column_name]
    # Here we can remove the outliers from both ends, or even add more restrictions.
    mask_outliers = (
        np.abs(interesting_col - interesting_col.mean()) <= (3 * interesting_col.std()))
    return dataset[mask_outliers]
    
    