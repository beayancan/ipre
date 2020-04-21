# importamos las liberías que vamos a utilizar
# herramientas y el dataset

import os, sys
import keras
import statistics
import collections
import numpy as np
from keras.datasets import reuters
from random import randrange

# Primero eliminaremos las referencias a elementos que no se encuentran
# dentro de las 1000 palabras más usadas en el vocabulario
# ademas de las entradas reservadas

  # Eliminamos
  # 0: '-PAD-'
  # 1: '-START-'
  # 2: '-UNK-'
  # 12: '3'
  # 17: 'reuter'

def filtrar_relevante(arreglo, por_eliminar=[0,1,2]):
  """
  Borra las palabras que pertenecen a los indices del array por_eliminar
  """
  return list(filter(lambda x: x not in por_eliminar, arreglo))

def eliminar_reservadas(x_array):
  """
  Eliminamos las entradas reservadas y entradas inutiles
  retorna el contenido homogeneo del doc
  """
  por_eliminar = [0,1,2,12,17]
  largo_test, = x_array.shape
  for i in range(largo_test):
    x_array[i] = filtrar_relevante(x_array[i], por_eliminar)
  return x_array

  # Para hacer un ejemplo más sencillo de entender
# Selecionaremos solo las primeras 7 clases

###########################################################################


def conteo_labels(n, p):
  """
  Para una mejor representacion de los labels
  utilizaremos medidores para que no supere ciertas barreras
  """
  return list( [0, randrange(p-3, p+3)] for _ in range(n) )

def reducir_labels(data_array, labels, k=7, pivote=25):
  """
  Filtramos los documentos que pertenezcan a las primeras k clases
  Retornamos el arreglo con los documentos y sus correspondientes labels
  """
  
  conteo = conteo_labels(max(labels), pivote)
  retorno, retorno_labels = list(), list()
  for i in range(len(data_array)):
    if labels[i] < k and conteo[labels[i]][0] < conteo[labels[i]][1]:
      retorno.append(data_array[i])
      retorno_labels.append(labels[i])
      conteo[labels[i]][0] += 1
  return np.array(retorno), retorno_labels



def rank2(A, W):
  """
  Recibe las matrices objetivo A y su matriz izquierda W
  Calcula la resolución iterativa de la minimización según el paper
  y obtenemos la minimización de H a partir de W
  Retorna las matrices W, H de las descomposición
  """
  m, n = np.shape(A)

  # resolvemos por minimos cuadrados
  H = np.linalg.solve(np.dot(np.transpose(W), W), np.dot(np.transpose(W), A))

  # Separamos en columnas
  w1, w2 = W[:, 0], W[:, 1]
  beta1, beta2 = np.linalg.norm(w1), np.linalg.norm(w2)

  # normalizamos
  u, v = np.dot(np.transpose(A), w1)/beta1, np.dot(np.transpose(A), w2)/beta2

  for j in range(n):
    # Para cada vector determinamos si cumple con la solucion
    retorno_j = np.zeros(2)
    if (H[:, j] >= 0).all():
      continue
    elif u[j]*beta1 >= v[j]*beta2:
      retorno_j[0] = u[j]
    else:
      retorno_j[1] = v[j]
    H[:, j] = retorno_j
  return W, H

def NMF_rank2(A, W=None, H=None, k=2, **kwargs):
  """
  Recibe la matriz objetivo y matrices iniciales
  Se realiza dos veces la minimización primero para H
  y luego para W
  Retorna la descomposición W, H de baja calidad
  """
  m, n = np.shape(A)

  # Iniciamos las matrices
  if W is None:
    W = np.random.rand(m, k)

  if H is None:
    H = np.zeros((k, n))
  
  # Realizamos las minimizaciones
  W, H = rank2(A, W)
  HT, WT = rank2(np.transpose(A), np.transpose(H))
  # Retornamos los valores que resultaron minimizados
  return np.transpose(WT), np.transpose(HT)



def calculo_NMF(A, max_iteraciones=15, k=2, W=None, H=None, error=0.5):
  """
  Recibe la matriz objetivo A, dimension k,
  máximo de iteraciones y matrices iniciales
  Realiza de forma recursiva la aplicación de rank-2
  para así obtener una mejor aproximación
  Retorna los elementos W, H que aproximan A
  al alcanzar una cota de error o superar el maximo
  """
  # Inicializamos las matrices
  m, n = np.shape(A)
  if W is None:
    W = np.random.rand(m, k)

  for i in range(max_iteraciones):
    W, H = rank2(A, W)
    HT, WT = rank2(np.transpose(A), np.transpose(H))
    W, H = np.transpose(WT), np.transpose(HT)
    if (np.linalg.norm(A - np.dot(W, H))) < error: break
  return W, H


def normalizar_descomposicion(W, H):
  """
  Normaliza las columnas de W y pondera respectivamente
  las filas de H para el resultado esperado
  """

  for j in range(2):
    norma = np.linalg.norm(W[:, j])
    W[:, j] = W[:, j]/norma
    H[j, :] = H[j, :]*norma
  return W, H


def calcular_descomposicion(A_matrix, max_iteraciones=15, max_intentos=10):
  """
  Recibe matriz objetivo, cantidad maxima iteraciones e intentos de calcular
  Calcula la descomposición rank-2 de forma reiterativa
  Si el i-esimo intento alcanza la cota
  se retorna la descomposición W, H
  """
  salida, excepcion = False, False
  for i in range(max_intentos):
    try:
      W, H = calculo_NMF(A_matrix, max_iteraciones, k=2, W=None, H=None)
      error = np.linalg.norm(np.dot(W, H) - A_matrix)
      if error < 60:
        salida = True
    except:
      excepcion = True
    else:
      if not excepcion and salida:
        return normalizar_descomposicion(W, H)



def idx_plbs(arreglo):
  """
  Recibe un arreglo de palabras
  Genera un diccionario con los detalles de la palabra
  retornando una lista ordenada según relevancia
  """
  largo = len(arreglo)
  retorno = list({'word': i, 'value': arreglo[i]} for i in range(largo))
  retorno = sorted(retorno, key=lambda x: x['value'], reverse=True)

  for i in range(largo):
    retorno[i]['id'] = i
  return retorno

def generar_arrays(array_N, array_L, array_R):
  """
  Recibe los arreglos para poder dividir
  Retorna los arreglos ordenados según relevancia de sus palabras
  """
  return idx_plbs(array_N), idx_plbs(array_L), idx_plbs(array_R)


def factor_descuento(word, array_L, array_R):
  """
  Recibe los arreglos y la palabra para la cual
  se va a calcular su descuento
  Retorna el descuento de la palabra
  """

  fi_L = next(x for x in array_L if x['word'] == word)
  fi_R = next(x for x in array_R if x['word'] == word)  
  return np.log2(len(array_L) - max(fi_L['id'], fi_R['id']) + 1)


def ganancia_palabra(word, array_N, array_L, array_R):
  """
  Recibe los arreglos y la palabra de la que se quiere obtener su ganancia
  Retorna la ganancia de la palabra
  """
  
  descuento = factor_descuento(word, array_L, array_R)
  elemento = next(x for x in array_N if x['word'] == word)
  return np.log2(len(array_L) - elemento['id'] + 1)/descuento


def ganancias(array_N, array_L, array_R):
  """
  Calcula la ganancia del arreglo
  Retorna el arreglo de las ganancias y ordenada según ganancia
  """
  retorno = list()
  for word in range(len(array_N)):
    gan_actual = ganancia_palabra(word, array_N, array_L, array_R)
    retorno.append({'palabra': word, 'ganancia': gan_actual})
  
  return retorno, sorted(retorno, key=lambda x: x['ganancia'], reverse=True)

def MDCG(gan_array):
  """
  Calculo de MDCG según el array que se entregue
  Retorna el valor de ganancia
  """
  largo = len(gan_array)
  elementos = list(gan_array[i]['ganancia']/np.log2(i+1) for i in range(1, largo))
  return gan_array[0]['ganancia'] + sum(elementos)


def mNDCG(gan_array, gan_sort):
  """
  Calculo del puntaje a través de los arrays listos
  """
  return MDCG(gan_array)/MDCG(gan_sort)


def puntaje(f_N, f_L, f_R):
  """
  Calcula el puntaje de la descomposición NMF actual
  Retorna el valor que nos ayuda a decidir
  """
  gan, gan_sort = ganancias(*generar_arrays(f_N, f_L, f_R))
  return mNDCG(gan, gan_sort)**2


def elem_puntaje(A_matrix, L_matrix, R_matrix):
  """
  Calcula la descomposición NMF de A (nodo)
  y de sus posibles hijos
  Retorna los elementos necesarios para determinar si conviene
  """

  condicion = False
  while not condicion:
    try:
      W, H = calcular_descomposicion(A_matrix)
      WL, HL = calcular_descomposicion(L_matrix)
      WR, HR = calcular_descomposicion(R_matrix)
      condicion = True
    except:
      condicion = False
    else:
      if condicion:
        return W, H, WL, HL, WR, HR

def calculo_puntajes(W, H, WL, HL, WR, HR, i):  
  """
  Calcula el puntaje de los hijos del nodo
  a partir de los 
  """
  X = W[:, i].copy()

  puntaje_N1 = puntaje(X, WL[:, 0], WL[:, 1])
  puntaje_N2 = puntaje(X, WR[:, 0], WR[:, 1])

  return puntaje_N1, puntaje_N2


def puntajes_hijos(A, L, R):
  """
  Genera el calculo del puntaje a partir de los
  elementos necesario a partir del nodo
  """
  # W, H, WL, HL, WR, HR = elem_puntaje(A, L, R)
  return calculo_puntajes(*elem_puntaje(A, L, R), 0)
  

def agregar_columna(A, columna):
  """
  Agrega columna a la matriz A sin importar su contenido
  Retorna la matriz con la columna añadida
  """
  if A is None:
    A = np.zeros((len(columna), 1))
    A[:, 0] = columna
  else:
    A = np.column_stack((A,columna))
  return A


def split_matrix(A_matrix, W, H, columnas):
  """
  Separación de la matriz por contenido
  Retorna la separación en dos matrices
  
  col_docs:
  """
  m, n = np.shape(A_matrix)

  A1, A2 = None, None
  
  retorno_A1 = list()
  retorno_A2 = list()

  for j in range(n):
    if H[0][j] > H [1][j]:
      A1 = agregar_columna(A1, A_matrix[:, j])
      retorno_A1.append(columnas[j])
    else:
      A2 = agregar_columna(A2, A_matrix[:, j])
      retorno_A2.append(columnas[j])

  if A1.shape[1] >= A2.shape[1]:
    return A1, A2, retorno_A1, retorno_A2
  return A2, A1, retorno_A2, retorno_A1
