# TESIS: Doctorado Jesús Martín Silva Fernández
import nltk
from collections import Counter
import re
import os

# Descarga del corpus de Cess_esp
nltk.download('cess_esp')
from nltk.corpus import cess_esp

def palabras(texto):
    return re.findall(r'\w+', texto.lower())

# Unión de todos los textos del corpus de Cess_esp en un solo string
texto_corpus = ' '.join([' '.join(sent) for sent in cess_esp.sents()])

# Construcción de un diccionario de frecuencias de palabras
WORD_COUNTS = Counter(palabras(texto_corpus))

def probabilidad(palabra, N=sum(WORD_COUNTS.values())): 
    "Probabilidad de la palabra."
    return WORD_COUNTS[palabra] / N

def correcciones(palabra): 
    "Generar todas las correcciones posibles para la palabra."
    return (conocido([palabra]) or conocido(ediciones1(palabra)) or conocido(ediciones2(palabra)) or [palabra])

def conocido(palabras): 
    "Filtrar las palabras que están en el diccionario."
    return set(w for w in palabras if w in WORD_COUNTS)

def ediciones1(palabra):
    "Generar ediciones a una distancia de la palabra."
    letras    = 'abcdefghijklmnopqrstuvwxyzáéíóúñ'
    divisiones = [(palabra[:i], palabra[i:]) for i in range(len(palabra) + 1)]
    # [('', 'gato'), ('g', 'ato'), ('ga', 'to'), ('gat', 'o'), ('gato', '')]
    eliminaciones    = [L + R[1:] for L, R in divisiones if R]
    # eliminaciones = [L + R[1:] for L, R in divisiones if R]
    transposiciones = [L + R[1] + R[0] + R[2:] for L, R in divisiones if len(R)>1]
    # ['agto', 'gtao', 'gaot']
    reemplazos    = [L + c + R[1:] for L, R in divisiones if R for c in letras]
    # ['aato', 'bato', ..., 'zato', 'áato', 'éato', ..., 'ñato', 'gato', 'gcto', ..., 'gzto', 'gáto', 'géto', ..., 'gñto', ...]
    inserciones   = [L + c + R for L, R in divisiones for c in letras]
    # ['agato', 'bgato', ..., 'zgato', 'ágato', 'égato', ..., 'ñgato', 'gaato', 'gbato', ..., 'gzato', 'gáato', 'géato', ..., 'gñato', ...]
    return set(eliminaciones + transposiciones + reemplazos + inserciones)

def ediciones2(palabra): 
    "Generar ediciones a dos distancias de la palabra."
    return (e2 for e1 in ediciones1(palabra) for e2 in ediciones1(e1))

def corregir(palabra): 
    "Corregir la palabra mal escrita."
    return max(correcciones(palabra), key=probabilidad)

# Ejemplo de uso
#texto_1 = ['La siscripcion es de',
#    'un peso mesual, y un',
#    'real el llúmero suelto:',
#    'para iino y otro ocur-',
#    'rase, si se cjuiere, a la',
#    'Imprenta y Litiigrafia',
#    'de sii redacción, calle',
#    'de Plateros No 216']
#for palabras_t in texto_1:
#    print(f'Origen   : {palabras_t}')
#    palabras_texto = palabras_t.split()
#    texto_corregido = ' '.join(corregir(palabra) for palabra in palabras_texto)
#    print(f'Corregido: {texto_corregido}')

def palabr(ruta,fn):
#ruta="static"
#fn="hist_02"
    fn1=os.path.join(ruta, fn+".txt")
    fn2=os.path.join(ruta, fn+"1.txt")
    arch_1 = open(fn1,"r", encoding="utf-8")
    arch_2 = open(fn2,"w", encoding="utf-8")

#with open(fn1, "r") as archivo:
    with arch_1 as archivo:
        contenido = archivo.read()
#        print(contenido)
        palabras_t = contenido.split()
        t_corregido = ' '.join(corregir(palabra) for palabra in palabras_t)
#        print(f'Corregido: {t_corregido}')
        arch_2.write(t_corregido)
    arch_1.close()
    arch_2.close()

    return t_corregido

#texto_incompleto = "Esto es un ejmpl de palabas correidas por algorimo."
#texto_incompleto = "Ejmplo de txto con plabas incmpletas y mal escritas."
#texto_incompleto = "Estn muhas palbras mal escitas lo que no ayua su signficado"

#palabras_texto = texto_incompleto.split()
#texto_corregido = ' '.join(corregir(palabra) for palabra in palabras_texto)
#print(texto_incompleto)
#print(texto_corregido)
