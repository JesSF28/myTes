import re
import matplotlib.pyplot as plt
import os

def contar_palabras(texto):
    """Cuenta todas las palabras en el texto usando una expresión regular."""
    # \w+ captura secuencias de caracteres alfanuméricos,
    # lo que en la práctica equivale a palabras.
    palabras = re.findall(r'\w+', texto, re.UNICODE)
    return len(palabras)

def contar_oraciones(texto):
    """Cuenta el número de oraciones en el texto.
       Separa usando signos de cierre de oración (punto, ?, !)."""
    # Dividimos por ".!?" y filtramos las cadenas vacías.
    oraciones = re.split(r'[.!?]+', texto)
    oraciones = [s for s in oraciones if s.strip() != '']
    return len(oraciones)

def contar_correcciones(original, corregido):
    """
    Compara palabra por palabra y cuenta las correcciones realizadas.
    Si los textos tienen distinta longitud, suma la diferencia.
    """
    orig_words = original.split()
    corr_words = corregido.split()
    
    # Contar diferencias en las posiciones comunes
    differences = sum(1 for o, c in zip(orig_words, corr_words) if o != c)
    # Sumar los restos si la longitud de las listas es distinta
    differences += abs(len(orig_words) - len(corr_words))
    return differences

def gen_estad(original, corregido):
    """Genera un diccionario con estadísticas del proceso de corrección."""
    total_palabras = contar_palabras(original)
#    total_oraciones = contar_oraciones(original)
    total_correcciones = contar_correcciones(original, corregido)
    
    porcentaje = (total_correcciones / total_palabras * 100) if total_palabras > 0 else 0

    return total_palabras,total_correcciones,round(porcentaje, 2)
#        'total_oraciones': total_oraciones,

def contar_plbr_dicc(texto, diccionario):
  """
  Cuenta cuántas veces aparecen las palabras de un diccionario en un texto.

  Args:
    texto: La cadena de texto a analizar.
    diccionario: Un diccionario con las palabras a contar.

  Returns:
    Un diccionario con el conteo de cada palabra del diccionario en el texto.
  """

  conteo = {}
  for palabra in diccionario:
    conteo[palabra] = texto.lower().count(palabra.lower())
  return conteo

# Ejemplo de uso:
#texto = "Este es un ejemplo de texto. Este texto es para mostrar cómo funciona la función."
#diccionario = {"es", "texto", "función"}
#conteo = contar_palabras_en_diccionario(texto, diccionario)

#print(conteo)

def est_graf(x, y,fn,ruta):
# Datos
#x = ["A", "B", "C"]
#y = [3, 5, 1]
    colores = ["#619cff", "#f8766d", "#00ba38"]
    plt.figure(figsize=(10, 5))

    # Gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(x = x, height = y, color = colores)
    fn1=os.path.join(ruta, fn+"g.jpg")
    plt.savefig(fn1, dpi=100, bbox_inches='tight')
    plt.close()
    return fn1
#     plt.show()