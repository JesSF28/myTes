import sys
import os
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def ChanVeseSegmentation(image,fn1,ruta1,multiple,num):
    # Cargar la imagen desde la ruta especificada
#    image = cv2.imread(filepath, 1)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convierte la imagen a escala de grises
    img = np.array(image2, dtype=np.float64)         # Convierte la imagen a un arreglo de tipo float64

    # Inicializar la función de nivel (Φ) para el método de conjuntos de nivel
    IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype) # Inicializa con valores positivos
    IniLSF[30:80, 30:80] = -1                                 # Define una región inicial con valores negativos
    IniLSF = -IniLSF                                          # Invierte la función de nivel

    # Convertir la imagen cargada a formato RGB para visualización
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    height, width = image.shape

    # Parámetros de evolución
    # mu = 1                 # Peso del término de suavidad
    # nu = 0.003 * 255 * 255 # Peso del término de longitud
    # num = 20               # Número de iteraciones
    # epison = 1             # Parámetro de regularización para Heaviside
    # step = 0.1             # Paso de evolución
    # LSF = IniLSF           # Inicializar la función de nivel

    # Parámetros de evolución
    mu = 1                 # Peso del término de suavidad
    nu = 0.003 * 255 * 255 # Peso del término de longitud
    # num = 20               # Número de iteraciones
    epison = 1             # Parámetro de regularización para Heaviside
    step = 0.1             # Paso de evolución
    LSF = IniLSF           # Inicializar la función de nivel


    # Evolución iterativa de la función de nivel
    info=[]
    info = [LSF]
    for i in range(1, num):
        LSF = CV(LSF, img, mu, nu, epison, step, False) # Actualizar LSF
        info.append(LSF) # Almacenar estado de la evolución

    # Devolver solo la última iteración si no se requiere almacenar todas
    if not multiple:
        info = [info[-1]]

#    plt.figure(figsize=(10, 5))
    n=0
    plt.figure()
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.contour(info[0], [0], colors="r", linewidths=2)
    plt.draw()
#    plt.axis([-n, width, -n, height])
    plt.gca().set_axis_off()
#    plt.show(block=False)
    current_filename = f"{fn1}.jpg"
    fn1=os.path.join(ruta1, current_filename)
    plt.savefig(fn1)
#        plt.cla()
    plt.close()        
#    exh_img(info)
#    return
#    ac_img,fn1=graba_img(image,info,fn,ruta1,n)
#    print("arch chan_ves",ac_img[-1])
#    print("ruta chan_ves",fn1)
#    return Image.open(os.path.join(ruta1, ac_img[-1])),fn1
    return Image.open(fn1),fn1

def graba_img(image,info,fn,ruta1,n):
#    filename = os.path.basename(filepath)
#    print(fn)
    results = []
    for i in range(len(info)):
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.contour(info[i], [0], colors="r", linewidths=2)
        plt.draw()
        plt.show(block=False)

        current_filename = f"{fn+n}{i + 1}.jpg"
        fn1=os.path.join(ruta1, current_filename)
        plt.savefig(fn1)
        results.append(current_filename)

#        plt.cla()
        plt.close()        
    return results,fn1

# Función auxiliar para operaciones matemáticas sobre matrices
def mat_math(intput, str,img):
    output = intput
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if str == "atan":
                output[i, j] = math.atan(intput[i, j])
            if str == "sqrt":
                output[i, j] = math.sqrt(intput[i, j])
    return output

# Método de evolución del contorno utilizando la ecuación de Chan-Vese
def CV(LSF, img, mu, nu, epison, step, show=False):
    # Calcular la derivada regularizada y la función Heaviside suavizada
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    Hea = 0.5 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan",img))

    # Calcular gradiente de la función de nivel
    Iy, Ix = np.gradient(LSF)
    s = mat_math(Ix * Ix + Iy * Iy, "sqrt",img) # Magnitud del gradiente
    Nx = Ix / (s + 0.000001)                # Componente X del vector normalizado
    Ny = Iy / (s + 0.000001)                # Componente Y del vector normalizado
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy # Curvatura media

    # Calcular la longitud del contorno
    Length = nu * Drc * cur

    # Calcular término de penalización
    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu * (Lap - cur)

    # Calcular términos de energía de región
    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum() # Promedio dentro del contorno
    C2 = s2.sum() / s3.sum()  # Promedio fuera del contorno
    CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

    # Actualizar la función de nivel
    LSF = LSF + step * (Length + Penalty + CVterm)

    if show:
        plt.imshow(s, cmap="gray"), plt.show()
    return LSF

def exh_img(image):

#    print("Image shape:", image.shape)

    # Access and print pixel values
#    for y in range(image.shape[0]):
#        for x in range(image.shape[1]):
            # Access pixel values at (x, y)
#            pixel = image[y, x]

            # Print pixel values
 #           print(f"Pixel at ({x}, {y}): {pixel}")

    # Display the image (optional)
 #   cv2.imshow('Image', image)
 #   cv2.waitKey(0)
 #   cv2.destroyAllWindows()
    plt.figure(1)
 #   plt.figure(figsize=(3,3))
 #   plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(image)
    plt.show()
  
    return
    