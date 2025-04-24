import os
import csv
import cv2
import math
import numpy as np
import random
import matplotlib.pyplot as plt

from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_fill_holes
from PIL import Image

def calculate_gvf(image, mu, gvf_iterations, kappa_1 = 0):
	fmin = np.min(image)
	fmax = np.max(image)
	image_normalized = (image - fmin) / (fmax - fmin)
	dy, dx = np.gradient(image_normalized.astype(float))
	u = dx.copy()
	v = dy.copy()
	norm_square = dx ** 2 + dy ** 2
	magnitude = np.sqrt(norm_square) + 1e-8
	
	for i in range(gvf_iterations):
		u_xx = laplace(u)
		v_yy = laplace(v)
		u = u + (mu * u_xx) - (norm_square * (u - dx)) + kappa_1 * dx / magnitude
		v = v + (mu * v_yy) - (norm_square * (v - dy)) + kappa_1 * dy / magnitude

#	if os.path.exists('campo_fuerzas_gvf.csv'): os.remove('campo_fuerzas_gvf.csv')
#	with open("campo_fuerzas_gvf.csv", mode="w", newline="") as file:
#		writer = csv.writer(file)
#		writer.writerow(["Fuerzas GVF U"])
#		for i in range(height): writer.writerow([round(u[i, j], 3) for j in range(width)])
#		writer.writerow(["Fuerzas GVF V"])
#		for i in range(height): writer.writerow([round(v[i, j], 3) for j in range(width)])
#		writer.writerow(["Magnitud GVF"])
#		for i in range(height): writer.writerow([round(np.sqrt(u[i, j]**2+ v[i, j]**2), 3) for j in range(width)])
	return [u, v]

def apply_dilate(image_matrix, kernel_size=(3,3), iterations=1):
	"""
	Aplica la operación de dilatación a una imagen dada.

	:param image_matrix: Matriz de la imagen (en escala de grises o binaria).
	:param kernel_size: Tamaño del kernel para la dilatación (por defecto 3x3).
	:param iterations: Número de iteraciones de dilatación (por defecto 1).
	:return: Imagen dilatada como matriz.
	"""
	kernel = np.ones(kernel_size, np.uint8)  # Se crea un kernel de unos
	dilated_image = cv2.dilate(image_matrix, kernel, iterations=iterations)
	return dilated_image

def fill_concavities(binary_image, kernel_size=(5,5), iterations=2):
	"""
	Rellena concavidades en una imagen binaria utilizando morfología de cierre.
	
	:param image_path: Ruta de la imagen en escala de grises o binaria.
	:param kernel_size: Tamaño del kernel para el cierre morfológico.
	:param iterations: Número de iteraciones para mejorar el resultado.
	:return: Imagen con concavidades rellenadas.
	"""
	kernel = np.ones(kernel_size, np.uint8)  # Crear un kernel
	closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

	return closed
def process_matrix(u, v, kernel_size_1=(2,2), iterations_1=2, kernel_size_2=(2,2), iterations_2=3):
	steps = []

	# Magnitude GVF
	result = np.sqrt(u**2+v**2)
	steps.append(result)

	# Rellenado GVF
	result = fill_concavities(result, kernel_size_1, iterations_1)
	steps.append(result)

	# Dilatacion GVF
	result = apply_dilate(result, kernel_size_2, iterations_2)
	steps.append(result)
	
	result = result * 255
	_, binaria_result = cv2.threshold(result.astype(np.uint8), 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	imagen_binaria_result = (binaria_result // 255).astype(np.uint8)
	imagen_binaria_result = (imagen_binaria_result==0).astype(np.uint8)

	return imagen_binaria_result, steps

def preproc2(image,fn,ruta):
#if __name__ == "__main__":
	# Cargar la imagen
#	filename = "car_4.jpg"
#	filename = "hist_0.jpg"
#	image = cv2.imread(os.path.join("assets", filename), cv2.IMREAD_GRAYSCALE)
	# image = gaussian_filter(image, sigma=1)
	height, width = image.shape

	# Variables GVF-Snake
	mu = 0.2                # Mu - densidad GVF
	gvf_iterations = 1      # Iteraciones para suavizar GVF
	kappa_1 = 0             # Coeficiente Balloon
	snake_iterations = 200  # Iteraciones para adaptar snake
	
	#Funcion Binaria
	_, binaria_otsu = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	imagen_binaria = (binaria_otsu // 255).astype(np.uint8)
	imagen_binaria = imagen_binaria[::-1]

	#Calculo GVF
	gvf_u, gvf_v = calculate_gvf(imagen_binaria, mu, gvf_iterations,kappa_1)

	result_matrix, steps = process_matrix(gvf_u, gvf_v, kernel_size_1=(2,1), iterations_1=1, kernel_size_2=(1,1), iterations_2=1)

	result_matrix = np.bitwise_and(result_matrix,imagen_binaria)

#	if os.path.exists('imagen_binaria.csv'): os.remove('imagen_binaria.csv')
#	with open("imagen_binaria.csv", mode="w", newline="") as file:
#		writer = csv.writer(file)
#		for i in range(height): writer.writerow([round(imagen_binaria[i, j], 3) for j in range(width)])

	# Mostrar la imagen original y el resultado del GVF
	nu=50
	fig = plt.figure(figsize=(10, 5))
#	fig.canvas.manager.set_window_title("TESIS")
#	plt.subplot(221)
#	plt.imshow(image[::-1], cmap="gray")       # 1
#	plt.axis([-nu, width+nu, -nu, height+nu])
#	plt.title("Imagen Original 1")
#	plt.subplot(222)
#	plt.imshow(steps[2], cmap="gray")          # 2
#	plt.axis([-nu, width+nu, -nu, height+nu])
#	plt.title("Magnitud GVF 2")
#	plt.subplot(223)
	plt.imshow(imagen_binaria, cmap="gray")    # 3
	plt.axis([-nu, width+nu, -nu, height+nu])
	plt.gca().set_axis_off()
	fn1=os.path.join(ruta,fn+".jpg")
	plt.savefig(fn1, dpi=100, bbox_inches='tight', pad_inches=0)
	plt.close()
#    fn1=os.path.join("static","hist_02.jpg")
#    plt.gca().set_axis_off()

#	plt.title("Imagen Binaria 3")
#	plt.subplot(224)
#	plt.imshow(result_matrix, cmap="gray")     # 4
#	plt.axis([-nu, width+nu, -nu, height+nu])
#	plt.title("GVF final 4")

#	plt.show()
	return Image.open(fn1),fn1

def preproc3(image,fn,ruta):
#if __name__ == "__main__":
	# Cargar la imagen
#	filename = "car_4.jpg"
#	filename = "hist_0.jpg"
#	image = cv2.imread(os.path.join("assets", filename), cv2.IMREAD_GRAYSCALE)
	# image = gaussian_filter(image, sigma=1)
	height, width = image.shape

	# Variables GVF-Snake
	mu = 0.2                # Mu - densidad GVF
	gvf_iterations = 1      # Iteraciones para suavizar GVF
	kappa_1 = 0             # Coeficiente Balloon
	snake_iterations = 200  # Iteraciones para adaptar snake
	
	#Funcion Binaria
	_, binaria_otsu = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	imagen_binaria = (binaria_otsu // 255).astype(np.uint8)
	imagen_binaria = imagen_binaria[::-1]

	#Calculo GVF
	gvf_u, gvf_v = calculate_gvf(imagen_binaria, mu, gvf_iterations,kappa_1)

	result_matrix, steps = process_matrix(gvf_u, gvf_v, kernel_size_1=(2,1), iterations_1=1, kernel_size_2=(1,1), iterations_2=1)

	result_matrix = np.bitwise_and(result_matrix,imagen_binaria)

#	if os.path.exists('imagen_binaria.csv'): os.remove('imagen_binaria.csv')
#	with open("imagen_binaria.csv", mode="w", newline="") as file:
#		writer = csv.writer(file)
#		for i in range(height): writer.writerow([round(imagen_binaria[i, j], 3) for j in range(width)])

	# Mostrar la imagen original y el resultado del GVF
	nu=50
	fig = plt.figure(figsize=(10, 5))
#	fig.canvas.manager.set_window_title("TESIS")
#	plt.subplot(221)
#	plt.imshow(image[::-1], cmap="gray")       # 1
#	plt.axis([-nu, width+nu, -nu, height+nu])
#	plt.title("Imagen Original 1")
#	plt.subplot(222)
#	plt.imshow(steps[2], cmap="gray")          # 2
#	plt.axis([-nu, width+nu, -nu, height+nu])
#	plt.title("Magnitud GVF 2")
#	plt.subplot(223)
#	plt.imshow(imagen_binaria, cmap="gray")    # 3
#	plt.axis([-nu, width+nu, -nu, height+nu])

#    fn1=os.path.join("static","hist_02.jpg")
#    plt.gca().set_axis_off()

#	plt.title("Imagen Binaria 3")
#	plt.subplot(224)
	plt.imshow(result_matrix, cmap="gray")     # 4
	plt.axis([-nu, width+nu, -nu, height+nu])
	plt.gca().set_axis_off()
	fn1=os.path.join(ruta,fn+".jpg")
	plt.savefig(fn1, dpi=100, bbox_inches='tight', pad_inches=0)
	plt.close()

#	plt.title("GVF final 4")

#	plt.show()
	return Image.open(fn1),fn1
