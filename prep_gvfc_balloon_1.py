# TESIS: Doctorado Jesús Martín Silva Fernández
import os

import matplotlib
matplotlib.use('agg')
import cv2
import csv
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import acos
from scipy.ndimage import laplace as del2
from skimage.segmentation import active_contour

from flask import Flask, render_template, request
from werkzeug.utils  import secure_filename
from PIL import Image

def calculate_gvf(image, mu, gvf_iterations):
	# Obtener Alto y Ancho
	[m, n] = image.shape
	
	# % Normaliza f al rango [0,1]
	fmin = np.min(image)
	fmax = np.max(image)
	image_normalized = (image - fmin) / (fmax - fmin)

	# Calcula el gradiente del mapa de bordes
	dy, dx = np.gradient(image_normalized.astype(float))

	# Magnitud al cuadrado del campo de gradiente
	u = dx.copy()
	v = dy.copy()

	norm_square = dx ** 2 + dy ** 2
	
	# Resuelve iterativamente el GVF
	for i in range(gvf_iterations):
		u_xx = del2(u)
		v_yy = del2(v)

		# Fuerza GVF: Atrae el campo hacia los bordes
		gvf_force_u = (mu * (u_xx)) + (norm_square * (u - dx))
		gvf_force_v = (mu * (v_yy)) + (norm_square * (v - dy))

		# Calcular las fuerzas externas
		u = u + gvf_force_u
		v = v + gvf_force_v

	test_u = (u)
	test_v = (v)
	return [u, v, test_u, test_v]


def snake_contour_manual(snake, alpha, beta, u, v, gamma, kappa_1, kappa_2, kappa_3, iterations=50):
	snake_iterations = []
	for _ in range(iterations):
		# Segunda y cuarta derivada (suavidad y rigidez)
		X_prev = np.roll(snake, 1, axis=0)
		X_next = np.roll(snake, -1, axis=0)
		X_ss = X_next - 2 * snake + X_prev
		X_prev2 = np.roll(snake, 2, axis=0)
		X_next2 = np.roll(snake, -2, axis=0)
		X_ssss = X_next2 - 4*X_next + 6*snake - 4*X_prev + X_prev2

		internal_force_x = alpha * X_ss[:, 0] - beta * X_ssss[:, 0]
		internal_force_y = alpha * X_ss[:, 1] - beta * X_ssss[:, 1]
		
		T = X_next - X_prev
		T = T / np.linalg.norm(T, axis=1, keepdims=True)

		# Obtener los límites de la matriz u y v
		# Suponiendo que u y v tienen el mismo tamaño
		height, width = u.shape

		# Asegurar que los valores de snake[:, 0] y snake[:, 1] estén dentro de los límites
		snake[:, 0] = np.clip(snake[:, 0], 0, width - 1)
		snake[:, 1] = np.clip(snake[:, 1], 0, height - 1)

		# Obtener la fuerza GVF en cada punto del snake
		gvf_force_x = u[snake[:, 1].astype(int), snake[:, 0].astype(int)]
		gvf_force_y = v[snake[:, 1].astype(int), snake[:, 0].astype(int)]

		# Obtener la fuerza normal en cada punto del snake
		normal_force_x = -1 * T[:, 1]
		normal_force_y = T[:, 0].copy()

		# Obtener la magnitud de la fuerza normal
		normal_magnitude = np.sqrt(normal_force_x**2 + normal_force_y**2)
		normal_magnitude[normal_magnitude == 0] = 1  # Evitar divisiones por cero
		
		# Normalizar la fuerza normal
		normal_force_x /= normal_magnitude
		normal_force_y /= normal_magnitude

		sign_normal_force_x = np.where((normal_force_x >= 0) == (gvf_force_x >= 0), 1, 1)
		sign_normal_force_y = np.where((normal_force_y >= 0) == (gvf_force_y >= 0), 1, 1)

		# Obtener la fuerza tangente en cada punto del snake
		tangent_force_x = T[:, 0].copy()
		tangent_force_y = T[:, 1].copy()

		# Obtener la magnitud de la fuerza tangente
		tangent_magnitude = np.sqrt(tangent_force_x**2 + tangent_force_y**2)
		tangent_magnitude[tangent_magnitude == 0] = 1  # Evitar divisiones por cero

		# Normalizar la fuerza tangente
		tangent_force_x /= tangent_magnitude
		tangent_force_y /= tangent_magnitude

		# Actualizar posición del Snake
		snake_iterations.append(snake.copy())
		snake[:, 0] = snake[:, 0] + gamma * internal_force_x + kappa_1 * gvf_force_x + kappa_2 * sign_normal_force_x * normal_force_x + kappa_3 * tangent_force_x
		snake[:, 1] = snake[:, 1] + gamma * internal_force_y + kappa_1 * gvf_force_y + kappa_2 * sign_normal_force_y * normal_force_y + kappa_3 * tangent_force_y 

	return snake, snake_iterations

def BoundMirrorEnsure(A):
	B = A.copy()
	B[[0, -1], :] = B[[1, -2], :]
	B[:, [0, -1]] = B[:, [1, -2]]
	return B

def BoundMirrorExpand(A):
	B = np.pad(A, 1, mode='edge')
	return B

def BoundMirrorShrink(A):
	return A[1:-1, 1:-1]

def preproc(image,fn,ruta):
#if __name__ == "__main__":
	# Cargar la imagen
#	filename = "car_2.bmp"
#	filename = "car_3.bmp"
#	filename = "car_4.bmp"

#    f = request.files['archivo']
#    filename = secure_filename(f.filename)
#    filename1 = os.path.join('assets', filename)
#    image = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)

#	filename = "hist_0.jpg"
#	image = cv2.imread(os.path.join("assets", filename), cv2.IMREAD_GRAYSCALE)

	# Variables de GVF
    mu = 0.2 				# Mu - densidad GVF
    gvf_iterations = 1   	# Iteraciones para suavizar GVF

	#Calculo GVF
    u, v, test_u, test_v = calculate_gvf(image, mu, gvf_iterations)

	# Variables de Snake
    alpha = 0.2 			# Coeficiente suavidad - Fuerza Interna
    beta = 0 	 			# Coeficiente rigidez - Fuerza Interna
    gamma = 1.0 			# Actualizacion en tiempo
    kappa_1 = 2 			# Coeficiente GVF
    kappa_2 = 0.1 			# Coeficiente Balloon
    kappa_3 = 0.1			# Coeficiente Tangencial
    snake_iterations = 150 	# Iteraciones para adaptar snake
	
	# Preparacion Snake
    height, width = image.shape
    radius_x = (width-2)/2
    radius_y = (height-2)/2
    t = np.linspace(0, 2 * np.pi, 100)
    x0, y0 = image.shape[1] // 2, image.shape[0] // 2
    snake = np.array([x0 + radius_x * np.cos(t), y0 + radius_y * np.sin(t)]).T
	
	# Definir los puntos del Rectángulo
	# margin = 1  # Margen para evitar tocar los bordes exactos de la imagen
	# top = [(x, margin) for x in range(margin, width - margin)]
	# right = [(width - margin, y) for y in range(margin, height - margin)]
	# bottom = [(x, height - margin) for x in range(width - margin, margin, -1)]
	# left = [(margin, y) for y in range(height - margin, margin, -1)]

	# snake = np.array(top + right + bottom + left)
	
    x = snake[:, 0]
    y = snake[:, 1]

	# Calculo Snake
    initial_snake = snake.copy()
    snake, snake_iter = snake_contour_manual(snake, alpha, beta, u, v, gamma, kappa_1, kappa_2, kappa_3, snake_iterations)
    final_snake = snake.copy()

	# Guardar valores de snake en el CSV
	#with open('snake_values.csv', 'w', newline='') as csvfile:
	#	fieldnames = ['iteration', 'snake_initial', 'snake_final']
	#	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
	#	for i in range(len(snake)):
	#		writer.writerow({
	#			'iteration': i,
	#			'snake_initial': initial_snake[i].tolist(),
	#			'snake_final': final_snake[i].tolist()
	#		})

	# Mostrar la imagen original y el resultado del GVF
#    fig = plt.figure(figsize=(10, 5))
    plt.figure(figsize=(10, 5))
#    fig.canvas.manager.set_window_title('TESIS')

#    plt.subplot(131)
#    plt.quiver(test_u[::-1],-test_v[::-1], angles="xy")
#    img1 = plt.quiver(test_u[::-1],-test_v[::-1], angles="xy")
#    img1
#    plt.title("Resultado 1: Campo GVF")

#    plt.subplot(132)
#    plt.imshow(np.sqrt(u**2 + v**2), cmap="gray")
#    img2 = plt.imshow(np.sqrt(u**2 + v**2), cmap="gray_r")
    plt.imshow(np.sqrt(u**2 + v**2), cmap="gray_r")
#    plt.savefig('testplot.png')
    fn1=os.path.join(ruta, fn+".jpg")
#    plt.figure(figsize=(2, 1))
    plt.gca().set_axis_off()
    plt.savefig(fn1, dpi=100, bbox_inches='tight')
    plt.close()

#    img2
#    plt.title("Resultado 2: Magnitud Campo GVF")

#    ax = fig.add_subplot(133)
#    ax.set_title("Snake F")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    ax.axis("equal")
#    ax.grid()

	# Función para actualizar la animación
#    def update(frame):
#        ax.clear()
#        ax.set_title("Snake Final")
#        ax.set_xlabel("x")
#        ax.set_ylabel("y")
#        ax.axis("equal")
#        ax.grid()
		
#        current_iteration = snake_iter[frame]
#        ax.imshow(image, cmap="gray")
#        ax.plot(current_iteration[:, 0], current_iteration[:, 1], '-r', linewidth=2, label="Serpiente")
#        ax.scatter(current_iteration[:, 0], current_iteration[:, 1], color="r", s=5, label="Puntos")
#        ax.legend()

	# Crear la animación
#    ani = animation.FuncAnimation(fig, update, frames=len(snake_iter), interval=100, repeat=False)

#    plt.show()
	# ani.save('test.mp4')
    
    return Image.open(fn1),fn1