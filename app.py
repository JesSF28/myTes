#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROCESO ALGORITMICO PARA RECUPERACION HISTORICA EFICIENTE GESTION  DOCUMENTAL
UNSA-FIPS-DCC, 2025
JMSF
"""
from flask import Flask, render_template, request

import os
from werkzeug.utils  import secure_filename
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
try:
    import Image
except ImportError:
    from PIL import Image
import prep_gvfc_balloon_1 as prp1
import prep_gvfc_balloon_2 as prp2

DEVELOPMENT_ENV = True

app = Flask(__name__)

app_data = {
    "titulo": "Proceso Algorítmico de Recuperación Eficiente en Gestión Documental Histórica",
    "especial": "Ciencias de la Computacion",
    "name": "Peter's Starter Template for a Flask Web App",
    "description": "Calcula como difusion del gradiente con vectores de un mapa de borde binario o nivel de grises derivado de imagen",
    "autor": "Jesus Martin Silva Fernandez",
    "html_title": "Peter's Starter Template for a Flask Web App",
    "project_name": "TESIS",
    "keywords": "flask, webapp, template, basic",
}


@app.route("/")
def index():
    return render_template("index.html", app_data=app_data)


@app.route("/inicio")
def about():
    return render_template("inicio.html", app_data=app_data)


@app.route("/documento")
def service():
    return render_template("jsf_tesis_tx.html", app_data=app_data)


@app.route("/codigo")
def contact():
    return render_template("codigo.html", app_data=app_data)

@app.route("/recursos")
def recursos():
    return render_template("jsf_tesis_rec.html", app_data=app_data)

@app.route("/contacto")
def contacto():
    return render_template("contacto.html", app_data=app_data)

@app.route("/uploader", methods=['POST'])
def uploader():
    if request.method == "POST":
        f = request.files['archivo']
#        f = request.FILES['ufile'].file.name
        filename = secure_filename(f.filename)
        filename1 = os.path.join('assets', filename)
        print("filename: ",filename)
        print("filename1: ",filename1)
        imagen = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)

        su = request.form.getlist("act")
        for s in su:
#            print(int(s))
            if int(s)==1:
                ocrt(filename,imagen,"0")          # ocr
            if int(s)==2:       
                img = prp1.preproc(imagen)         # preproceso 1
                ocrt(filename,img,"1")             # ocr
            if int(s)==3:       
                img = prp2.preproc2(imagen)         # preproceso 2
                ocrt(filename,img,"2")             # ocr
#        mens=mult()
#        graf()
    return render_template("codigo.html", app_data=app_data)

def p_():
    f = request.files['archivo']
    filename = secure_filename(f.filename)
    filename1 = os.path.join('assets', filename)
#                filename2 = os.path.join('models', filename)
#        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    imagen = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
#        cv2.imwrite(os.path.join(filename2),imagen)
    height, width = imagen.shape
    nu=50
    plt.subplot(221)
    plt.imshow(imagen[::-1], cmap="gray")       # 1
    plt.axis([-nu, width+nu, -nu, height+nu])
    plt.title("Imagen Original 1")
    plt.show()
    return filename,imagen

def ocrt(fn,img,nu):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    txt = pytesseract.image_to_string(img)          # Genera texto
    n,e = os.path.splitext(fn)                      # separa nombre arch
    t = open (os.path.join('models',n+nu),'w')
    t.write(txt)
    t.close()

if __name__ == "__main__":
    app.run(debug=DEVELOPMENT_ENV)
