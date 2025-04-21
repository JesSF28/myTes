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
    return render_template("codigo1.html", app_data=app_data)

@app.route("/recursos")
def recursos():
    return render_template("jsf_tesis_rec.html", app_data=app_data)

@app.route("/contacto")
def contacto():
    return render_template("contacto.html", app_data=app_data)

@app.route("/uploader", methods=['POST'])
def uploader():
    if request.method == "POST":
        ruta="static"
        f = request.files['archivo']
#        f = request.FILES['ufile'].file.name
        filename = secure_filename(f.filename)           # Nom Arch
        filename1 = os.path.join(ruta, filename)     # Ruta+nom Arch
        fn,ex = os.path.splitext(filename)               # Nom Arch, Ext
#        print("filename: ",filename)
#        print("filename1: ",filename1)
        imagen = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)

        su = request.form.getlist("act")
        mtxt = ["","","",""]
        mimg = ["","","",""]
        mnum = []
        for s in su:
            print(int(s))
            if int(s)==1:
                nu="0"
                mtxt.insert(0,ocrt(fn,imagen,nu))          # imagen original, ocr 0
                mimg.insert(0,filename1)
                mnum.insert(0,0)
            if int(s)==2:       
                nu="1"
                img,fn1 = prp1.preproc(imagen,fn+nu,ruta)               # imagen de preproceso 1
                mtxt.insert(1,ocrt(fn,img,nu))          # ocr 1
                mimg.insert(1,fn1)
                mnum.insert(1,1)
            if int(s)==3:       
                nu="2"
                img = prp2.preproc2(imagen)              # preproceso 2
                mtxt.insert(2,ocrt(fn,imagen,nu))          # ocr 2
                mnum.insert(2,2)
            print("mtxt: ",mtxt)
            print("mimg: ",mimg)
            print("mnum: ",mnum)
#        mens=mult()
#        graf()
#    return render_template("codigo2.html", app_data=app_data, filename1=filename1, txt=mtxt)
    return render_template("codigo2.html", app_data=app_data, fn1=mimg, txt=mtxt,nu=mnum)

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
#    n,e = os.path.splitext(fn)                      # separa nombre arch
    tx = open (os.path.join('static',fn+nu+".txt"),'w')
    tx.write(txt)
    tx.close()
    return txt

if __name__ == "__main__":
    app.run(debug=DEVELOPMENT_ENV)
