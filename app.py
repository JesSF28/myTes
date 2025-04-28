# TESIS: Doctorado Jesús Martín Silva Fernández
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROCESO ALGORITMICO PARA RECUPERACION HISTORICA EFICIENTE GESTION  DOCUMENTAL
UNSA-FIPS-DCC, 2025
JMSF
"""
filename=""
pr=0
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
import post_pnorv_1 as postp1
#import prep_balloon_final as prp2
#import prep_gvfc_balloon_2 as prp2

DEVELOPMENT_ENV = True

app = Flask(__name__)

app_data = {
    "titulo": "Proceso Algorítmico de Recuperación Eficiente en Gestión Documental Histórica",
    "especial": "Ciencias de la Computacion",
    "name": "Peter's Starter Template for a Flask Web App",
    "descripcion": "Proceso documental histórico:",
    "referencia": "Referencia Bibliográfica:",
    "presenta": "Presentación:",
    "autor": "Jesus Martin Silva Fernandez",
    "html_title": "TESIS: Jesus Martin Silva Fernandez",
    "project_name": "TESIS",
    "keywords": "flask, webapp, template, basic",
}

@app.route("/")
def index():
    return render_template("index.html", app_data=app_data)

@app.route("/resumen")
def about():
    return render_template("resumen.html", app_data=app_data)

@app.route("/presenta")
def presenta():
    return render_template("presenta.html", app_data=app_data)

@app.route("/documento")
def service():
    return render_template("jsf_tesis_tx.html", app_data=app_data)

@app.route("/codigo")
def codigo():
    global filename
    return render_template("codigo1.html", app_data=app_data, fn=filename)

@app.route("/recursos")
def recursos():
    return render_template("jsf_tesis_rec.html", app_data=app_data)

@app.route("/referencia")
def referencia():
    return render_template("referencia.html", app_data=app_data)

@app.route("/contacto")
def contacto():
    return render_template("contacto.html", app_data=app_data)

@app.route("/uploader", methods=['POST'])
def uploader():
    global filename,pr
    if request.method == "POST":
        oimg = request.form.get('otra')
        if oimg=='Otra Imagen':
            filename=""
            return render_template("codigo1.html", app_data=app_data, fn=filename)
        ruta="static"
        if filename=="":
            f = request.files['archivo']
#        f = request.FILES['ufile'].file.name
            filename = secure_filename(f.filename)       # Nom Arch
            if filename=="":
               return render_template("codigo1.html", app_data=app_data, fn=filename)
                
        filename1 = os.path.join(ruta, filename)         # Ruta+nom Arch
        fn,ex = os.path.splitext(filename)               # Nom Arch, Ext
        print("filename: ",filename)
        print("filename1: ",filename1)
        print("Long filename: ",len(filename))
        imagen = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)

        su = request.form.getlist("act")
        print("su:",su)
        mtxt = []
        mimg = []
        mnum = []
        mpst = []
        n=0
        ti=[]
        for s in su:
            print(int(s))
            if int(s)==0:
                nu="0"
                mimg.append(filename1)                        # Nomb imagen original
                mtxt.append(ocrt(fn,imagen,nu))               # Agrega texto de ocr 0
                mpst.append(postp1.palabr(ruta,fn+nu))        # Agrega texto corregido
                mnum.append(n)                                # Agrega Num opcion
                ti.append("Imagen Original")                  # Agrega titulo Imagen
            if int(s)==1:       
                nu="1"
                img,fn1 = prp1.preproc(imagen,fn+nu,ruta)      
                mimg.append(fn1)                               # preproceso 1
                mtxt.append(ocrt(fn,img,nu))                   # ocr 1
                mpst.append(postp1.palabr(ruta,fn+nu))         # Txt corregido 1
                mnum.append(n)
                ti.append("Imagen Algoritmo 1")
            if int(s)==2:       
                nu="2"
                img,fn2 = prp2.preproc2(imagen,fn,nu,ruta)     
                mimg.append(fn2)                               # preproceso 2
                mtxt.append(ocrt(fn,img,nu))                   # ocr 2
                mpst.append(postp1.palabr(ruta,fn+nu))         # Txt corregido 2
                mnum.append(n)
                ti.append("Imagen Algoritmo 2")
            if int(s)==3:       
                nu="3"
                img,fn2 = prp2.preproc2(imagen,fn,nu,ruta)     # preproceso 3
                mimg.append(fn2)
                mtxt.append(ocrt(fn,img,nu))                   # ocr 3
                mpst.append(postp1.palabr(ruta,fn+nu))         # Txt corregido 3
                mnum.append(n)
                ti.append("Imagen Algoritmo 3")
            n += 1
            print("mtxt: ",mtxt)
            print("mimg: ",mimg)
            print("mnum: ",mnum)
            print("mpst: ",mpst)
    pr+=1    
    print("prueba: ",pr)
#        mens=mult()
#        graf()
#    return render_template("codigo2.html", app_data=app_data, filename1=filename1, txt=mtxt)
    return render_template("codigo2.html", app_data=app_data, fn1=mimg, txt=mtxt,tcorr=mpst,nu=mnum,ti=ti,pr=pr)

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
