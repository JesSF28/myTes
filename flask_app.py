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
import prep_gvfc_balloon_1 as prp1       # Algoritmo Preproceso 1
import prep_gvfc_balloon_2 as prp2       # Algoritmo Preproceso 2
import post_pnorv_1 as postp1            # Algoritmo Postproceso
import palabr_clave as palc              # Algoritmo Palabras clave
import estadistica as est                # Conteo estadistica

DEVELOPMENT_ENV = True

app = Flask(__name__)

app_data = {
    "titulo": "Proceso Algorítmico de Recuperación Eficiente en Gestión Documental Histórica",
    "especial": "Ciencias de la Computacion",
    "name": "Peter's Starter Template for a Flask Web App",
    "descripcion": "Proceso documental histórico:",
    "referencia": "Referencia Bibliográfica de Tesis:",
    "recursos": "Recursos utilizados en Tesis:",
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
    return render_template("jsf_res_tesis_1_0.html", app_data=app_data)

@app.route("/presenta")
def presenta():
#    return render_template("presenta.html", app_data=app_data)
    return render_template("jsf_pres_tesis_0_1.html", app_data=app_data)

@app.route("/documento")
def service():
    return render_template("jsf_doc_tesis_14_0.html", app_data=app_data)

@app.route("/codigo")
def codigo():
    global filename
    return render_template("codigo1.html", app_data=app_data, fn=filename)

@app.route("/recursos")
def recursos():
    return render_template("recursos.html", app_data=app_data)

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
#        ruta0="assets"
        ruta0="static"
        ruta1="static/result"
        if filename=="":
            f = request.files['archivo']
#        f = request.FILES['ufile'].file.name
            filename = secure_filename(f.filename)       # Nom Arch
            if filename=="":
               return render_template("codigo1.html", app_data=app_data, fn=filename)
                
        filename0 = os.path.join(ruta0, filename)         # Ruta+nom Arch origen
        filename1 = os.path.join(ruta1, filename)         # Ruta+nom Arch destino
        fn,ex = os.path.splitext(filename)               # Nom Arch, Ext
        print("filename: ",filename)
        print("filename0: ",filename0)
        print("filename1: ",filename1)
        print("Long filename: ",len(filename))
        imagen = cv2.imread(filename0, cv2.IMREAD_GRAYSCALE)

        su =  request.form.getlist("act")
        su2 = request.form.getlist("act2")
        print("su:",su)
        mtxt = []
        mimg = []
        mnum = []
        mpst = []
        mkyw = []

        mtp=[]
        mtc=[]
        mp= []
        mn= []
        n=0
        ti=[]
        for s in su:
            print(int(s))
            if int(s)==0:
                nu="0"
                mimg.append(filename0)                                # Nomb imagen original
                if "3" in su2: mtxt.append(ocrt(fn,imagen,nu,ruta1))        # Agrega texto de ocr 0
                if "4" in su2: mpst.append(postp1.palabr(ruta1,fn+nu)) # Agrega texto corregido
                if "5" in su2: 
                    kyw,akyw,yr,se,re=palc.palabr_c(ruta1,fn+nu,3) 
                    mkyw.append(kyw)                                  # Agrega palabras clave
                mnum.append(n)                                        # Agrega Num opcion
                ti.append("Imagen Original")                          # Agrega titulo Imagen
            if int(s)==1:                                             # Alg 1 - Prepr       
                nu="1"
                img,fn1 = prp1.preproc(imagen,fn+nu,ruta1)      
                mimg.append(fn1)                                      # preproceso 1
                tx1=""
                if "3" in su2: 
                    tx1=ocrt(fn,img,nu,ruta1)                               # ocr 1
                    mtxt.append(tx1)           
                if "4" in su2: 
                    tx2=postp1.palabr(ruta1,fn+nu)
                    mpst.append(tx2)                                  # Txt corregido 1
                    tp,tc,p=est.gen_estad(tx1, tx2)                   # Estadística
                    mtp.append(tp)
                    mtc.append(tc)
                    mp.append(p)
                if "5" in su2: 
                    kyw,akyw,yr,se,re=palc.palabr_c(ruta1,fn+nu,3) 
                    mkyw.append(kyw)                                  # Agrega palabras clave
                mnum.append(n)
                mn.append("Alg-"+nu)
                ti.append("Imagen Algoritmo 1")
            if int(s)==2:                                             # Alg 2 - Prepr
                nu="2"
                img,fn2 = prp2.preproc2(imagen,fn,nu,ruta1)     
                mimg.append(fn2)
                tx1=""                                      
                if "3" in su2:
                    tx1=ocrt(fn,img,nu,ruta1)                               # ocr 2
                    mtxt.append(tx1)           
                if "4" in su2:
                    tx2=postp1.palabr(ruta1,fn+nu)                     # postpoceso
                    mpst.append(tx2)                                  # Txt corregido 2
                    tp,tc,p=est.gen_estad(tx1, tx2)                   # Estadística
                    mtp.append(tp)
                    mtc.append(tc)
                    mp.append(p)
                if "5" in su2: 
                    kyw,akyw,yr,se,re=palc.palabr_c(ruta1,fn+nu,3) 
                    mkyw.append(kyw)                                  # Agrega palabras clave
                mnum.append(n)
                mn.append("Alg-"+nu)
                ti.append("Imagen Algoritmo 2")
            if int(s)==3:                                             # Alg 3 - Prepr
                nu="3"
                img,fn2 = prp2.preproc2(imagen,fn,nu,ruta1)            # preproceso 3
                mimg.append(fn2)
                tx1=""
                if "3" in su2: 
                    tx1=ocrt(fn,img,nu,ruta1)                               # ocr 3
                    mtxt.append(tx1)           
                if "4" in su2: 
                    tx2=postp1.palabr(ruta1,fn+nu)                     # Txt corregido 3
                    mpst.append(tx2)                                  
                    tp,tc,p=est.gen_estad(tx1, tx2)                   # Estadística
                    mtp.append(tp)
                    mtc.append(tc)
                    mp.append(p)
                if "5" in su2: 
                    kyw,akyw,yr,se,re=palc.palabr_c(ruta1,fn+nu,3) 
                    mkyw.append(kyw)                                  # Agrega palabras clave
                mnum.append(n)
                mn.append("Alg-"+nu)
                ti.append("Imagen Algoritmo 3")
            n += 1
            print("mtxt: ",mtxt)
            print("mimg: ",mimg)
            print("mnum: ",mnum)
            print("mpst: ",mpst)
            print("mkyw: ",mkyw)
            print("% corr: ",mp)
            print("Alg Nu: ",mn)
    pr+=1    
    print("prueba: ",pr)
    fng=""
    if len(mn)>0 and len(mp)>0:
        fng=est.est_graf(mn, mp,fn,ruta1)
    return render_template("codigo2.html", app_data=app_data, fn1=mimg, txt=mtxt,tcorr=mpst,plbc=mkyw,nu=mnum,ti=ti,pr=pr,su2=su2,fng=fng)

def ocrt(fn,img,nu,ruta1):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    txt = pytesseract.image_to_string(img)                                    # Genera texto
    tx = open (os.path.join(ruta1,fn+nu+".txt"),'w', encoding="utf-8")
    tx.write(txt)
    tx.close()
    return txt

if __name__ == "__main__":
    app.run(debug=DEVELOPMENT_ENV)
