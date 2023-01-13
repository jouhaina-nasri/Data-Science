from flask import Flask, render_template,request,redirect,session,url_for,jsonify,flash
import os
import numpy
from PIL import Image
import cv2
import skimage.feature 
import math

app=Flask(__name__)
#connecter avec la base #
app.secret_key=os.urandom(24)


#-------------------------------------------------------------------------------------------------------#
# page d'accueil #
@app.route('/')
def accueil ():
    return render_template('index.html')
###########

def histogram(killy):
    maximum = numpy.max(killy)
    minimum = numpy.min(killy)
    dim = maximum - minimum + 1
    hist, bins = numpy.histogram(killy, bins=dim)
    return hist

def readFiles(entry):
    image1 = Image.open('images/'+entry)
    image_gray = image1.convert('L')
    killy = numpy.array(image_gray)
    hist1 = histogram(killy)
    return(hist1)

def calcul_distance(h1, h2):
    size1 = len(h1)
    size2 = len(h2)
    somme = 0
    somme2 = 0
    if size2 > size1:
        for i in range(size2):
            if i < size1:
                somme = somme + (min(h1[i], h2[i]))
            else:
                somme = somme + h2[i]
    else:
        for i in range(size1):
            if i < size2:
                somme = somme + (min(h1[i], h2[i]))
            else:
                somme = somme + h1[i]

    for i in range(size1):
        somme2 = somme2 + h1[i]
    distance = 1 - somme / somme2
    return distance

# upload #

dictionary={}
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
      image = request.files['file'].read()
      npimg = numpy.fromstring(image,numpy.uint8)
      img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      hist1 = histogram(gray)
      entries = os.listdir('images/')
      i=0
      for entry in entries:
          i=i+1
          hist = readFiles(entry)
          distance = calcul_distance(hist1, hist)
          dictionary[str(entry)] = distance
      sorted_desserts = {w:dictionary[w] for w in sorted(dictionary, key=lambda x: dictionary[x] if x in dictionary else None, reverse=False)}
    return render_template('my_other_template.html', 
                       keys=sorted_desserts)

def Gris(image):
    gray = normalisationImage(image)
    return gray

#Normalisation de l'image
def normalisationImage(image):
    normImage = image//16
    normImage = normImage.astype('uint32')
    return normImage

    
#Histogramme d'une image à niveau de gris
def Histogramme(mat):
    histogramme = cv2.calcHist([mat], [0], None, [24,24, 24],[0, 256])
    return histogramme
    
#Calcul de la distance
def CalculDistance(image1,image2):
    d = (numpy.linalg.norm((image1-image2))/5)
    return d

#Matrice de co-occurence
def MatCooccurence(image_gris):
    matCo = skimage.feature.graycomatrix(image_gris, [5], [0,numpy.pi/2,numpy.pi/4,(numpy.pi*3)/4], 256,
                         symmetric=True, normed=True)
    return matCo

#Calcul des parametre de la co occurence
def ParamCooccurence(HistomatCoo):
    #Calcul de l'energie
    energie = skimage.feature.graycoprops(HistomatCoo,'energy')
    contraste = skimage.feature.graycoprops(HistomatCoo,'contrast')
    dissimilarite = skimage.feature.graycoprops(HistomatCoo,'dissimilarity')
    homogeneite = skimage.feature.graycoprops(HistomatCoo,'homogeneity')
    correlation = skimage.feature.graycoprops(HistomatCoo,'correlation')
    return energie, contraste, dissimilarite,homogeneite,correlation

#Apprentissage
def Apprentissage(greyImage):
        MatCoo = MatCooccurence(greyImage)
        energie,contraste,dissimilarite,homogeneite,correlation = ParamCooccurence(MatCoo)
        energie = energie[0][0]
        contraste = contraste[0][0]
        dissimilarite = dissimilarite[0][0]
        homogeneite = homogeneite[0][0]
        correlation = correlation[0][0]
        return energie,contraste,dissimilarite,homogeneite,correlation

def Calcul_distance(energie,contraste,dissimilarite,homogeneite,correlation,energie2,contraste2,dissimilarite2,homogeneite2,correlation2):
    d= math.sqrt(math.pow(energie-energie2,2)+math.pow(contraste-contraste2,2)+math.pow(dissimilarite-dissimilarite2,2)+
           math.pow(homogeneite-homogeneite2,2)+math.pow(correlation-correlation2,2))
    return d

def read(entry):
    image = cv2.imread('images/'+entry)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = normalisationImage(gray)
    return(gray)
    
@app.route('/desctex', methods = ['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        image = request.files['file'].read()
        npimg = numpy.fromstring(image,numpy.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        greyImage = Gris(gray)
        energie,contraste,dissimilarite,homogeneite,correlation = Apprentissage(greyImage)
        entries = os.listdir('images/')
        i=0
        for entry in entries:
          i=i+1
          hist = read(entry)
          energie2,contraste2,dissimilarite2,homogeneite2,correlation2 = Apprentissage(hist)
          distance = Calcul_distance(energie,contraste,dissimilarite,homogeneite,correlation,energie2,contraste2,dissimilarite2,homogeneite2,correlation2)
          dictionary[str(entry)] = distance
    sorted_desserts = {w:dictionary[w] for w in sorted(dictionary, key=lambda x: dictionary[x] if x in dictionary else None, reverse=False)}
    return render_template('my_other_template.html', 
                       keys=sorted_desserts)


dictionary2={}
@app.route('/twodesc', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
      image = request.files['file'].read()
      npimg = numpy.fromstring(image,numpy.uint8)
      img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      greyImage = Gris(gray)
      energie,contraste,dissimilarite,homogeneite,correlation = Apprentissage(greyImage)
      hist1 = histogram(gray)
      entries = os.listdir('images/')
      i=0
      for entry in entries:
          i=i+1
          hist = readFiles(entry)
          text = read(entry)
          distance = calcul_distance(hist1, hist)
          energie2,contraste2,dissimilarite2,homogeneite2,correlation2 = Apprentissage(text)
          distance2 = Calcul_distance(energie,contraste,dissimilarite,homogeneite,correlation,energie2,contraste2,dissimilarite2,homogeneite2,correlation2)
          dictionary[str(entry)] = distance
          dictionary2[str(entry)] = distance2
      sorted_desserts = {w:dictionary[w] for w in sorted(dictionary, key=lambda x: dictionary[x] if x in dictionary else None, reverse=False)}
      sorted_desserts2 = {w:dictionary2[w] for w in sorted(dictionary2, key=lambda x: dictionary2[x] if x in dictionary2 else None, reverse=False)}
      
    return render_template('my_other_template1.html', 
                       keys=sorted_desserts, values=sorted_desserts2)

@app.route('/test')
def get_image_category():
    return render_template('test.html')


#main#
if __name__=="__main__":
    app.run(debug=True)
#------------------------#
