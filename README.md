# LCMML
## Dokumentation

## Notizen
Hallo Ronald xoxo Test Github funktion
Hallo Ronald zum zweiten
## Links
[So erstellt man einen Link für bsp. Google](https://www.google.com)



# Notizen zur VU
## Session 6
### Machine Learing
#### Supervised Learning
Trainingsdatensatz wird vorgegeben. Für die Trainingsdaten hat man Labels, die beschreiben, welche Klasse welches Label hat. Kann zur Klassifikation angewandt oder für Regression angewandt werden.
Es gibt Pixel- oder Objektbasierte Methoden.
**Man benötigt:**
* Raw Data (bei uns Satellitendaten )
* Labels (numerische Werte, zB.: 0-3, wobei 0 = schnee, 1 = fels, 2 = wald und 3 = wiese)
  * Labels werden normal Manuel aufgenommen (Arbeitsintensiv)
  * Nur für einen Ausschnitt
Aus Labels und Rohdaten wird der Trainings- und Testdatensatz abgeleitet.
Beim Trainingsdatensatz wird:
* Das Lernmodell zum trainieren gewählt. Bei uns Random Forrest.
* Danach soll das Modell eine Vorhersage treffen können.
* Diese Vorhersage muss dann noch validiert überprüft werden.

**Hyperparameter:**
* Steuern den Lernprozess.
* Durchtesten dieser ist notwendig um gute Ergebnisse zu liefern.
* Beispiele hierfür sind:
    * Lernrate
    * Anzahl der Bäume im Random Forest
        * Sollte nicht kleiner als 10 sein aber auch nicht zu groß
    * Wichtigkeit der Variablen wird zudem ausgegeben.


**Modellparameter**
* Sind Modellintern und werden im Lernprozes erzeugt/eingestellt.
* Wir als nutzer haben keinen Einfluss darauf.


**Terminologie:**
* Feature, variable, predictor: Werden Synonym genutz. Eine Variable kann z.B. das rote Band sein. Üblicherweise mehrere Feature.
* Labels: Vorgegebenen Labels um den Klassifikator zu Trainieren. Z.B: Klassenwerte pro Pixel.
* Feature space: Mehrdimensionale Raum, den die Variablen aufspannen. Die Dimension entspricht der Anzahl an Features die man zur verfügung stellt.
* Training, validation and test (set): Training -> Hiervon Lernt das Modell. Validation -> meistens ganz am Ende, hiermit wird die Genauigkeit des Modells eingeschätzt. Test -> wird verwendet um Hyperparameter zu optimieren. 
* (non) Parametric: Parametrische Methoden nehmen eine bestimmte Verteilung der Variablen an und umgekehrt. z.B. Einer Gauskurve folgend. In Fernerkundung sind nichtparametrische Methoden beliebter.
* "X": So werden oft die Daten/Variablen bezeichnet.
* "y": So werden oft die Labels bezeichnet.

**Entscheidungsbäume:**
* Werden für Klassifikation und Regression genutzt.
* Funktion:
    * Man Teilt die Daten anhand einer Variable auf.
    * Hierfür sucht man (bzw. der Entscheidungsbaum) die Passendste
    * Der Baum teilt sich nach der Entscheidung.
    * Es gibt in diesem Fall zwei neue Punkte, an denen sich der Baum teilen kann. Dies führt der Baum iterativ weiter.
    * Werden neue Daten durch den Baum geschickt, nehmen sie den Pfad, der dem vorherigen Training entsprungen ist.
* Basiert auf boolscher Logik.

**Random Forest:**
* Random Forest fasst vieler solcher Entscheidungsbäume in einem Modell.
* 2/3 der Trainingsdaten werden als Samples herangezogen.
* Es wird immer nur ein Subset an Daten herangezogen, wodurch die Bäume untereinander nicht so sehr korrelieren.
* An einem Entscheidungspunkt wird die Variable zur Teilung des öfteren Zufällig gesetzt und somit der beste Wert hierfür bestimmt.
* 1/3 wird für "out-of-bag" accuracy genommen, also zum validieren. 
* Wenn man die Bäume dann hat:
    * Man schickt neue Daten durch jeden Baum im Forrest durch.
    * Jeder Baum gibt eine Klasse aus.
    * Bei z.B. 50 Bäumen und 50 Entscheidungen...
    * Der Algorithmus schaut dann, welche Klasse am öftesten auftritt und gibt diese dann aus Voraussage aus.


**model.fit(X,y)**
X sind hier die Trainingsdaten. Shape of X: Die Reihen repräsentieren die Anzahl der Samples, also Pixel im Bild. Die Spalten die Variablen (Bänder, Zeitschritte, o.Ä.). Somit ist es ein 2 Dimensionales Array. Shape of y: 1 Dimensional, gibt Labels passend zu den Samples an. 

**model.predict(X)**
X können hier alle/neue Daten sein. Gibt ein anderes y aus, welches die Klassenlabels repräsentiert. Muss danach wieder in ein 2 Dimensionales Array umgeformt werden.


**predict_proba(X)**
Wahrscheinlichkeiten für alle Klassen. Array der 2. Dimension . Erste ist die Anzahl an Samples/Pixel und zweite gibt die Wahrscheinlichkeit für jede Klasse aus.

**n_jobs Parameter**
Hiermit kann man die Anzahl der genutzten Kerne für den Rechenauftrag einstellen. 

***WORKFLOW***
* **Training, Test und Validierungsdaten besorgen**
* **Geodaten besorgen**
* **Imports**
```python
# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
```
* **Daten und Funktionen vorbereiten**
    * Reprojezieren und Rasterisieren der Daten
```python
# Pfade für die Umgebung Setzen!
bands_10m = ["B02", "B03", "B04", "B08"]
path_sat_img = Path("./data_2021/s2_img")

# training test daten
training_vector = "XXX"
training_raster = "XXX" #output Pfad für das zu erzeugende Raster

# rasterize training and test data
xmin, ymin, xmax, ymax = 654000, 5200140, 672760, 5227400
res = 10
datatype_out = "Int16"
burn_field = "class_id"
nodata = "-1"

cmd = f"gdal_rasterize -a {burn_field} -of GTiff -te {xmin} {ymin} {xmax} {ymax} -tr {res} {res} -a_nodata {nodata} -ot {datatype_out} {training_vector} {training_raster}"

# os.system(cmd) #Einmal ausführen, zum Umwandeln

# Hilfsfunktion zum Rastereinlesen
def read_img(path_to_img):
    with rasterio.open(path_to_img, "r") as img:
        return img.read(1).astype(np.float32)
```
* **Daten Laden**
```python
# Bänder Einlesen 
bands = []
for f in path_sat_img.glob("*tif"):
    # print(f) #sollte hier alle benötigten Bilder ausdrücken.
    band_id = f.stem.split("_")[2]
    if band_id in bands_10m:
        print(f)
        bands.append(read_img(f))

#Stapelt Bänder
bands = np.dstack(bands)

print(bands.shape) #  Gibt Reihen, Spalten, Kanäle aus.

# read metadata for output
template = {}
with rasterio.open(training_raster, "r") as img:
    template["transform"] = img.transform
    template["crs"] = img.crs
    template["height"] = img.height
    template["width"] = img.width

# read training/test data
# just for reference
mapping = {
    0: "rock_debris",
    1: "forest",
    2: "grassland",
    3: "ice_snow"
}
labels = read_img(training_raster)

print(labels.shape)
print(np.unique(labels, return_counts=True))

# reshape training test data 
labels_1d = labels[labels>=0].reshape((-1))
print(labels_1d.shape)


# split into 
X_train, X_test, y_train, y_test = train_test_split(bands[labels>=0,:], labels_1d, test_size=0.3, random_state=0, shuffle=True)

# instantiate classificator
rf = RF(n_estimators=50, n_jobs=-1, oob_score=True)


# train our model
rf.fit(X_train, y_train)

# extract information for reshaping
rows, cols, n_bands = bands.shape

# transform input data into 2d array of shape 
X_predict = bands.reshape((rows*cols, n_bands))

# apply our model on (unseen) data
y_pred = rf.predict(X_predict)

# reshape to shape of (rows, cols) for output as 2d gis raster
y_pred_2d = y_pred.reshape((rows, cols))

with rasterio.open(
    "prediction.tif",
    "w",
    driver="GTiff",
    height=y_pred_2d.shape[0],
    width=y_pred_2d.shape[1],
    count=1,
    dtype=y_pred_2d.dtype,
    crs=template["crs"],
    transform=template["transform"]) as fobj:
    fobj.write(y_pred_2d, 1)
```
* **Model Trainieren**
* **Model Validieren**
* **Model Anwenden**