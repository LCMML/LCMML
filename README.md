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



