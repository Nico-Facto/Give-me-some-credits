Analyse avec le script pand.py

Pour générer une analyse =
python pand.py pand matrix target pred --file pred_bigml.csv --extract analyse.csv

Pour générer une analyse avec seuil opti =
python pand.py pand seuil target pred --file pred_bigml.csv --extract analyseSeuill.csv

Ensuite, on peut appliquer et générer un fichier erreure majeur depuis
le menu dynamique choisir 1 et exporter dans un csv appart.

D'autres options sont disponibles dans le script pour les cas de regréssion par exemple, étudié le script et les args. pour comprendre comment les éxécuter.

Ce script est automatisé pour tout fichier.csv, je génére la colonne target qui est l'output à prédire, cette colonne est nomée depuis mon script bigml, idem pour la colonne prédiction qui s'appellera toujours pred. 

Ensuite je ne change pas le nom des colonnes proba, de cette façon le script restera automatisé pour tout mes projets.


Analyse AUC/Ammont of data

placé le script bigzer.py et graph.py dans votre dossier de travail et exécuté le script bigzer, préparé les id des datasets à analyser.
Un graphique sera affiché et crée en local au format .png

 