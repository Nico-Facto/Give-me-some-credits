Rapport : Rapport_Ensemble_Deep2.pdf
Meilleur résultat avec seuil 0.31
gain : 10 160 500
--+ 319 000

Rapport : ensemble sans seuil opti
résultat : 9 844 000


Analyse générée avec le script pand.py

Pour générer une analyse =
python pand.py pand matrix target pred --file pred_bigml.csv --extract analyse.csv

Pour générer une analyse avec seuil opti =
python pand.py pand seuil target pred --file pred_bigml.csv --extract analyseSeuill.csv

Ensuite, on peut appliquer et générer un fichier erreurmajeur depuis
le menu dynamique selection 1 et l'exporter dans un csv appart.

D'autres options sont disponibles dans le script pour les cas de regréssion par exemple, étudié le script et les args. pour comprendre comment les éxécuter.

Ce script est automatisé pour tout fichier.csv, je génére la colonne target qui est l'output à prédire, cette colonne est nomée depuis mon script bigml, idem pour la colonne prédiction qui s'appelera toujours pred. 

Ensuite je ne change pas le nom des collones proba, de cette façon le script restera automatisé pour tout mes projets.

 