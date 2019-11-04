Placer les scripts .py dans votre dossier.

Pour un résultat optimal, la colonne output et pred doivent être nomées :

target, pred

les autres colonnes générées par bigml ne doivent pas être modifiés (confidence, proba ect ect)

j'ai normalement ouvert les modules à tout projet, cependant pour confort stocker les nons des colones dans des
variables courtes au début, ils vous seront souvent demandés en cas d'appel de fonction, cela permet d'utiliser les
modules sur différent fichier si comparaison entre projet ou autre besoin... 

Ex: 
    output = 'target'
    pred = 'pred'


Concernant l'optimisation, la méthode de calcul de seuil opti ne l'est pas :) compter 30 min pour 30 000 lignes ...  
Je sais ou ca coince, si vous avez une solution :)  sachant qu'il faut prendre en compte la structure ouverte des class comme
contrainte.

Si vous êtez utilisateur des scripts, merci de me reporter tout bug, suggestion, amelioraiton, optimisation que vous pouvez y apporter !!

 