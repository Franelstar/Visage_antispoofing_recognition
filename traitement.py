import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# Chargement des images
path_train = 'test'
path_json = 'test_label.json'
person_content = ['live', 'spoof']

list_person_folder = os.listdir(path_train)

cols = ['image_name','class', 'score', 'type_usurpation', 'condition_eclairage', 'environnement']
images_ = pd.DataFrame([], columns=cols)
first = 'TEST'

# On charge les caractéristiques des images
with open(path_json) as file:
    jsonData = json.load(file)

# On crée le dossier de destination
image_path = 'visage_dataset_test'
if not os.path.exists(image_path):
    os.mkdir(image_path)

# Nombre d'image total
file_count = sum(len(files) for _, _, files in os.walk(path_train))

i = 0

with tqdm(total=int(file_count/2), desc="Traitement en cour ...", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
    for person in list_person_folder:
        for label in person_content:
            path_person = path_train + '/' + person + '/' + label
            if os.path.exists(path_person):
                images_person = [ f for f in os.listdir(path_person) if os.path.isfile(os.path.join(path_person,f)) ]
                for img in images_person:
                    if img.split('.')[1] in ['png', 'jpg']: 
                        # On charge l'image
                        image = cv.imread(os.path.join(path_person, img))

                        try:
                            # On change l'image en RGB
                            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                            # On récupère le fichier de l'image
                            fichier = open(os.path.join(path_person, img.split('.')[0] + "_BB.txt"), "r")
                            texte = fichier.readline()
                            fichier.close()

                            # On récupère les dimensions de l'image
                            real_h, real_w, real_d = image.shape

                            # On récupère les coordonnées du visage
                            x1 = int(int(texte.split(' ')[0])*(real_w / 224))
                            y1 = int(int(texte.split(' ')[1])*(real_h / 224))
                            w1 = int(int(texte.split(' ')[2])*(real_w / 224))
                            h1 = int(int(texte.split(' ')[3])*(real_h / 224))

                            # On récupère le visage
                            visage = image[y1:y1+h1, x1:x1+w1]
                        
                            # On sauvegarde le visage
                            name_i = '{}__{}_{}__{}.jpg'.format(first, person, img.split('.')[0], i)
                            plt.imsave(image_path + '/' + name_i, visage)
                            i += 1

                            # On sauvegarde l'image et son label
                            score = texte.split(' ')[4].split('\n')[0]
                            type_usurpation = jsonData['Data/test/'+person+'/'+label+'/'+img][40]
                            condition_eclairage = jsonData['Data/test/'+person+'/'+label+'/'+img][41]
                            environnement = jsonData['Data/test/'+person+'/'+label+'/'+img][42]
                            df = pd.DataFrame([[name_i, label, score, type_usurpation, condition_eclairage, environnement]], columns=cols)
                            images_ = images_.append(df, ignore_index=True)
                        except:
                            pass

                        pbar.update(1)
                
images_.to_csv(image_path + '/dataset_' + first + '.csv')

print("Traitement terminé")
print("Nous avons un total de {} images".format(i))
