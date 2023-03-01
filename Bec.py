#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import hog
import pandas as pd
import os # Pentru a sterge csv
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Sa nu mai apara warning la hog 
n = 160; # Nr de poze
tip_bec = 2 # Cate tipuri de becuri avem

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def Binarizare( img ):
	# Binarizam imaginea -> 255-alb 0-negru
	thresh = 128
	maxval = 255
	im_bin = (img >= thresh ) * maxval
	img = np.uint8(im_bin)

	return img
def Read(n): 

	if os.path.isfile('Baza_de_date.csv'): # Verifica existenta fisierului
		os.remove('Baza_de_date.csv')

	k = 1 
	for i in range(1,n+1): # numele imaginilor este inte 1-80

		Img = imread(f"./Data/{i}.jpg") # Citire imagine
		gray = rgb2gray(Img) # Convert imagine la alb-negru
		gray = Binarizare( gray )

		# Creating hog features
		vTest, hog_image = hog(gray, orientations=9, 
                       		   pixels_per_cell=(8, 8), 
                      		   cells_per_block=(2, 2), 
                       		   visualize=True, multichannel=False)

		#x = np.ndarray.flatten(hog_image)
		x = vTest.flatten() # Transpusa vectorului coloana
		#print(np.size(x)) # o imagine de 24x24 de pixeli

		x.resize(np.size(x) + 1)
		if k <= ( n / tip_bec / 2 ): # consideram fiecare set de imagini avand jumatate de 
			x[np.size(x) - 1] = 1  # date cu becul aprins si jumatate cu becul stins
		else: 
			x[np.size(x) - 1] = -1

		with open('Baza_de_date.csv', 'a') as f: # append
			writer = csv.writer(f, lineterminator='\r') # Creare CSV
			if i == 1: # linia 1
				index = [j for j in range(np.size(x))] # intializam un vector cu valori de la 0 la 575 si label
				index[np.size(x) - 1] = "label"
				writer.writerow(index)

			writer.writerow(x) # Scriem vectroul x
		
		if k == 80:
			k = 0
		else:
			k = k + 1

def Light(x, n):

	i = input("Alegeti o valoare intre 1 si 4: " )
	img = imread( i + '.jpg')

	plt.subplot(1, 2, 1)
	plt.axis("off") # Scoatem axele
	plt.imshow(img, cmap="gray")

	gray = rgb2gray( img ) # Convert to gray
	gray = Binarizare( gray )

	vTest, hog_image = hog( gray, orientations=9, 
							pixels_per_cell=(8, 8), 
							cells_per_block=(2, 2), 
							visualize=True, 
							multichannel=False )

	plt.subplot(1,2,2)
	plt.axis("off")
	plt.imshow(hog_image, cmap="gray")
	plt.show()

	# .T = transpusa unei matrici
	test_label = vTest.T@x[:n,] + x[n]
	print(test_label)

	return test_label

Read(n) # Citirea imaginlor si scrierea lor in csv

#Citim baza de date pentru antrenare si testare
df = pd.read_csv('Baza_de_date.csv') # Fisierul pentru antrenare
df_t = pd.read_csv('Baza_de_date_test.csv') # Fisierul pentru testare
df['label'].nunique()
print(f'Data Frame: \n {df}') # DataFrame

# Eliminam coloana label pentru a ramane doar cu caracterisiticile pozelor
v = df.drop('label', axis=1)  
eticheta = df['label'] 

N,m = v.shape # N este nr de poze si dimensiunea vectroului caracteristic
A = np.append( v, np.ones((N,1)), axis=1) # adaugam o noua coloana la matrice

x = np.linalg.lstsq(A, eticheta, rcond=None)[0] # returneaza metoda in sens CMMP
print(f'\nMetoda in sens CMMP: \n {x} \n\n')

test_label = Light(x, m) # Afiseaza cele doua poze si testeaza clasificatorul
if test_label >= 0:
		print('Becul este APRINS!')
else:
		print('Becul este STINS!')


