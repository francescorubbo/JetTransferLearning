import h5py
f = h5py.File('data/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5','r')

images = f['image'].value
signal = f['signal'].value

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images[::10], signal[::10], test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

pathtrain = 'data/train/'
pathtest = 'data/test/'
pathval = 'data/val/'

from PIL import Image,ImageOps
paths = [pathtrain,pathtest,pathval]
Xs = [X_train,X_test,X_val]
ys = [y_train,y_test,y_val]

import os

for basepath,X,y in zip(paths,Xs,ys):

    pathsig = basepath+'signal/'
    pathbkg = basepath+'background/'

    for directory in [pathsig,pathbkg]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for ii,(img,sig) in enumerate(zip(X,y)):
        if not ii%10000: print(ii)
        image = Image.fromarray(img)
        image = image.convert('RGB')
        inverted_image = ImageOps.invert(image)
        path = pathsig if sig else pathbkg
        inverted_image.save(path+'/img%s.jpg'%ii)
    
