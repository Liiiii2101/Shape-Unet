from data_processing import *
from get_shape_unet import *
import sys
from sklearn.model_selection import train_test_split


path_im = sys.argv[1]

path_lab = sys.argv[2]

images = np.load(path_im)
labels = np.load(path_lab)

labels = (labels>0.5)

trainx,testx,trainy,testy = train_test_split(images,labels,test_size=0.1,shuffle = True)

b,v,trainy_mean = eigen_matrix(trainy)

vtrans = tf.convert_to_tensor(v,tf.float32)
print('pca finished')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
model = get_unet(vtrans)
model.compile(loss=Euclidean_loss, optimizer=opt.Adam(lr=0.0001))
model.summary()
model.fit(trainx,b,validation_split=0.1,batch_size=32,epochs=400,verbose =1,callbacks=[early_stopping])

