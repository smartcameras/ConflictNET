# seed initialization for reproducible results
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# imports
from keras.models import Sequential,Model
from keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,LSTM,Dense,GlobalAveragePooling1D,AveragePooling1D  
from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
import argparse
import numpy as np
import keras.backend as K
from dataLoad import load_tr,load_val,load_test
from keras.models import load_model,Model
from keras_self_attention import SeqSelfAttention                     
from keras import losses

#custom metric
def pearson_cc(x,y): # x-ground truth and y-prediction

        x_mean = K.mean(x,axis=0)
        y_mean = K.mean(y,axis=0)
        x_std = K.std(x,axis=0)
        y_std = K.std(y,axis=0)
        n = K.mean((x - x_mean) * (y - y_mean))
        d = x_std*y_std
        return (n/d)

#custom loss
def pearson_loss(x,y):

        x_mean = K.mean(x,axis=0)
        y_mean = K.mean(y,axis=0)
        x_std = K.std(x,axis=0)
        y_std = K.std(y,axis=0)
        n = K.mean((x - x_mean) * (y - y_mean))
        d = x_std*y_std
        p_loss = (1-(n/d)) # pearson cc loss
        return p_loss

# range modification
def normalize(x,old_max,old_min,new_max,new_min):

        new_x = (((x-old_min)*(new_max-new_min))/(old_max-old_min))+new_min
        return new_x
        
def conflictNet(input_shape):

        inp = Input(shape=input_shape)

        x = Conv1D(filters=64, kernel_size=6, strides=1, padding='valid', data_format='channels_last', activation = 'relu',input_shape=input_shape)(inp)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=8, strides=8)(x)

        x = Conv1D(filters=128, kernel_size=4, strides=1, padding='valid', activation = 'relu', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=6,strides=6)(x)

        x = Conv1D(filters=256, kernel_size=4, strides=1, padding='valid', activation = 'relu', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=6,strides=6)(x)

        #Average Pooling 1D block
        x = AveragePooling1D(pool_size=4)(x)

        #LSTM1
	x = LSTM(units=128,activation='tanh',return_sequences=True)(x)

        # Self Attention
        x = SeqSelfAttention(attention_activation='tanh')(x)

        #LSTM2
	x = LSTM(units=64,activation='tanh',return_sequences=False)(x)

        #Dense
	y = Dense(units=1,use_bias=True)(x) # regression
        
        model = Model(inputs=inp,outputs=y)

        opt = optimizers.Adam(lr=0.01,decay=0.6)
        model.compile(optimizer=opt,loss=pearson_loss,metrics=[pearson_cc]) # regression
        
        return model
        
def train(model, x_tr, y_tr, x_val, y_val, args):

        reduce_lr = ReduceLROnPlateau(monitor='val_pearson_cc', factor=0.2, patience=3, mode='max', min_lr=0.001,verbose=1)        
        es = EarlyStopping(monitor='val_pearson_cc',patience=10,mode='max',restore_best_weights=True) # regression
        mc = ModelCheckpoint('best_model.h5', monitor='val_pearson_cc', mode='max', verbose=1, save_best_only=True) # regression
        history = model.fit(x_tr,y_tr,batch_size=args.batch_size,epochs=args.num_epochs,validation_data=(x_val,y_val),callbacks=[mc,es])
        return model
        
def test(model,x_t,y_t):

        score = model.evaluate(x_t,y_t,batch_size=32)
        print(model.metrics_names)
        print(score)
        return score

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        args.batch_size = 32
        args.num_epochs = 1500 #best model will be saved before number of epochs reach this value

        x_tr, y_tr,z = load_tr()
        x_val, y_val,z = load_val()
        x_t, y_t,z = load_test()        

        y_tr = normalize(y_tr,10,-10,1,-1) # change output label range from [-10,10] to [-1,1]
        y_tr = np.reshape(y_tr,(y_tr.shape[0],1))
        y_val = normalize(y_val,10,-10,1,-1)
        y_val = np.reshape(y_val,(y_val.shape[0],1))
        y_t = normalize(y_t,10,-10,1,-1)
        y_t = np.reshape(y_t,(y_t.shape[0],1))
        
        #define model
        model = conflictNet(input_shape=(240000,1))
        model.summary()

        #train model
        model = train(model,x_tr,y_tr,x_val,y_val,args=args)
        
        #test model
        model = load_model('best_model.h5',custom_objects={'pearson_loss':pearson_loss,'pearson_cc':pearson_cc,'SeqSelfAttention':SeqSelfAttention})

        score = test(model,x_t,y_t)
        pred_values = model.predict(x_t,batch_size=32)
        
        #Modify predicted values range
        max_tr = np.max(y_tr)
        min_tr = np.min(y_tr)

        p = normalize(pred_values,np.max(pred_values),np.min(pred_values),max_tr,min_tr)        

        print('########################PCC-Value#####################')
        y_mean = np.mean(y_t,axis=0)
        p_mean = np.mean(p,axis=0)
        y_std = np.std(y_t,axis=0)
        p_std = np.std(p,axis=0)
        n = np.mean((y_t-y_mean)*(p-p_mean))
        d = y_std*p_std
        pcc = n/d
        print('PCC:',pcc)
        
        print('######################WAR and UAR#########################')
        tp = 0
        tn = 0
        tot_p = 0
        tot_n = 0
        for i in range(len(p)):
                if(y_t[i]>=0):
                        tot_p = tot_p+1
                        if(p[i]>=0):
                                tp = tp+1
                else:
                     	tot_n = tot_n + 1
                        if(p[i]<0):
                                tn = tn+1

        r1 = float(tp)/tot_p
        r2 = float(tn)/tot_n

        uar = (r1+r2)/2
        war = float(tp+tn)/(tot_p+tot_n)

        print('R1:',r1,'R2:',r2,'UAR:',uar,'WAR:',war)

