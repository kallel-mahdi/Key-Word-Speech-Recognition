import os
from MFCC import mfcc
import time
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.keras import backend as k


class translator:
    def __init__(self):
        self.model=None
        self.dict=None
        self.load_model()
    def pad(self,data):
        a=np.asarray(data)
        pad_data=[]
        
        max_len=130
        
        for mfcc in a:
          
          mfcc=mfcc[:max_len]
          mfcc=np.pad(mfcc,((0,max_len-len(mfcc)),(0,0)),mode='constant')
          pad_data.append(mfcc)      
          pad_data=np.asarray(pad_data)
        return(pad_data)

    def load_model(self):
        model = tf.keras.models.load_model('best_model.h5')
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']) 
        
        print("Model compiled, now you can talk ... ")
        
        
        dict={}
        file=open("dict",'rb')
        dict=pickle.load(file)
        file.close()
        dict2={dict[i]: i for i in dict}
        
        self.dict=dict2
        self.model=model
        self.graph = tf.get_default_graph()


    def translate(self,rec_path):
       
        

     
        #rec_path = os.path.join(os.path.expanduser('~'),'Desktop', 'Biip',str(recording))
        mfcc=mfcc(rec_path,numcep=20)
        l=[]
        l.append(mfcc[0])          
        l=self.pad(l)
        l=l.reshape(l.shape[0],l.shape[1],l.shape[2],1)
                
        predictions=self.model.predict(l)
                
        key=np.argmax(predictions[0])
        word=self.dict[key] 
        return(word)          
        
                    
          
            
          
         
    
    



