import pandas as pd
from PIL import Image
import numpy as np

dataset = pd.read_csv("fer2013.csv")


a=[]
emotions = ["angry","disgust","fear","happy","sad","surprise","neutral"]

for j in range(0,7):
    for i in range(0,len(dataset["Usage"])):
        # Change dataset["Usage"][i]=="Training" for generating training set images
        if(dataset["Usage"][i]!="Training" and dataset["emotion"][i]==j):
            pixels = dataset["pixels"][i].split(' ')
            m = 0
            test_list = []
            for x in range(48):
                for y in range(0,48):
                    a.append(int(pixels[m]))
                    m = m+1
                test_list.append(a)
                a = []
            array = np.array(test_list, dtype=np.uint8)
            new_image = Image.fromarray(array)
            # Save the training set images to "train" folder and test set to "test".
            new_image.save('test/'+emotions[j]+'/new'+str(i)+'.jpg')
