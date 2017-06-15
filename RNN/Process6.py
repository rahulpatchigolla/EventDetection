from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
fp=open("bestTestPredictionsArguements.txt","r")
actual = []
predicted = []
for line in fp:
    if not line=="\n":
        line =line.strip('\n').split(' ')
        actual.append(line[1])
        predicted.append(line[2])
# This file gives the final F1-Score of the argument identification results  and plots the confusion matrix also
x=precision_score(actual,predicted,labels=['AtLoc','Cause','FromLoc','Instrument','Theme','ToLoc'],average='micro')
y=recall_score(actual,predicted,labels=['AtLoc','Cause','FromLoc','Instrument','Theme','ToLoc'],average='micro')

z=f1_score(actual,predicted,labels=['AtLoc','Cause','FromLoc','Instrument','Theme','ToLoc'],average='micro')
print x,y,z
matrix=confusion_matrix(actual,predicted,labels=['AtLoc','Cause','FromLoc','Instrument','Theme','ToLoc','No'])
labels=['AtLoc','Cause','FromLoc','Instrument','Theme','ToLoc','No']
print matrix
df_cm = pd.DataFrame(matrix,index = [i for i in labels],columns=[i for i in labels])
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
plt.figure(figsize = (19,12))
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True,fmt=".0f",annot_kws={"size":20},linewidths=1,linecolor="Black",robust=True)
plt.savefig('confusion_matrix_arguement.png', format='png')
