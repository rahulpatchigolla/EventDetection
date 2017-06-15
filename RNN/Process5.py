from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# This file gives the final F1-Score of the trigger identification results by considering the false negatives and plots the confusion matrix also
fp=open("bestTestPredictionsTrigger.txt","r")
actual = []
predicted = []
for line in fp:
    if not line=="\n":
        line =line.strip('\n').split(' ')
        actual.append(line[1])
        predicted.append(line[2])

rareEvents = {"Phosphorylation": 3, "Synthesis": 4, "Transcription": 7, "Catabolism": 4,"Dephosphorylation":1, "Remodeling": 10}
for key in rareEvents.keys():
    #print key
    actualval=rareEvents[key]
    takenval=0
    diff=actualval-takenval
    while diff>0:
        diff-=1
        actual.append(key)
        predicted.append('Other')
fp.close()
x=precision_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development',
                                                         'Localization','Binding','Gene_expression','Regulation','Positive_regulation',
                                                         'Negative_regulation','Planned_process','Phosphorylation', 'Synthesis', 'Transcription', 'Catabolism','Dephosphorylation','Remodeling'],average='micro')
y=recall_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development',
                                                         'Localization','Binding','Gene_expression','Regulation','Positive_regulation',
                                                         'Negative_regulation','Planned_process','Phosphorylation', 'Synthesis', 'Transcription', 'Catabolism','Dephosphorylation','Remodeling'],average='micro')

z=f1_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development',
                                                         'Localization','Binding','Gene_expression','Regulation','Positive_regulation',
                                                         'Negative_regulation','Planned_process','Phosphorylation', 'Synthesis', 'Transcription', 'Catabolism','Dephosphorylation','Remodeling'],average='micro')
print x,y,z
matrix=confusion_matrix(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development',
                                                         'Localization','Binding','Gene_expression','Regulation','Positive_regulation',
                                                         'Negative_regulation','Planned_process','Phosphorylation', 'Synthesis', 'Transcription', 'Catabolism','Dephosphorylation','Remodeling','Other'])

labels=['DEV','GRO','BRK','DTH','CellP','BVD','LOC','BIND','GENEXP','REG','PREG','NREG','PLP','PHO', 'SYN', 'TRANS', 'CATA','DEPHO','REMDL','OTH']
print matrix
df_cm = pd.DataFrame(matrix,index = [i for i in labels],columns=[i for i in labels])
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
plt.figure(figsize = (19,12))
sn.set(font_scale=0.8)
sn.heatmap(df_cm, annot=True,fmt=".0f",annot_kws={"size":20},linewidths=1,linecolor="Black",robust=True)
plt.savefig('confusion_matrix_trigger.png', format='png')

