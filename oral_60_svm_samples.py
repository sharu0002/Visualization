from google.colab import files
uploaded=files.upload()
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('oral_1_2_3_class.csv')
x=dataset.iloc[:,1:1025].values
# Separating out the target
y = dataset.loc[:,['target']].values

x
y

from sklearn.preprocessing import StandardScaler

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf

finalDf = pd.concat([principalDf, dataset[['target']]], axis = 1)
finalDf

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1,2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

finalDf.to_csv('foo.csv')

df = pd.read_csv('foo.csv')
X = df.iloc[:,[1,2]].values
Y = df.iloc[:,3].values

#splitting the dataset into the training set and test set (train split -function)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size = 0.35, random_state = 0)

#test_size = 0.35
#X_train
#X_test
#y_train
#y_test

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

#Making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm

# Transform to df for easier plotting
import seaborn as sns
cm_df = pd.DataFrame(cm,
                     index = ['Malignant','Normal','PreMalignant'], 
                     columns = ['Malignant','Normal','PreMalignant'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('SVM Classifier')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

cma=accuracy_score(y_test, y_pred)
cma

arr=np.array(cm)
TPa=int(arr[0,0])
TPb=int(arr[1,1])
TPc=int(arr[2,2])
Fab=int(arr[0,1])
Fac=int(arr[0,2])
Fba=int(arr[1,0])
Fbc=int(arr[1,2])
Fca=int(arr[2,0])
Fcb=int(arr[2,1])

Total=TPa+TPb+TPc+Fab+Fac+Fba+Fbc+Fca+Fcb
Total

accuracy=(TPa+TPb+TPc)/Total
accuracy

misclassification=(Fab+Fac+Fba+Fbc+Fca+Fcb)/Total
misclassification

precisionA=TPa/(TPa+Fba+Fca)
precisionA

precisionB=TPb/(TPb+Fab+Fcb)
precisionB

precisionC=TPc/(TPc+Fac+Fbc)
precisionC

sensitivityA=TPa/(TPa+Fab+Fac)
sensitivityA

sensitivityB=TPb/(TPb+Fba+Fbc)
sensitivityB

specificityA=(TPb+Fbc+Fcb+TPc)/(TPb+Fbc+Fcb+TPc+TPa+Fba)
specificityA

specificityB=(TPa+Fac+Fca+TPc)/(TPa+Fac+Fca+TPc+Fab+Fcb)
specificityB

specificityC=(TPa+Fab+Fba+TPb)/(TPa+Fab+Fba+TPb+Fac+Fbc)
specificityC

#visualizing the training set result
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1, step = 0.01),np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red','green','blue'))(i), label = j)
plt.title('SVM Classification(Training set)')
plt.xlabel('Principal Compoent Analysis 1')
plt.ylabel('Principal Compoent Analysis 2')
plt.legend()
plt.show()

#visualizing the testing set result
from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1, step = 0.01),np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red','green','blue'))(i), label = j)
plt.title('SVM Classification(Testing set)')
plt.xlabel('Principal Compoent Analysis 1')
plt.ylabel('Principal Compoent Analysis 2')
plt.legend()
plt.show()