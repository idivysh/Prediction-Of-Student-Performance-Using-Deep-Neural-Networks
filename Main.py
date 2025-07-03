

#medium = 2, Low = 1, High = 0 l,h,h,m,l,h

from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam


main = tkinter.Tk()
main.title("Predicting Student Performance with Deep Neural Networks") #designing main screen
main.geometry("1300x1200")

global filename
X = []
Y = []
global X_train, X_test, Y_train, Y_test
global model
global knn_acc,nb_acc,dt_acc,rf_acc,lr_acc,sv_acc,nn_acc
global train

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def Preprocessing():
    train = pd.read_csv(filename)
    le = LabelEncoder()
    train['gender'] = pd.Series(le.fit_transform(train['gender']))
    train['NationalITy'] = pd.Series(le.fit_transform(train['NationalITy']))
    train['PlaceofBirth'] = pd.Series(le.fit_transform(train['PlaceofBirth']))
    train['StageID'] = pd.Series(le.fit_transform(train['StageID']))
    train['GradeID'] = pd.Series(le.fit_transform(train['GradeID']))
    train['SectionID'] = pd.Series(le.fit_transform(train['SectionID']))
    train['Topic'] = pd.Series(le.fit_transform(train['Topic']))
    train['Semester'] = pd.Series(le.fit_transform(train['Semester']))
    train['Relation'] = pd.Series(le.fit_transform(train['Relation']))

    train['ParentAnsweringSurvey'] = pd.Series(le.fit_transform(train['ParentAnsweringSurvey']))
    train['ParentschoolSatisfaction'] = pd.Series(le.fit_transform(train['ParentschoolSatisfaction']))
    train['StudentAbsenceDays'] = pd.Series(le.fit_transform(train['StudentAbsenceDays']))
    train['Class'] = pd.Series(le.fit_transform(train['Class']))
    train.to_csv('transform.csv', index=False)
    text.insert(END,"\nPreprocessing & Data Transformation Completed. All transform data saved inside transform.csv file");

def generateModel():
    text.delete('1.0', END)
    global train
    global X, Y
    global X_train, X_test, Y_train, Y_test
    X.clear()
    Y.clear()

    train = pd.read_csv('transform.csv')
    X = train.values[:, 0:16] 
    Y = train.values[:, 16]
    print(X)
    print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    text.insert(END,'Total Dataset Size  : '+str(len(X))+"\n")
    text.insert(END,'Total splitted records used for training : '+str(len(X_train))+"\n")
    text.insert(END,'Total splitted records used for testing  : '+str(len(X_test))+"\n") 
    
def KNN():
    text.insert(END,"\n")
    global knn_acc
    knn = KNeighborsClassifier() 
    knn.fit(X_train, Y_train) 
    prediction_data = knn.predict(X_test)
    knn_acc = accuracy_score(Y_test,prediction_data)*100
    text.insert(END,"KNN Accuracy : "+str(knn_acc)+"\n")

def NaiveBayes():
    global nb_acc
    nb = BernoulliNB()
    nb.fit(X_train, Y_train)
    prediction_data = nb.predict(X_test)
    nb_acc = accuracy_score(Y_test,prediction_data)*100
    text.insert(END,"Naive Bayes Accuracy : "+str(nb_acc)+"\n")

def DecisionTree():
    global dt_acc
    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    prediction_data = dt.predict(X_test)
    dt_acc = accuracy_score(Y_test,prediction_data)*100
    text.insert(END,"Decision Tree Accuracy : "+str(dt_acc)+"\n")


def RandomForest():
    global rf_acc
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    prediction_data = rf.predict(X_test)
    rf_acc = accuracy_score(Y_test,prediction_data)*100
    text.insert(END,"Random Forest Accuracy : "+str(rf_acc)+"\n")


def Logistic():
    global lr_acc
    lr = LogisticRegression(max_iter=2000,dual=False)
    lr.fit(X_train, Y_train)
    prediction_data = lr.predict(X_test)
    lr_acc = accuracy_score(Y_test,prediction_data)*100
    text.insert(END,"Logistic Regression Accuracy : "+str(lr_acc)+"\n")


def SupportVector():
    global sv_acc
    sv = svm.SVC()
    sv.fit(X_train, Y_train)
    prediction_data = sv.predict(X_test)
    sv_acc = accuracy_score(Y_test,prediction_data)*100
    text.insert(END,"SVM Accuracy : "+str(sv_acc)+"\n")


def NeuralNetworks():
    global nn_acc
    global model
    X1 = train.values[:, 0:16] 
    y_= train.values[:, 16]
    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y1 = encoder.fit_transform(y_)
    print(Y1)
    train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, Y1, test_size=0.2)
    model = Sequential()
    model.add(Dense(64, input_shape=(16,), activation='relu', name='fc1'))
    model.add(Dense(32, activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(train_x1, train_y1, verbose=1, batch_size=5, epochs=100)
    accuracy = hist.history['accuracy']
    nn_acc = accuracy[99] * 100
    text.insert(END,"Neural Networks Accuracy : "+str(nn_acc)+"\n\n")
    text.insert(END,'Neural Network Model Summary. See black console for NN layer details\n')
    print(model.summary())



def predictPerformance():
    text.delete('1.0', END)
    testfile = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(testfile)
    test1 = pd.read_csv(testfile)
    le = LabelEncoder()
    test['gender'] = pd.Series(le.fit_transform(test['gender']))
    test['NationalITy'] = pd.Series(le.fit_transform(test['NationalITy']))
    test['PlaceofBirth'] = pd.Series(le.fit_transform(test['PlaceofBirth']))
    test['StageID'] = pd.Series(le.fit_transform(test['StageID']))
    test['GradeID'] = pd.Series(le.fit_transform(test['GradeID']))
    test['SectionID'] = pd.Series(le.fit_transform(test['SectionID']))
    test['Topic'] = pd.Series(le.fit_transform(test['Topic']))
    test['Semester'] = pd.Series(le.fit_transform(test['Semester']))
    test['Relation'] = pd.Series(le.fit_transform(test['Relation']))

    test['ParentAnsweringSurvey'] = pd.Series(le.fit_transform(test['ParentAnsweringSurvey']))
    test['ParentschoolSatisfaction'] = pd.Series(le.fit_transform(test['ParentschoolSatisfaction']))
    test['StudentAbsenceDays'] = pd.Series(le.fit_transform(test['StudentAbsenceDays']))
    testx = test.values[:, 0:16]
    test1 = test1.values[:, 0:16]
    predict = model.predict(testx)
    msg = ''
    for i in range(len(test)):
        performance = np.argmax(predict[i])
        if performance == 0:
            msg = 'HIGH'
        if performance == 1:
            msg = 'LOW'
        if performance == 2:
            msg = 'MEDIUM'
        text.insert(END,str(test1[i])+" Predicted Performance As : "+msg+"\n")
        

def graph():
    height = [knn_acc,nb_acc,dt_acc,rf_acc,lr_acc,sv_acc,nn_acc]
    bars = ('KNN ACC','NB ACC','DT ACC','RF ACC','LR ACC','SVM ACC','NN ACC')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()       

font = ('times', 16, 'bold')
title = Label(main, text='Predicting Student Performance with Deep Neural Networks')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Student Performance Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Preprocessing & Data Transformation", command=Preprocessing)
preButton.place(x=340,y=100)
preButton.config(font=font1) 

modelButton = Button(main, text="Split Dataset into Train & Test Model", command=generateModel)
modelButton.place(x=640,y=100)
modelButton.config(font=font1) 

knnButton = Button(main, text="Run KNN Algorithm", command=KNN)
knnButton.place(x=50,y=150)
knnButton.config(font=font1) 

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=NaiveBayes)
nbButton.place(x=340,y=150)
nbButton.config(font=font1)

dtButton = Button(main, text="Run Decision Tree Algorithm", command=DecisionTree)
dtButton.place(x=640,y=150)
dtButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithm", command=RandomForest)
rfButton.place(x=50,y=200)
rfButton.config(font=font1)

lrButton = Button(main, text="Run Logistic Regression Algorithm", command=Logistic)
lrButton.place(x=340,y=200)
lrButton.config(font=font1)

svButton = Button(main, text="Run SVM Algorithm", command=SupportVector)
svButton.place(x=640,y=200)
svButton.config(font=font1)

nnButton = Button(main, text="Run Neural Networks Algorithm", command=NeuralNetworks)
nnButton.place(x=50,y=250)
nnButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=340,y=250)
graphButton.config(font=font1)

predictButton = Button(main, text="Upload & Predict New Student Performance", command=predictPerformance)
predictButton.place(x=640,y=250)
predictButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
