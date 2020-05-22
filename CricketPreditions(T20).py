import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
results = pd.read_csv('C:/Users/rahul.DESKTOP-PUQD80G/Desktop/DataSets/T20.csv')

final = pd.get_dummies(results, prefix=['team', 'team2'], columns=['team', 'team2'])

X = final.drop(['winner','toss_winner','city','venue'] ,axis=1)
y = final["winner"]
a=results["toss_winner"]
b=results["venue"]
X=X.fillna(0)
y=y.fillna("Looser")
le=LabelEncoder()
le.fit(y)
y=le.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train.head()


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=0) 
rf.fit(X_train, y_train)
C=rf.predict(X_test)
score = rf.score(X_train, y_train)
score2 = rf.score(X_test, y_test)

j=rf.predict(X_test)


le.fit(a)
a=le.transform(a)

a=a.reshape(-1,1)


le.fit(b)
b=le.transform(b)
b=b.reshape(-1,1)

a_train, a_test, y_train, y_test = train_test_split(a, y, test_size=0.30, random_state=42)
b_train, b_test, y_train, y_test = train_test_split(b, y, test_size=0.30, random_state=42)
a_train

tree = DecisionTreeClassifier(random_state=0)
tree.fit(a_train, y_train)
sc=tree.predict(a_test)
sco=tree.score(a_test,y_test)
print(sco)


tree.fit(b_train,y_train)
ab=tree.predict(b_test)
scoo=tree.score(b_test,y_test)
print(scoo)

p=[]
for i in range (0,len(j)):
    if(j[i]==sc[i] and j[i]==ab[i]):
        p.append(j[i])
    elif(j[i]==sc[i] and j[i]!=ab[i]):
        p.append(j[i])
    
    elif(j[i]==ab[i] and j[i]!=sc[i]):
        p.append(j[i])
  
    elif(ab[i]==sc[i] and ab[i]!=j[i]):
        p.append(ab[i])
    else:
        p.append(j[i])
arr= np.array(p) 
arr

from sklearn.metrics import accuracy_score
scooo=accuracy_score(y_test,arr)
print(scooo)







