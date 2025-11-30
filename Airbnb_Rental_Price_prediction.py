#Importing libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
#Reading the files using pandas
df=pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Airbnb.csv')
df=pd.get_dummies(df,columns=['city','room_type'],drop_first=True)
#Y is the output to predict and x is the input 
y=df['price']
X=df.drop('price', axis=1)
#Splitting the data to 75:25 ratio
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7)
#Using RandomForest model
model=RandomForestRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
#Plotting the graph for the prediction price and original price
plt.scatter(y_test,y_pred,color='red')
plt.xlabel('Original Price')
plt.ylabel('Predicted Price')
plt.title('Prediction of the price on the original ')
plt.show()
