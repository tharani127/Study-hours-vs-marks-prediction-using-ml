import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_csv("Study_hours.csv")
X=data[["HOURS"]]
Y=data["MARKS"]
model= LinearRegression()
model.fit(X,Y)
print("Model Trained successfully")
HOURS=[[6]]
predicted_marks=model.predict(HOURS)
print("If student stuided for 6 hours,predicted mark is:",predicted_marks[0])
plt.scatter(X,Y)
plt.plot(X,model.predict(X))
plt.xlabel("Study hours")
plt.ylabel("marks")
plt.show()