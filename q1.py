import pandas as  pd
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate
data={"age":[20,25,30,35,40,45],
      "salary":[10000,20000,50000,30000,35000,40000],
      "buy_car":[1,1,0,0,1,1]}
dp=pd.DataFrame(data)
x=dp[["age","salary"]]
y=dp["buy_car"]
model=GaussianNB()
model.fit(x,y)
classes=model.classes_
priors=model.class_prior_
table_data=[]
for c,p in zip(classes,priors):
    table_data.append([f"target({c})",f"{p:.4f}"])
print(tabulate(table_data,headers=["target","chi(phi)"],tablefmt="grid"))  
