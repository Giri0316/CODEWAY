import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

file_path = "Churn_Modelling.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

X = df.drop(columns=['Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

new_customer_data = {
    'CreditScore': float(input("Enter Credit Score: ")),
    'Age': float(input("Enter Age: ")),
    'Tenure': float(input("Enter Tenure: ")),
    'Balance': float(input("Enter Balance: ")),
    'NumOfProducts': float(input("Enter Number of Products: ")),
    'HasCrCard': float(input("Enter Has Credit Card (1 for Yes, 0 for No): ")),
    'IsActiveMember': float(input("Enter IsActiveMember (1 for Yes, 0 for No): ")),
    'EstimatedSalary': float(input("Enter Estimated Salary: ")),
    'Geography_Germany': float(input("Enter 1 if in Germany, 0 otherwise: ")),
    'Geography_Spain': float(input("Enter 1 if in Spain, 0 otherwise: ")),
    'Gender_Male': float(input("Enter 1 for Male, 0 for Female: "))
}

new_customer_data_df = pd.DataFrame([new_customer_data])
new_customer_data_transformed = scaler.transform(new_customer_data_df)
prediction = model.predict(new_customer_data_transformed)

if prediction[0] == 1:
    print("This customer is likely to churn.")
else:
    print("This customer is not likely to churn.")
