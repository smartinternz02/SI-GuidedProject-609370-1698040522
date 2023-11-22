import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

# Read the dataset
df = pd.read_csv("Car Purchase prediction dataset.csv")

# Encode categorical variables using label encoding
lb_make = LabelEncoder()
df['purchased_car'] = lb_make.fit_transform(df['purchased_car'])
df['gender'] = lb_make.fit_transform(df['gender'])

# Select relevant features
features = ['age', 'estimated_salary_per_month', 'gender']  # Replace with the actual feature names
x = df[features]

# Scale the features
ms = MinMaxScaler()
x = pd.DataFrame(ms.fit_transform(x), columns=x.columns)

# Encode the target variable
y = df['purchased_car']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
SVC_model = SVC(kernel='rbf', random_state=0)
SVC_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = SVC_model.predict(x_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Save the model using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(SVC_model, model_file)
