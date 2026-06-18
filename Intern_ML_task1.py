# libraries importing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#import model for decision tree
from sklearn.tree import DecisionTreeRegressor
#import evaluation metrics
from sklearn.metrics import r2_score,mean_squared_error

# Load Dataset
df = pd.read_csv("Dataset .csv")
print(df.head())
print(df.shape)
print(df.columns)
# check for  missing values
print(df.isnull().sum())


#Handle missing values
#fill missing values in cuisines with most frequent value
df['Cuisines']=df['Cuisines'].fillna(df['Cuisines'].mode()[0])

#verify again
print(df.isnull().sum())

# Remove useless columns

df = df.drop(['Restaurant ID',
              'Restaurant Name',
              'Address',
              'Locality',
              'Locality Verbose'],axis=1)
print(df.shape)
#step 7 check data type
print(df.dtypes)

# Drop leakage columns

df =df.drop(['Rating color','Rating text'],axis=1,errors='ignore')
print(df.shape)
print(df.dtypes)

#  Encoding(label encoding)
#process 1 import the library sklearn.preprocessing from sklearn.precessing import label encoder
#process 2 create an encoder object
le=LabelEncoder()
#process 3 choose which column to encode
categorical_cols=['City',
                  'Cuisines',
                  'Currency',
                  'Has Table booking',
                  'Has Online delivery',
                  'Is delivering now',
                  'Switch to order menu'
                  ]
#process 4 Encode each column
for col in categorical_cols:
    df[col]=le.fit_transform(df[col])
#process 5 verify the data type
print(df.dtypes)
#TRAIN - TEST - SPLIT
# separate features and Target
x = df.drop('Aggregate rating',axis=1)
y=df['Aggregate rating']

print(x.shape)
print(y.shape)

# train-test split
#import the library sklearn.model_selection import train_test_split
#process 2 Actual split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
#process 3 verify split
print(x_train.shape)
print(x_test.shape)

#Implementation of Decision Tree
#process 1 Import Decision Tree model
#process 2 Create the model
dt=DecisionTreeRegressor(random_state=42)
#process 3 Train the model
dt.fit(x_train,y_train)
#process 4 Make predictions
y_pred=dt.predict(x_test)
#process 5 check predictions
print("Actual values:",y_test.values[:5])
print("predicted values:",y_pred[:5])


#Model evaluation
# process 1 import evaluation metrics
#process 2 calculate metrics
r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)

print("R2 Score:",r2)
print(("MSE:",mse))


