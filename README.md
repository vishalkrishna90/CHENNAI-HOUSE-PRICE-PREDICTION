
# Chennai House Price Preciction 

This is a Chennai House Price Prediction project in which I tried to build Machine Learning Model very efficiently by that anyone can predict house prices in Chennai through the Chennai House Price Prediction Web App. 
[Go The Web App](https://chennaihousepricepredict.herokuapp.com/)

![Chennai House](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Chennai_img_1.jpg)
## Process Followed To Complite This Project
- Problem Statement
- Data Collection 
- Data Description
- Data Preprocessing
- Handle Outliers 
- Exploratory Data Analysis (EDA)
- Data Encoding
- Feature Selection
- Data Scaling
- Model Building
- Model Performances & Feature Importance
- Rebuild Model With Imp Features
- Final Score By The Best Model
- Make Pickle File
- Create New Enviornment
- Create Web App With Streamlit
- Upload All Files In Github repository
- Deploy Model On Heroku

**Web App Overview**

![Chennai House Web App](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Web_App_2.png)
## Problem Statement

Real estate transactions are quite opaque sometimes and it may be difficult for a newbie to know the fair price of any given home. Thus, multiple real estate websites have the functionality to predict the prices of houses given different features regarding it. Such forecasting models will help buyers to identify a fair price for the home and also give insights to sellers as to how to build homes that fetch them more money. Chennai house sale price data is shared here and the participants are expected to build a sale price prediction model that will aid the customers to find a fair price for their homes and also help the sellers understand what factors are fetching more money for the houses?
## Data Collection
I got this dataset from Kaggle, first I downloaded this dataset from Kaggle to
my local storage and then imported in jupyter notebook

```
df = pd.read_csv('Chennai houseing sale.csv')

```
**Data Overview**

![DataFrame Overview 1](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Overview_1.png)
![DataFrame Overview 2](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Overview_2.png)
## Data Description

The Chennai dataset has 7109 rows and 22 columns. This data frame contains the following columns:

PRT_ID - 
Id of house

AREA - 
In which area house is located in Chennai

INT_SQFT -
Area in sqft


DATE_SALE - 
When house was sold

DIST_MAINROAD - 
Distance of house from main road

N_BEDROOM -
Number of Bedrooms

N_BATHROOM -
Number of Bathrooms

N_ROOM - 
Number of Rooms

SALE_COND - 
Sale condition

PARK_FACIL - 
Is parking available or not

DATE_BUILD -
Date house was built

BUILDTYPE - 
Purpose of house

UTILITY_AVAIL -
Facilities available there

STREET - 
How is street outside that house

MZZONE - 
Which zone it is in

QS_ROOMS -
It is masked data

QS_BATHROOM - 
It is masked data

QS_BEDROOM - 
It is masked data

QS_OVERALL - 
It is masked data

REG_FEE - 
Registration fee after sales

COMMIS - 
Commission fee after sales

SALES_PRICE - 
Sale price of house


## Data Preprocessing
In this step first I checked there are any null and duplicate values present or not, and I found there are some null values present, I dropped them and there were no duplicate values after that checked there are any incorrect data present or not, and I found much amount of wrong data was there and I corrected them.

![Data Preprocessing 1](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Preprocess_1.png)
![Data Preprocessing 2](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Preprocess_2.png)
![Data Preprocessing 3](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Preprocess_3.png)
![Data Preprocessing 4](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Preprocess_4.png)
![Data Preprocessing 5](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Preprocess_5.png)
![Data Preprocessing 6](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Preprocess_6.png)

## Handle outliers
After correcting incorrect data, I checked whether there are any outliers present or not, and I found that there are some amount of outliers present, for more clarification, I used the IQR method to check outliers and I found that there are no outliers.

![Outliers 1](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Outliers_1.png)
![Outliers 2](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Outliers_2.png)
![Outliers 3](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Outliers_3.png)
![Outliers 4](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Outliers_4.png)



## Exploratory Data Analysis (EDA)
In this step I tried to Analyze the data very efficiently and deeply, first I checked correlation between features, 
then check feature distribution and then relation between features and target by graph

![EDA 1](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/EDA_1.png)
![EDA 2](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/EDA_2.png)
![EDA 3](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/EDA_3.png)
![EDA 4](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/EDA_4.png)
![EDA 5](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/EDA_5.png)

## Data Encoding

After EDA I did data encoding by label and on hot encoder and create a new data frame for the next process

![Data Encoding](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Data_Encoding.png)

## Feature Selection
In this step first I splits features and target, then splits data into train
and test data

```
# split data into features and target
X  = df3.drop('SALES_PRICE',axis = 1)
y = df3['SALES_PRICE']
```

```
# import train test split to split train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)
```

## Data Scaling

In this step, I scaled or features train and test data 

```
# import standard scaler to scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

```

## Model Building
In this step, I build different - different models and checked their performance, and then chose the best model for the dataset

![Model 1](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Model_1.png)
![Model 1_1](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Model_1_1.png)
![Model 2](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Model_2.png)
![Model 2_2](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Model_2_2.png)
![Model 3](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Model_3.png)
![Model 3_3](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Model_3_3.png)


## Model Performances & Feature Importance
After model Building, I checked their performance by r2 score and the model whose r2 score was higher I consider that model to be a final model and then I checked important features based on that model 

![Model Performance](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Model_Performance.png)
![Feature Importance](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Feature_Importance.png)


## Rebuild Model With Important Features

After getting feature Importance I created a new data frame by Important features and then did data splitting and data scaling and then rebuild the model by important features

```
new_df = df3[['INT_SQFT','MZZONE', 'N_ROOM','AREA','PARK_FACIL', 'STREET','BUILDTYPE_Others','BUILDTYPE_House','SALES_PRICE']]
```

```
X  = new_df.drop('SALES_PRICE',axis = 1)
y = new_df['SALES_PRICE']
```

```
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)
```

```
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

```

## Final Score By The Best Model

![Final Score](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Final_Score.png)

## Make Pickle File
After getting the final score from the best model I created a Pickle file for the web app

```
import pickle as pkl
pkl.dump(grid_xgb, (open('xgb_model.pkl','wb')))
```

## Create New Enviornment
After making pickle file I created new virtual environment for the 
project and install required libraries and created web app in VS Code IDE

```
conda create -p chennaihouseprice python==3.9 -y
```

```
pip install streamlit numpy pandas sklearn xgboost
``` 

## Create Web App With Streamlit
After installing all required libraries and dependencies I created web app 

![Web App 2](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Web_App_2.png)

## Upload All Files In Github repository

After creating web app I uploaded all files in github repository by git CLI
```
git config --global user.name "FIRST_NAME LAST_NAME"
```

```
git config --global user.email "myemail@gmail.com"
```

```
git add files_name
```

```
git commit -m  "about the commit"
```

```
git push origin main
```

## Deploy Model On Heroku

In the end, I deployed the model on Heroku, so that anybody can use the web app

[Chennai House Price Prediction Web App](https://chennaihousepricepredict.herokuapp.com/)

![Chennai House Price Precictin](https://github.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/blob/main/Images/Web_App_img.png)
## Deployment Requirement Tools 

 - [Streamlit](https://streamlit.io/)
 - [Github Account](https://github.com/)
 - [Heroku Account](https://dashboard.heroku.com/apps)
 - [Visual Studio Code](https://code.visualstudio.com/)
 - [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)



## Challenges

- Much amount of wrong data was present and I had to correct them one by one
- Based on data there was some amount of outliers present but when I applied some approaches I found that are not outliers
- There were many categorical values present and I had to encode them one by one
- Creating web app and deployment on Heroku