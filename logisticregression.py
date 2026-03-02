# implement LogisticRegression algorithm for stocks prices prediction

from multiprocessing.util import info
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import yfinance as yf   


stock = yf.download('AAPL', start = "2020-01-01", end = "2026-01-01")          

print(f"type:{type(stock)}")
print(f"head:{stock.head()}")
print(f"columns:{stock.columns}")

print(f"no.of.columns:{len(stock.columns)}")
print(f"info:{stock.info()}")


#-----------------------------#
# step-2: FeatureEngineering
#-----------------------------#

stock['Returns'] = stock['Close'].pct_change()
stock['sma_5'] = stock['Close'].rolling(5).mean()
stock['sma_20'] = stock['Close'].rolling(20).mean()
stock['sma_50'] = stock['Close'].rolling(50).mean()
stock['volatility'] = stock['Returns'].rolling(20).std()

#-----------------------------#
print(f"info:{stock.info()}")


stock.dropna(inplace=True)
print(f"info:{stock.info()}")

#--------------------------------------------#
# Target-1: if tomorrows price > todays price 
#--------------------------------------------#

stock['Target'] = (stock['Close'].shift(-1) > stock['Close']).astype(int)
print(stock.head())


#-------------------------------#
# Prepare the date for modeling
#-------------------------------#

features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Returns', 'sma_5', 'sma_20', 'sma_50', 'volatility']
print(stock[features].head())

features1 = stock.columns
print(features1)

X = stock[features1]
X = X.drop('Target', axis=1)
y = stock['Target'].values

x_train, x_test, y_train, y_test = train_test_split( X,y, test_size=0.2, random_state=42)

# Build the model 
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("\n" + "="*50)
print("Classification Report:")
print("="*50)
print(classification_report(y_test, y_pred))

print("\n" + "="*50)
print("Confusion Matrix:")
print("="*50)
print(confusion_matrix(y_test, y_pred))

print("\n" + "="*50)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("="*50)