import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import pickle as pkl
import util
#replace with your own path to test
#data will be the csvfile we open to read from the data for the model
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv')
output_dir = os.path.join(current_dir, '..','models')
data = pd.read_csv(data_path, delimiter='|')

# select the desired columns
dataColumns = ['duration', 'orig_bytes', 'resp_bytes','orig_pkts', 'orig_ip_bytes', 'resp_pkts','resp_ip_bytes']

# loop for the table dataColumns and change the data inside the columns to numeric type
# errors will be just a test case wich will make a faulty conversion be equal to '-'
#drop any rows with the '-' no # value
for dataC in dataColumns:
    data[dataC] = pd.to_numeric(data[dataC], errors='coerce')
data.dropna(inplace=True)

# print the first few rows of the data
# print summary statistics of the data
print(data.head())
print(data.describe())

# x = the inputted matrix
# y = the target label to map between Mal and Benign
x = data[dataColumns]
y = data['label'].map({'Malicious': 1, 'Benign': 0})

# split the data with training and testing
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

# scale the data so we can get the SD(standered deviation) and the mean
scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(xTest)

# create and train the LR(logistic regression) model with the desired iteration s
# fit the model to the scaled training data
model = LogisticRegression(max_iter=500000, solver='liblinear')
model.fit(xTrainScaled, yTrain)

#with open(os.path.join(output_dir,'lrSaved.pkl'),"wb") as file:
#    pkl.dump(model,file) 


util.save_model(model,os.path.join(output_dir,'svmSaved.pkl'))
predictions, probabilities, metrics = util.evaluate_model(model,xTestScaled,yTest)

# Predict/evaluate the model
#predictions = model.predict(xTestScaled)
#probabilities = model.predict_proba(xTestScaled)  # Needed for log loss

# Get the metrics so we can display it
#accuracy = accuracy_score(yTest, predictions)
#cm = confusion_matrix(yTest, predictions)
#cr = classification_report(yTest, predictions)
#loss = log_loss(yTest, probabilities)

print(f'\nModel Evaluation Metrics:')
print(f"Accuracy: {metrics['Accuracy']:.2f}")
print(f"Precision: {metrics['Precision']:.2f}")
print(f"Recall: {metrics['Recall']:.2f}")
print(f"Log Loss: {metrics['Log Loss']:.2f}")

util.plot_confusion_matrix(yTest,predictions)

# plotting convergence curve
# loss vs epochs
lossValues = []
for i in range(1, 101):
    model = LogisticRegression(max_iter=i, verbose=0, solver='liblinear')
    model.fit(xTrainScaled, yTrain)
    probabilities = model.predict_proba(xTestScaled)
    loss = log_loss(yTest, probabilities)
    lossValues.append(loss)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), lossValues, marker='o')
plt.title('Convergence Curve - Logistic Regression')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.grid()
plt.show()



