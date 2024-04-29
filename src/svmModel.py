import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import pickle as pkl
import os
import util

# Load file with the correct delimiter
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv')
output_dir = os.path.join(current_dir, '..','models')
data = pd.read_csv(data_path, delimiter='|')

# Print the column names to ensure they are correct
print("Columns in the data:", data.columns)

# Selecting the columns that we want
dataColumns = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']

# Convert non-numeric entries to numeric and handle non-convertible values
for column in dataColumns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
    data[column] = data[column].fillna(0)  # Replace NaNs with 0

data.dropna(inplace=True)

# Select specific variables
x = data[dataColumns]
y = data['label'].map({'Malicious': 1, 'Benign': 0})  # Converting to binary

# Split the data
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# Fitted SVM model
 
svm = LinearSVC(random_state=42, dual='auto', tol=1e-3, max_iter=1000)

svm.fit(xTrain, yTrain)

# Calibrated to get probability values for log loss calculation
calibrated_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
calibrated_svm.fit(xTrain, yTrain)
model = calibrated_svm

#with open(os.path.join(output_dir,'svmSaved.pkl'),"wb") as file:
#    pkl.dump(model,file)

util.save_model(model,os.path.join(output_dir,'svmSaved.pkl'))

# Predictions and probability
yPrediction, yProbability, metrics = util.evaluate_model(model,xTest,yTest)

#yPrediction = calibrated_svm.predict(xTest)
#yProbability = calibrated_svm.predict_proba(xTest)[:, 1]

# Getting and displaying metrics
#accuracy = accuracy_score(yTest, yPrediction)
#precision = precision_score(yTest, yPrediction)
#recall = recall_score(yTest, yPrediction)
#log_loss_val = log_loss(yTest, yProbability)

print(f'\nModel Evaluation Metrics:')
print(f"Accuracy: {metrics['Accuracy']:.2f}")
print(f"Precision: {metrics['Precision']:.2f}")
print(f"Recall: {metrics['Recall']:.2f}")
print(f"Log Loss: {metrics['Log Loss']:.2f}")

# Confusion Matrix
util.plot_confusion_matrix(yTest,yPrediction)
#cm = confusion_matrix(yTest, yPrediction)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot()
#plt.title('Confusion Matrix for SVM Model')
#plt.show()
