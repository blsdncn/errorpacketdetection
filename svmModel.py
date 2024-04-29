import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# load file
path = '/Users/jakevaldez/Downloads/archive/CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv'
data = pd.read_csv(path)

#selecting the columns that we want
dataColumns = ['duration', 'orig_bytes', 'resp_bytes','orig_pkts', 'orig_ip_bytes', 'resp_pkts','resp_ip_bytes']

# Select specific variables
x = data[dataColumns]
y = data['label'].map({'Malicious': 1, 'Benign': 0}) #converting to binary


# Split the data
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# fitted SVM model 
svm = SVC(kernel='linear', random_state=42, probability=True)
svm.fit(xTrain, yTrain)

# Calibrated to get probability values for log loss calculation
calibrated_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
calibrated_svm.fit(xTrain, yTrain)

# predictions and probability
yPrediction = calibrated_svm.predict(xTest)
yProbability = calibrated_svm.predict_proba(xTest)[:, 1]

# getting and displaying metrics
accuracy = accuracy_score(yTest, yPrediction)
precision = precision_score(yTest, yPrediction)
recall = recall_score(yTest, yPrediction)
log_loss_val = log_loss(yTest, yProbability)

print(f'\nModel Evaluation Metrics:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Log Loss: {log_loss_val:.2f}')

# Confusion Matrix
cm = confusion_matrix(yTest, yPrediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for SVM Model')
plt.show()
