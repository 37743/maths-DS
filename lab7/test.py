from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_actual = [1,0,1,0,1,0]
y_pred = [1,1,0,0,1,1]

cm = confusion_matrix(y_actual, y_pred)

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, linewidths=.5, square=True)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix', size = 15)
plt.show()