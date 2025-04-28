import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import os

# Ensure output folder exists
out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv('Sales_Dataset.csv')  # Make sure Sales_Dataset.csv is in the same folder

# See the data
print("Dataset Preview:")
print(data.head())

# Features and Target
X = data[['Sales', 'Profit']]      # Features
Y = data['Category']               # Target

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Evaluate
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# ------------------ PLOTS ------------------

# 1) Scatter plot
fig1 = plt.figure(figsize=(8,6))
sns.scatterplot(x='Sales', y='Profit', hue='Category', data=data, palette='Set1')
plt.title('Sales vs Profit Scatter Plot')
plt.grid(True)
scatter_path = os.path.join(out_dir, 'scatter_plot.png')
plt.savefig(scatter_path, dpi=300)
print(f"Saved scatter plot to: {scatter_path}")
plt.show()

# 2) Confusion matrix heatmap
fig2 = plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
cm_path = os.path.join(out_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
print(f"Saved confusion matrix to: {cm_path}")
plt.show()

# 3) Decision Tree visualization
fig3 = plt.figure(figsize=(12,8))
plot_tree(model, feature_names=['Sales', 'Profit'], class_names=model.classes_, filled=True)
plt.title('Decision Tree')
tree_path = os.path.join(out_dir, 'decision_tree.png')
plt.savefig(tree_path, dpi=300)
print(f"Saved decision tree to: {tree_path}")
plt.show()
