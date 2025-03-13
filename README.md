### Deep-Learning-Challenge

# 🏆 Charity Donation Success Prediction  

## 📌 Overview  
This project builds and trains a **deep learning model** using **TensorFlow** to predict charity donation success. The dataset is preprocessed, categorical variables are encoded, and a **neural network (DNN)** is trained for classification.

---

## 🚀 Quick Start (Google Colab)  
1. Open **[Google Colab](https://colab.research.google.com/)**  
2. Run:  
   ```python
   !wget https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv
   !pip install tensorflow pandas scikit-learn numpy
   ```
3. Copy & paste the code below to load and train the model.

---

## 📊 Steps Covered  

### **1️⃣ Data Preprocessing**  
- Dropped non-beneficial columns (`EIN`, `NAME`)  
- Encoded categorical variables (`APPLICATION_TYPE`, `CLASSIFICATION`)  
- Converted categorical data to numerical using `pd.get_dummies()`  
- Split dataset into **training & testing sets**  

### **2️⃣ Building & Training the Model**  
- Defined a **sequential neural network** with:  
  - **Two hidden layers** (`ReLU` activation)  
  - **Output layer** (`Sigmoid` activation for binary classification)  
- Compiled with `adam` optimizer & `binary_crossentropy` loss function  
- Trained the model for **50 epochs**  

### **3️⃣ Saving & Exporting the Model**  
- Saved as an **HDF5 file (`.h5`)** for future use  

---

## 🔥 How to Run in Google Colab  

### **1️⃣ Load & Preprocess Data**  
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("charity_data.csv")

# Drop unnecessary columns
df = df.drop(columns=['EIN', 'NAME'])

# Convert categorical variables
df = pd.get_dummies(df)

# Split features (X) and target (y)
X = df.drop(columns=['IS_SUCCESSFUL'])
y = df['IS_SUCCESSFUL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data types
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_train = np.asarray(y_train).astype('int32')
y_test = np.asarray(y_test).astype('int32')
```

### **2️⃣ Define & Train the Model**  
```python
# Define neural network
nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(80, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = nn.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```

### **3️⃣ Save the Model**  
```python
nn.save("charity_optimization_model.h5")
```

---

## 📌 Next Steps  
- **Optimize model performance** (Hyperparameter tuning)  
- **Feature selection & engineering**  
- **Deploy as an API or web app**  

---

## 📧 Contact & Contributions  
Feel free to contribute or reach out with suggestions!  
📩 **Email**: your.email@example.com  
💡 **GitHub**: [Your Repo Link]  

---

### 🎉 Happy Coding & Machine Learning! 🚀

