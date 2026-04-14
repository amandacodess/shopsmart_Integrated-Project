# 🛍️ ShopSmart – E-Commerce Recommendation System

## 📌 Overview

**ShopSmart** is a full-stack E-Commerce web application built using Flask that combines **product browsing, intelligent recommendations, and business analytics** into a single platform.

Unlike basic frontend-only projects, this system integrates:

* Backend logic (Flask)
* Machine Learning-based recommendations
* Data analytics for admin insights

---

## 🎯 Key Features

### 🛒 Customer Interface

* Browse a wide range of products
* Search products using multiple algorithms
* Get **smart product recommendations**
* Clean, responsive UI with a light blue theme

---

### 🧠 Recommendation Engine

* **Content-Based Filtering (TF-IDF + Cosine Similarity)**
* **Hybrid Recommendation System (TF-IDF + SVD)**
* Handles **cold start problem**
* Suggests similar and relevant products dynamically

---

### 📊 Admin Dashboard

* Business insights and analytics
* Category distribution visualization
* Top-rated and trending products
* Price distribution analysis
* Summary statistics for decision-making

---

### ⚙️ Data Structures & Algorithms Used

* **Hash Map** → O(1) product lookup
* **Linear Search** → general search
* **Binary Search** → optimized rating-based filtering
* **Sorting (Timsort)** → ranking products

---

### 🧹 Data Processing

* Data cleaning and preprocessing
* Handling missing and inconsistent values
* Structured dataset for analysis and recommendations

---

## 🛠️ Tech Stack

### 💻 Backend

* Python (Flask)
* Pandas, NumPy
* Scikit-learn (TF-IDF, SVD, similarity models)

### 🎨 Frontend

* HTML, CSS, JavaScript
* Custom light-blue UI theme

### 📊 Concepts Applied

* Machine Learning
* Data Analytics
* Business Intelligence
* DSA integration in real-world application

---

## 📁 Project Structure

```bash
shopsmart/
│── app.py                  # Flask backend (routes + APIs)
│── generate_data.py        # Data preprocessing script
│── requirements.txt        # Dependencies
│── README.md               # Documentation

│── models/
│   └── clean_data.csv      # Dataset used for recommendations

│── templates/
│   ├── index.html          # User interface
│   └── main.html           # Admin dashboard

│── static/
│   ├── style.css           # Styling
│   ├── script.js           # Frontend logic & API calls
│   └── assets/             # Images (if any)
```

---

## 🚀 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/your-username/shopsmart.git
cd shopsmart
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app:

```bash
python app.py
```

4. Open in browser:

```bash
http://localhost:5000
```

---

## 🌐 Deployment

This project is a **Flask-based application**, so it requires a Python-supported hosting platform.

### ✅ Recommended Platforms:

* Railway
* Render
* PythonAnywhere

> Note: Static hosting platforms like GitHub Pages or Netlify will NOT work for this project since it requires a backend server.

---

## 🔮 Future Improvements

* 🔐 User authentication system
* 🤖 Advanced AI/ML recommendation models
* 🗄️ Database integration (MongoDB / MySQL)
* 📈 Interactive dashboards with charts
* ⚡ Real-time search and filtering
* ☁️ Cloud deployment optimization

---

## 💡 Learning Outcomes

* Building full-stack applications using Flask
* Implementing recommendation systems
* Applying DSA concepts in real-world scenarios
* Working with real datasets and analytics
* Understanding scalable system design

---

## 👨‍💻 Author

Developed as part of a Computer Science project focused on:

* Intelligent systems
* Data-driven applications
* E-Commerce architecture

---

## ⭐ Final Note

ShopSmart is more than just a UI project—it’s a **data-driven intelligent system**.
It demonstrates how machine learning, backend development, and analytics can come together to build real-world applications.

---
