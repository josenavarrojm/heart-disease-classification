# 🤝 Contributing to Heart Disease Classification

First off, thank you for taking the time to contribute to this project! ❤️  
The following guidelines will help streamline the collaboration process.

---

## 📦 Project Overview

This repository contains a machine learning pipeline for classifying heart disease based on clinical data. It includes data preprocessing, model training, evaluation, testing, and visualization.

---

## 🛠 How to Contribute

### 1. Fork the Repository

Click the `Fork` button at the top-right corner of this repo, then clone your forked copy locally:

```bash
git clone https://github.com/<your-username>/heart-disease-classification.git
cd heart-disease-classification
```

### 2. Set Up the Environment

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 3. Create a Branch

Always create a new branch for your changes:

```bash
git checkout -b my-feature-branch
```

### 4. Make Your Changes

- Add new features, tests, or documentation
- Follow the project structure
- Write clean, readable, and commented code

### 5. Run Tests

Make sure everything passes:

```bash
pytest tests/
```

### 6. Commit and Push

```bash
git add .
git commit -m "Add: Description of your changes"
git push origin my-feature-branch
```

### 7. Open a Pull Request

Go to the original repository and open a new pull request describing your changes and why they are valuable.

---

## 📌 Contribution Rules

- Use descriptive commit messages.
- Make sure all scripts and notebooks run correctly.
- Keep pull requests focused; avoid bundling unrelated changes.
- Update or add documentation where needed.

---

## 🧪 Test Coverage

When possible, add unit tests for your contributions in the `tests/` folder.

---

## 🙏 Thank You

Your ideas, code, and time are appreciated. This project thrives thanks to community contributions!