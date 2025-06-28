from setuptools import find_packages, setup

setup(
    name="heart-disease-classification",
    version="0.1.0",
    description="Heart disease classification project using scikit-learn and GridSearchCV.",
    author="Jose Navarro Meneses",
    author_email="josenavarrojmx@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "seaborn",
        "matplotlib",
        "joblib",
        "kagglehub",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
