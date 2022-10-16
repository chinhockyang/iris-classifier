# Iris Data Analysis
Chin Hock Yang

## About Project
This project aims to perform data analysis on Iris flower classes in a garden and to create a program that is capable of making predictions of a new Iris flower. The project repository contains of 2 main files, ```analysis.ipynb``` and ```app.py```.

### analysis.ipynb
This Jupyter Notebook assesses the quality of the Iris data, and constructs Supervised Machine Learning models such as Decision Tree, Logistic Regression and K-Nearest Neighbours models that are capable of making predictions on a new Iris flower data. It also contains codes that perform Unsupervised Learning such as identifying data clusters using K-Means Clustering and using Cosine-Similarity to measure the "similarity" between data points. A Logistic Regression model is generated and stored in the ```model``` folder within this notebook, which will be used by the programme created in ```app.py```.

### app.py
This Python script contains a setup of a Plotly Dash dashboard application. There are 3 main sections in this application.
| Section | Description |
| :------------- |:-------------|
| Data Exploration | View the proportion of the Iris classes, and inspect for missing data in the dataset |
| Comparison across Iris Types | Visualise the distribution and differences of the various attributes and characteristics of the Iris types |
| New Data Prediction | Provide value of the attributes of the new Iris data, and view the predicted Iris type and 10 most similar existing Iris data to the input Iris data |
 
The application has been deployed (temporary) on https://hy-garden-dashboard.herokuapp.com/ for quick access and demonstration.
 
## Environment Setup
This project uses local Virtual Environment to manage project dependencies. [(environment creation and activation guide)](https://docs.python.org/3/tutorial/venv.html). To setup the repository, Python (version 3.8 and above) and Pip (package installer) will need to be first installed on the local machine.

##### Steps to setup Virtual Environment
1. In the project root directory, run ```python -m venv venv``` to create a virtual environment folder ```venv``` at the root level of the repository
2. Activate the environment by running ```source venv/bin/activate``` (for Mac Users) or ```venv\Scripts\activate``` (for Windows Users)
3. Run the command ```pip install -r requirements.txt``` after activating virtual environment to install all necessary dependencies.

## Getting Started
##### If no data are present in the ```data``` folder
1. Download the data ```iris.data``` from the [data source](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/) and rename the file as ```iris.csv```. This file should contain 5 columns of data and should not contain any header rows.
2. Store the ```iris.csv``` in the ```data``` folder.
3. If the name of the main dataset is changed, the filename within ```pd.read_csv``` function in Section 1.2 of ```analysis.ipynb``` and the ```IRIS_CSV_DATA_FILENAME``` variable in ```app.py``` will need to be updated accordingly.

##### If no files are present in the ```model``` folder
1. Run all the cells in the ```analysis.ipynb``` Jupyter Notebook to generate a ```best_model.pkl``` file (containing a fitted Logistic Regression model)
2. Within the notebook, changes and improvements can be made to the model and the model can be saved into the ```model``` folder using the ```save_file``` function in the notebook.
3. If the name of the main predictive model is changed, the filename within ```load_file``` function in Section 6.3 of ```analysis.ipynb``` and the ```MODEL_PKL_FILENAME``` variable in ```app.py``` will need to be updated accordingly.

##### If the data and model files are already present in their respective folders folder
1. Activate the virtual environment in the terminal or command prompt
2. Run ```python app.py``` to start the Dashboard
3. The app should be running on http://localhost:8050/
