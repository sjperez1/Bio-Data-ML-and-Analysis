import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, Response, render_template
app = Flask(__name__)

@app.route("/")
def display_home():
    return render_template("index.html")

@app.route("/age_and_sex")
def manipulate_b_cancer_df():
    # read the breast cancer data file with pandas to make a dataframe
    b_data = pd.read_csv('brca_mbcproject_2022_clinical_data.tsv', sep='\t')
    # selecting columns from the original dataframe to have in this new dataframe
    b_cancer = b_data[['Patient ID', 'Sample ID', 'Cancer Type Detailed', 'MedR Age at Diagnosis', 
    'MedR Time to Metastatic Diagnosis (Calculated Months)', 'Number of Samples Per Patient', 'Mutation Count', 'PATH Procedure Location',
    'PATH Sample Histology', 'PATH Sample Grade', 'PATH Estrogen Receptor Status', 'PRD Ever Inflammatory', 'PATH Progesterone Receptor Status', 'Fraction Genome Altered', 'PATH HER2 Status','PRD Any Therapy More Than 2 Years', 'PATH Sample Treatment Naive', 'PATH Sample In Metastatic Setting', 'PATH DCIS Reported','PATH LCIS Reported', 'MedR Sex']]

    # See the number of NAs in each column of b_cancer, then dropping those values
    print(b_cancer.isna().sum())
    b_cancer = b_cancer.dropna()
    # helpful to see data types in each column
    print(b_cancer.dtypes)
    # Finding the unique values in the age at diagnosis column because there was a data type error when trying to find 'middle_age'
    print(b_cancer['MedR Age at Diagnosis'].unique())
    # One row contains less than 25, so dropped because cannot accurately pinpoint patient age in this range
    b_cancer = b_cancer[b_cancer['MedR Age at Diagnosis']!='<25']

    # MedR Age at Diagnosis is classified in a range rather than one specific number, so splitting the range string, converting to numbers, and finding the middle of the range to work with one number. This number will be called 'middle_age'
    age = b_cancer['MedR Age at Diagnosis'].str.split('-',1,expand=True)
    age = age.astype('int')
    middle_age = (age[1]+age[0])/2
    b_cancer['middle_age'] = middle_age

    # getting summary statistics for 'middle_age'
    sum_stats = b_cancer['middle_age'].describe()
    return render_template("age_and_sex.html",b_cancer = b_cancer, sum_stats = sum_stats.to_dict())
# print(manipulate_b_cancer_df())

# @app.route('/age_and_sex')
# def display_age_sex():


if __name__ == "__main__":
    app.run(debug=True)