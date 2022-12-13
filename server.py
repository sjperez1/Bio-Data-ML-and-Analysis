import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# from matplotlib.figure import Figure
# The following import and use() were to get rid of the threading error
import matplotlib
matplotlib.use('Agg')
import io
from flask import Flask, Response, render_template
app = Flask(__name__)

@app.route("/")
def display_home():
    return render_template("index.html")

@app.route("/age_and_sex")
def display_age_sex():
    b_cancer_df_returns = b_cancer_df()
    p_cancer_df_returns = p_cancer_df()
    # getting each part needed from the function returns
    sum_stats_bc = b_cancer_df_returns[1]
    unique_sexes = b_cancer_df_returns[2]
    corr_middle_age_met_diagnosis = b_cancer_df_returns[3]

    sum_stats_pc = p_cancer_df_returns[1]
    return render_template("age_and_sex.html", sum_stats_bc = sum_stats_bc, unique_sexes = unique_sexes, corr_middle_age_met_diagnosis = round(corr_middle_age_met_diagnosis, 3), sum_stats_pc = sum_stats_pc)

@app.route('/b_cancer_age_dist.png')
def b_cancer_age_dist_png():
    fig = create_b_cancer_age_dist_png()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype = 'image/png')

def create_b_cancer_age_dist_png():
    age_sort = b_cancer_df()[0].sort_values('MedR Age at Diagnosis')
    fig, ax = plt.subplots()
    ax = sns.histplot(data=age_sort, x="MedR Age at Diagnosis")
    return fig

@app.route('/b_cancer_age_time_scatter.png')
def b_cancer_scatterplot_png():
    fig = create_b_cancer_scatterplot_png()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype = 'image/png')

def create_b_cancer_scatterplot_png():
    b_cancer = b_cancer_df()[0]
    fig, ax = plt.subplots()
    ax = sns.scatterplot(data=b_cancer, x="middle_age", y="MedR Time to Metastatic Diagnosis (Calculated Months)", hue='middle_age')
    return fig

def b_cancer_df():
    # read the breast cancer data file with pandas to make a dataframe
    b_data = pd.read_csv('brca_mbcproject_2022_clinical_data.tsv', sep='\t')
    # selecting columns from the original dataframe to have in this new dataframe
    b_cancer = b_data[['Patient ID', 'Sample ID', 'Cancer Type Detailed', 'MedR Age at Diagnosis', 
    'MedR Time to Metastatic Diagnosis (Calculated Months)', 'Number of Samples Per Patient', 'Mutation Count', 'PATH Procedure Location',
    'PATH Sample Histology', 'PATH Sample Grade', 'PATH Estrogen Receptor Status', 'PRD Ever Inflammatory', 'PATH Progesterone Receptor Status', 'Fraction Genome Altered', 'PATH HER2 Status','PRD Any Therapy More Than 2 Years', 'PATH Sample Treatment Naive', 'PATH Sample In Metastatic Setting', 'PATH DCIS Reported','PATH LCIS Reported', 'MedR Sex']]

    # See the number of NAs in each column of b_cancer, then dropping those values
    print(b_cancer.isna().sum())
    # dropping columns that won't be used and have NAs
    b_cancer = b_cancer.drop(['PATH DCIS Reported', 'PATH LCIS Reported'], axis = 1)
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
    sum_stats_bc = b_cancer['middle_age'].describe()
    sum_stats_bc = sum_stats_bc.to_dict()

    # Finding whether there are any males in dataset. There are not.
    unique_sexes = b_cancer['MedR Sex'].unique()
    print(unique_sexes)

    # Correlation between middle age and time to metastatic diagnosis
    corr_middle_age_met_diagnosis = np.corrcoef(b_cancer['middle_age'],b_cancer['MedR Time to Metastatic Diagnosis (Calculated Months)'])[0][1] # this function returns a 4 number matrix and can use the second column in first row or first column in second row to get correlation coefficient.
    return [b_cancer, sum_stats_bc, unique_sexes[0], corr_middle_age_met_diagnosis] # only getting unique_sexes at 0 position of array because printing the array showed only one value in it.


@app.route('/p_cancer_age_dist.png')
def p_cancer_age_dist_png():
    fig = create_p_cancer_age_dist_png()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype = 'image/png')

def create_p_cancer_age_dist_png():
    p_cancer = p_cancer_df()[0]
    fig, ax = plt.subplots()
    ax = sns.histplot(data=p_cancer, x="Diagnosis Age", binwidth=4)
    return fig

def p_cancer_df():
    # read the prostate cancer data file with pandas to make a dataframe
    p_cancer = pd.read_csv('prostate_dkfz_2018_clinical_data.tsv', sep='\t')

    # See the number of NAs in each column of b_cancer, then dropping those values
    print(p_cancer.isna().sum())
    # dropping columns that won't be used and have NAs
    p_cancer = p_cancer.drop(['BCR Status', 'Clonality', 'Median Purity', 'Mono or Multifocal Status', 'Preop PSA', 'Stage', 'Time from Surgery to BCR/Last Follow Up', 'TMB (nonsynonymous)'], axis = 1)
    # Dropping the NA values detected.
    p_cancer = p_cancer.dropna()

    # helpful to see data types in each column
    print(p_cancer.dtypes)

    # Verifying that there are no unexpected data points here
    print(p_cancer['Sex'].unique())

    # getting summary statistics for 'Diagnosis Age'
    sum_stats_pc = p_cancer['Diagnosis Age'].describe()
    sum_stats_pc = sum_stats_pc.to_dict()

    return [p_cancer, sum_stats_pc]

@app.route("/machine_learning")
def display_machine_learning():
    return render_template("machine_learning.html")

if __name__ == "__main__":
    app.run(debug=True)