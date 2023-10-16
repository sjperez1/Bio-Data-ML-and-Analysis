import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def display_home():
    return render_template("index.html")

@app.route("/age_and_sex")
def display_age_sex():
    # calling the functions where dataframes were manipulated and analyzed
    b_cancer_df_returns = b_cancer_df()
    p_cancer_df_returns = p_cancer_df()
    # getting each part needed from the function returns to render to template
    sum_stats_bc = b_cancer_df_returns[1]
    unique_sexes = b_cancer_df_returns[2]
    corr_middle_age_met_diagnosis = b_cancer_df_returns[3]
    sum_stats_pc = p_cancer_df_returns[1]
    figure1 = b_cancer_df_returns[4]
    figure2 = p_cancer_df_returns[2]
    figure3 = b_cancer_df_returns[5]

    # specifying the html document and what to send to it
    return render_template("age_and_sex.html", sum_stats_bc = sum_stats_bc, unique_sexes = unique_sexes, corr_middle_age_met_diagnosis = round(corr_middle_age_met_diagnosis, 3), sum_stats_pc = sum_stats_pc, figure1 = figure1, figure2 = figure2, figure3 = figure3)

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
    age = b_cancer['MedR Age at Diagnosis'].str.split('-',n=1,expand=True)
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
    corr_middle_age_met_diagnosis = np.corrcoef(b_cancer['middle_age'],b_cancer['MedR Time to Metastatic Diagnosis (Calculated Months)'])[0][1] # this function returns a 4 number matrix, so can use the second column in first row or first column in second row to get correlation coefficient.

    # making figure 1 and encoding it in base64 string to be decoded on the front end
    age_sort = b_cancer.sort_values('MedR Age at Diagnosis')
    fig1 = Figure() 
    ax1 = fig1.add_subplot()
    ax1.hist(data=age_sort, x="MedR Age at Diagnosis", bins = np.arange(10)-.5, edgecolor = "black")
    ax1.set_title("Histogram of MedR Age at Diagnosis")
    ax1.set_xlabel("MedR Age at Diagnosis")
    ax1.set_ylabel("Count")
    img1 = io.BytesIO()
    FigureCanvas(fig1).print_png(img1)
    img1_str = "data:image/png;base64,"
    img1_str += base64.b64encode(img1.getvalue()).decode('utf8')

    # making figure 3 and encoding it in base64 string to be decoded on the front end
    fig3 = Figure() 
    ax3 = fig3.add_subplot()
    ax3.scatter(b_cancer["middle_age"], b_cancer["MedR Time to Metastatic Diagnosis (Calculated Months)"])
    ax3.set_title("Scatterplot of Diagnosis Age and Time to \n Metastatic Diagnosis (Calculated in Months)")
    ax3.set_xlabel("Middle Age")
    ax3.set_ylabel("MedR Time to Metastatic Diagnosis")
    img3 = io.BytesIO()
    FigureCanvas(fig3).print_png(img3)
    img3_str = "data:image/png;base64,"
    img3_str += base64.b64encode(img3.getvalue()).decode('utf8')


    # MACHINE LEARNING - Predicting age at diagnosis by mutation count
    # Split data
    X = b_cancer['Mutation Count']
    y = b_cancer['middle_age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Logistic Regression
    lr = LogisticRegression(random_state=7)
    lr.fit(X_train.values.reshape(-1, 1), y_train)
    lr.predict(X_test.values.reshape(-1, 1))
    acc_lr = lr.score(X_test.values.reshape(-1, 1), y_test)
    print("LR accuracy", acc_lr)
    # Predicting age for every mutation count
    y_pred = lr.predict(X_test.values.reshape(-1, 1))
    print(classification_report(y_test,y_pred))

    # SVC
    svm = SVC(random_state=7)
    svm.fit(X_train.values.reshape(-1, 1), y_train)
    acc_svm = svm.score(X_test.values.reshape(-1, 1), y_test)
    print("SVC accuracy", acc_svm )
    # Predicting age for every mutation count
    y_pred = svm.predict(X_test.values.reshape(-1, 1))
    print(classification_report(y_test,y_pred))
    
    # making figure 4 and encoding it in base64 string to be decoded on the front end
    c_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig4 = Figure() 
    ax4 = fig4.add_subplot()
    im = ax4.matshow(c_matrix,  cmap=plt.cm.Blues)
    fig4.colorbar(im)
    ax4.set_ylabel("Actuals")
    ax4.set_xlabel("Predictions")
    ax4.set_title("Confusion Matrix of SVC")
    img4 = io.BytesIO()
    FigureCanvas(fig4).print_png(img4)
    img4_str = "data:image/png;base64,"
    img4_str += base64.b64encode(img4.getvalue()).decode('utf8')

    # Random Forest
    forest = RandomForestClassifier(random_state=7)
    forest.fit(X_train.values.reshape(-1, 1), y_train)
    print("Random forest accuracy", forest.score(X_test.values.reshape(-1, 1), y_test))

    # returning what needs to be used in other functions
    return [b_cancer, sum_stats_bc, unique_sexes[0], corr_middle_age_met_diagnosis, img1_str, img3_str, img4_str, acc_svm] # only getting unique_sexes at 0 position of array because printing the array showed only one value in it.


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

    # verifying that there are no unexpected data points here
    print(p_cancer['Sex'].unique())

    # getting summary statistics for 'Diagnosis Age'
    sum_stats_pc = p_cancer['Diagnosis Age'].describe()
    sum_stats_pc = sum_stats_pc.to_dict()

    # making figure 2 and encoding it in base64 string to be decoded on the front end
    fig2 = Figure()
    ax2 = fig2.add_subplot()
    ax2.hist(data=p_cancer, x="Diagnosis Age", bins=12, edgecolor="black")
    ax2.set_title("Histogram of Diagnosis Age")
    ax2.set_xlabel("Diagnosis Age")
    ax2.set_ylabel("Count")
    img2 = io.BytesIO()
    FigureCanvas(fig2).print_png(img2)
    img2_str = "data:image/png;base64,"
    img2_str += base64.b64encode(img2.getvalue()).decode('utf8')


    # MACHINE LEARNING - Predicting diagnosis age by mutation count
    # Split data
    X = p_cancer['Mutation Count']
    y = p_cancer['Diagnosis Age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Linear Regression
    linr = LinearRegression()
    linr.fit(X_train.values.reshape(-1, 1), y_train)
    y_test_pred = linr.predict(X_test.values.reshape(-1, 1))
    acc_train = linr.score(X_train.values.reshape(-1, 1), y_train)
    acc_test = linr.score(X_test.values.reshape(-1, 1), y_test)
    print("Train accuracy", acc_train, "Test accuracy", acc_test)

    # SVR
    from sklearn.svm import SVR
    svr = SVR(C=10000)
    svr.fit(X_train.values.reshape(-1, 1), y_train)
    y_test_pred = svr.predict(X_test.values.reshape(-1, 1))
    acc_train = svr.score(X_train.values.reshape(-1, 1), y_train)
    acc_test = svr.score(X_test.values.reshape(-1, 1), y_test)
    print("Train accuracy", acc_train, "Test accuracy", acc_test)

    # making figure 5 and encoding it in base64 string to be decoded on the front end
    fig5 = Figure() 
    ax5 = fig5.add_subplot()
    ax5.scatter(y_test_pred, y_test, c='purple')
    ax5.axline((0,0), slope=1, c='black')
    ax5.set_xlim([45,75])
    ax5.set_ylim([30,80])
    ax5.set_title("Support Vector Regression Prediction of Diagnosis Age")
    ax5.set_xlabel("Predictions")
    ax5.set_ylabel("Actuals")
    img5 = io.BytesIO()
    FigureCanvas(fig5).print_png(img5)
    img5_str = "data:image/png;base64,"
    img5_str += base64.b64encode(img5.getvalue()).decode('utf8')

    # returning what needs to be used in other functions
    return [p_cancer, sum_stats_pc, img2_str, img5_str]

@app.route("/machine_learning")
def display_machine_learning():
    # calling the functions where dataframes were manipulated and analyzed
    b_cancer_df_returns = b_cancer_df()
    p_cancer_df_returns = p_cancer_df()
    # getting each part needed from the function returns to render to template
    figure4 = b_cancer_df_returns[6]
    figure5 = p_cancer_df_returns[3]
    accuracy = b_cancer_df_returns[7]

    # specifying the html document and what to send to it
    return render_template("machine_learning.html", figure4 = figure4, figure5 = figure5, accuracy = round(accuracy* 100, 2))

if __name__ == "__main__":
    app.run(debug=True)