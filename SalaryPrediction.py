# Importing all the Libraries

import streamlit as st
import pandas as pd
import numpy as np

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.metrics import make_scorer, confusion_matrix
# from sklearn.model_selection import learning_curve
# from sklearn.metrics import make_scorer, r2_score, mean_squared_error, auc, mean_absolute_error
import matplotlib.pyplot as plt

import seaborn as sns 
from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

# from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import RFE
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import make_scorer, r2_score, mean_squared_error, auc, mean_absolute_error
# from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.cross_validation import KFold # old version

# from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt



st.set_page_config(layout="wide")


st.markdown('# Web APP - Kaggle 2019 Data Survey')
'By - Tushar Aggarwal, M.Eng - University of Toronto'
'Github - https://github.com/TA-Engineer'








@st.cache(persist=True)
def load_data(filename):
    data = pd.read_csv(filename,low_memory=False)
    return data
Salaries = load_data("Kaggle_Salary.csv")


st.sidebar.header('User Input Features')

if st.sidebar.checkbox('Raw Data Information', True): 
    ''' This is a Web App built using:
    - Streamlit
    - Scikit Learn
    - Plotly
    - Pandas
    - etc.


    This Data Science web app explores the 2019 Kaggle Machine Learning and Data Science Survey of 19717 respondents from more than 171 countries. 

    Through this project I have demonstrated data mining skills from data extraction, data cleaning, data exploratino & visualization to model prediction.


    The Tabs on the left guide you through each stage of the project

    The model predictor was a updated predictor which predicts data scienctist Salary based on few features compared to over 200+ in our dataset.

    
    '''




    if st.sidebar.checkbox('Raw Data', False):
        '## Displaying Raw Data', Salaries

    if st.sidebar.checkbox('Raw Data HeatMap'):

        fig, ax = plt.subplots(figsize=(50,20))
        ' ## Raw Data HeatMap'
        sns.heatmap(Salaries.isnull(), cmap='coolwarm', yticklabels=False, cbar=False, ax=ax)
        st.pyplot(fig)
        ' It can be observered that there are several rows in each column that are empty'



    if st.sidebar.checkbox('Show Survey Questions'):
        '''
        # Survey Questions \n
        \n
        \nQ1: What is your age (# years)? \n
        \nQ2: What is your gender? - Selected Choice
        \nQ3: In which country do you currently reside?
        \nQ4: What is the highest level of formal education that you have attained or plan to attain within the next 2 years?
        \nQ5: Select the title most similar to your current role (or most recent title if retired): - Selected Choice
        \nQ6: What is the size of the company where you are employed?
        \nQ7: Approximately how many individuals are responsible for data science workloads at your place of business?
        \nQ8: Does your current employer incorporate machine learning methods into their business?
        \nQ9: Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice
        \nQ10: What is your current yearly compensation (approximate $USD)?
        \nQ11: Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?
        \nQ12: Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice
        \nQ13: On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice
        \nQ14: What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice
        \nQ15: How long have you been writing code to analyze data (at work or at school)?
        \nQ16: Which of the following integrated development environments (IDE's) do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ17: Which of the following hosted notebook products do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ18: What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ19: What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice
        \nQ20: What data visualization libraries or tools do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ21: Which types of specialized hardware do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ22: Have you ever used a TPU (tensor processing unit)?
        \nQ23: For how many years have you used machine learning methods?
        \nQ24: Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice
        \nQ25: Which categories of ML tools do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ26: Which categories of computer vision methods do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ27: Which of the following natural language processing (NLP) methods do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ28: Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ29: Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ30: Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ31: Which specific big data / analytics products do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ32: Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ33: Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis? (Select all that apply) - Selected Choice
        \nQ34: Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice
        '''


###################### DATA CLEARNING #############################

st.sidebar.subheader('Data Cleaning Steps')
if st.sidebar.checkbox('Show Data Clearning Basic Steps', False):

    st.markdown('# Data Cleaning')

    'The following data Cleaning Steps were Taken from Raw Data to model Preperation.'
    '* Row 1 was dropped as it contained all the survey questions.'
    "* Then the Following columns were dropped:"
    "\n'Q2_OTHER_TEXT', 'Q5_OTHER_TEXT', 'Q9_OTHER_TEXT', 'Q9_OTHER_TEXT', 'Q12_OTHER_TEXT', 'Q13_OTHER_TEXT', 'Q14_OTHER_TEXT', 'Q16_OTHER_TEXT', 'Q17_OTHER_TEXT', 'Q18_OTHER_TEXT', 'Q19_OTHER_TEXT', 'Q20_OTHER_TEXT', 'Q21_OTHER_TEXT', 'Q24_OTHER_TEXT', 'Q25_OTHER_TEXT', 'Q26_OTHER_TEXT', 'Q27_OTHER_TEXT', 'Q28_OTHER_TEXT', 'Q29_OTHER_TEXT', 'Q30_OTHER_TEXT', 'Q31_OTHER_TEXT','Q32_OTHER_TEXT', 'Q33_OTHER_TEXT', 'Q34_OTHER_TEXT'"
    "These are the other option selected by people and mostly contain -1."
    "* Columns with less than 2% of missing data, null value rows were dropped. These consist of only 249 rows out of 12000+ records, hence should not effect our prediction."
    '* Dropped rows of people who took more than 4 hours to complete the survey. (median time to complete the survey was 10.7minutes and 4rson average.'
    
    '''* Furhter Cleaning was done based on questions. (Questions can be seen on top left side of the sidebar): \n      
    * Q1: 10 Age buckets were created from 18 to 70+ with increament of 10 years. All values were encoded to these age buckets.  
    * Q2: Gender questions was also encoded into 4 customs buckets: Male, Female, Prefer Not to Say and Prefer not to self describe from 0,1,2 and 2 respectively.
    * Q3: In which country do you currently reside?.   There are 59 countries. Therefore Label Encoding were used for countries. Howevever, countries with less than 60 values respondents were changed to other countries.
    * Q4: This was the highest level of education the person has attained which was simply label encoded using scikit learn.
    * Q5: this was similar to Q4 was label encoded using scikit learn library.
    * Q6: Company Size. Here also 5 buckets were created based on company size. These custom buckets were then used to encode the columns.
    * Q7: 7 Buckets were created based on number of people working in Data Science department in the company. Values were encoded in them.
    * Q8: This question's column was label encoded as it was a categorical columnw with 5 multiple choice answers.
    *  Questions 11, 15,22 and 23 countained values, therefore buckets were created for each questions and columns were then coded based on bucket values.
    * Q19: This was a recommendation column and does not provide informatin on the Salary Prediction Hence this was dropped.
    * All the remaining questions were multiple choice questions. Each coice for each of the MCQ question was represented as an individual column.
     Therefore, for wach questions. All the column were coded using Binarization Encoding. This allows to keep the choice selected as 1 and other column for that questions as 0.
    * Q14: This question as label exncoded as it shows the development environment.  
    '''
    





# Salary1 = Salaries.copy()
# question_salaries = Salaries[:1]
# Salaries.drop(Salaries.index[0], axis = 0, inplace=True)

# # Dropping all columns containing OTHER_TEXT
# # Dropping all column containing "OTHER_TEXT" in header
# # OTHER_TEXT columns represent respondents answer when they select others in multiple choise answers. 
# # The columns contain only numbers (mostly -1) or text and do not help in salary prediction. 
# # We will be encoding the 'other'choice column of question instead therefore dropping 'other_text' column is justified. 
# # These columns also donot provide any additional information hence will not contribute to the model.

# Salaries.drop([ 'Q2_OTHER_TEXT','Q5_OTHER_TEXT', 'Q9_OTHER_TEXT', 'Q9_OTHER_TEXT','Q12_OTHER_TEXT','Q13_OTHER_TEXT', 'Q14_OTHER_TEXT', 'Q16_OTHER_TEXT', 'Q17_OTHER_TEXT', 'Q18_OTHER_TEXT', 'Q19_OTHER_TEXT', 'Q20_OTHER_TEXT', 'Q21_OTHER_TEXT', 'Q24_OTHER_TEXT', 'Q25_OTHER_TEXT', 'Q26_OTHER_TEXT', 'Q27_OTHER_TEXT', 'Q28_OTHER_TEXT', 'Q29_OTHER_TEXT', 'Q30_OTHER_TEXT', 'Q31_OTHER_TEXT','Q32_OTHER_TEXT', 'Q33_OTHER_TEXT', 'Q34_OTHER_TEXT'], axis=1, inplace=True)

# # Dropping Rows for columns missing less than 2% of Data
# # For columns with less than 2% of missing data, rows with null/Nan values will be dropped. This should not effect our prediction from 12497 records
# lessthan_2 = Salaries.shape[0]*0.02
# # print('2% of rows = ',lessthan_2)

# cols_null_less_2 = [col for col in Salaries.columns if (Salaries[col].isnull().sum()>0) & (Salaries[col].isnull().sum()<lessthan_2)]
# # print('Columns with less than 2% Values to be dropped is/are: ',cols_null_less_2)

# Salaries.dropna(subset=cols_null_less_2,axis=0, inplace=True)


# # Dropping people who took more than 4 hours to complete the survey
# # Time from Start to Finish (seconds)' column

# minutes = Salaries['Time from Start to Finish (seconds)'].astype(str).astype(int)/60
# average_time = minutes.mean()
# median_time = minutes.median()
# Salaries = Salaries[Salaries['Time from Start to Finish (seconds)'].astype(str).astype(int)<(240*60)]
# Salaries.drop([ 'Time from Start to Finish (seconds)'],axis=1, inplace=True)


# # Defining an Age bracket bucket
# age_bucket = {
# '18-21':0,
# '22-24':1,
# '25-29':2,
# '30-34':3,
# '35-39':4,
# '40-44':5,
# '45-49':6,
# '50-54':7,
# '55-59':8,
# '60-69':9,
# '70+':10}

# Salaries.loc[:,'Q1_Encoded'] = Salaries.loc[:,'Q1'].map(age_bucket)
# Salaries.loc[:,'Q1_Encoded'] = Salaries.loc[:,'Q1_Encoded'].astype(int)
# Salaries.Q1_Encoded.unique()

# # Q2: Gender
# gender_bucket = {
# 'Male' : 0,
# 'Female': 1,
# 'Prefer not to say':2,
# 'Prefer to self-describe':2
# }

# Salaries.loc[:,'Q2_Encoded'] = Salaries.loc[:,'Q2'].map(gender_bucket)
# Salaries.loc[:,'Q2_Encoded'] = Salaries.loc[:,'Q2_Encoded'].astype(int)

# Salary2 = Salaries.copy()

# # Question 3: Which Country do you reside in?
# Salary=Salaries

# country_transform = Salaries['Q3'].isin(Salaries['Q3'].value_counts().index[Salaries['Q3'].value_counts() < 50])
# Salaries.loc[country_transform,'Q3'] = "Other"
# Salaries['Q3_Encoded'] = LabelEncoder().fit_transform(Salaries['Q3'])
# Salaries = pd.get_dummies(Salary,columns = ['Q3'])

# # Q4 and Q5: Label Encoding Ques 4 and Ques 5.
# Salaries['Q4_Encoded'] = LabelEncoder().fit_transform(Salaries['Q4'])
# Salaries['Q5_Encoded'] = LabelEncoder().fit_transform(Salaries['Q5'])

# # Question 6: Company Size

# size_company_bucket={
#     '0-49 employees':0,
#     '50-249 employees':1,
#     '250-999 employees':2,
#     '1000-9,999 employees':3,
#     '> 10,000 employees':4  
# }    
 

     
# Salaries.loc[:,'Q6_Encoded'] = Salaries.loc[:,'Q6'].map(size_company_bucket)
# Salaries.loc[:,'Q6_Encoded'] = Salaries.loc[:,'Q6_Encoded'].astype(int)
# # Salaries.Q6_Encoded.unique()


# # Question 7: 
# indi_company_bucket={
#     '0':0,
#     '1-2':1,
#     '3-4':2,
#     '5-9':3,
#     '10-14':4,
#     '15-19':5,
#     '20+':6
# }    


# Salaries.loc[:,'Q7_Encoded'] = Salaries.loc[:,'Q7'].map(indi_company_bucket)
# Salaries.loc[:,'Q7_Encoded'] = Salaries.loc[:,'Q7_Encoded'].astype(int)

# # Question 8
# Salaries['Q8_Encoded'] = LabelEncoder().fit_transform(Salaries['Q8'])


# # Qeustion 11:

# spend_bucket={
#     '$0 (USD)':0,
#     '$1-$99':1,
#     '$100-$999':2,
#     '$1000-$9,999':3,      
#     '$10,000-$99,999':4, 
#     '> $100,000 ($USD)':5   
# }  

# Salaries.loc[:,'Q11_Encoded'] = Salaries.loc[:,'Q11'].map(spend_bucket)
# Salaries.loc[:,'Q11_Encoded'] = Salaries.loc[:,'Q11_Encoded'].astype(int)

# # Question 15 dropping NAN rows
# Salaries.dropna(subset=['Q15'],inplace=True) 

# code_bucket = {
#     '1-2 years':2,
#     'I have never written code':0,
#     '< 1 years':1,
#     '20+ years':6,
#     '3-5 years':3,
#     '5-10 years':4,
#     '10-20 years':5
# }


# Salaries.loc[:,'Q15_Encoded'] = Salaries.loc[:,'Q15'].map(code_bucket)
# Salaries.loc[:,'Q15_Encoded'] = Salaries.loc[:,'Q15_Encoded'].astype(int)


# # Q19: Dropping it
# Salaries.drop(columns = ['Q19'],inplace =True)

# # Q22: Encoding

# Salaries['Q22'].fillna(Salary['Q22'].mode()[0],inplace=True)


# tpu_bucket = {
#     '2-5 times':2,
#     'Never':0,
#     'Once':1,
#     '6-24 times':3,
#     '> 25 times':4

# }
# Salaries.loc[:,'Q22_Encoded'] = Salaries.loc[:,'Q22'].map(tpu_bucket)
# Salaries.loc[:,'Q22_Encoded'] = Salaries.loc[:,'Q22_Encoded'].astype(int)

# # Quesiton 23
# Salaries['Q23'].fillna(Salary['Q23'].mode()[0],inplace=True)



# ml_bucket = {
#     '2-3 years':2,
#     '< 1 years':0,
#     '1-2 years':1,
#     '3-4 years':3,
#     '4-5 years':4,
#     '5-10 years':5,
#     '10-15 years':6,
#     '20+ years':7

# }
# Salaries.loc[:,'Q23_Encoded'] = Salaries.loc[:,'Q23'].map(ml_bucket)
# Salaries.loc[:,'Q23_Encoded'] = Salaries.loc[:,'Q23_Encoded'].astype(int)







# # # First we will remove general questions that have single rows in one dataframe


# salaries_Q9 = Salaries[[ 'Q9_Part_1', 'Q9_Part_2', 'Q9_Part_3', 'Q9_Part_4', 'Q9_Part_5', 'Q9_Part_6', 'Q9_Part_7',
#                          'Q9_Part_8']]

# salaries_Q12 = Salaries[[ 'Q12_Part_1', 'Q12_Part_2', 'Q12_Part_3', 'Q12_Part_4', 'Q12_Part_5', 'Q12_Part_6','Q12_Part_7', 
#                          'Q12_Part_8','Q12_Part_9', 'Q12_Part_10', 'Q12_Part_11', 'Q12_Part_12']]

# salaries_Q13 = Salaries[['Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3', 'Q13_Part_4', 'Q13_Part_5', 'Q13_Part_6', 'Q13_Part_7',
#                           'Q13_Part_8', 'Q13_Part_9', 'Q13_Part_10', 'Q13_Part_11', 'Q13_Part_12']]

# salaries_Q14=Salaries[[ 'Q14', 'Q14_Part_1_TEXT', 'Q14_Part_2_TEXT', 'Q14_Part_3_TEXT', 'Q14_Part_4_TEXT', 'Q14_Part_5_TEXT']]

# salaries_Q16=Salaries[[ 'Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4', 'Q16_Part_5', 'Q16_Part_6', 'Q16_Part_7', 
#                'Q16_Part_8','Q16_Part_9', 'Q16_Part_10', 'Q16_Part_11', 'Q16_Part_12']]

# salaries_Q17=Salaries[['Q17_Part_1', 'Q17_Part_2', 'Q17_Part_3', 'Q17_Part_4', 'Q17_Part_5', 'Q17_Part_6', 'Q17_Part_7',
#                 'Q17_Part_8', 'Q17_Part_9', 'Q17_Part_10', 'Q17_Part_11', 'Q17_Part_12']]

# salaries_Q18=Salaries[[ 'Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5', 'Q18_Part_6', 'Q18_Part_7',
#                 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10', 'Q18_Part_11', 'Q18_Part_12']]

# salaries_Q20=Salaries[['Q20_Part_1', 'Q20_Part_2', 'Q20_Part_3', 'Q20_Part_4', 'Q20_Part_5','Q20_Part_6', 'Q20_Part_7',
#                 'Q20_Part_8', 'Q20_Part_9', 'Q20_Part_10', 'Q20_Part_11', 'Q20_Part_12']]

# salaries_Q21=Salaries[[ 'Q21_Part_1', 'Q21_Part_2', 'Q21_Part_3', 'Q21_Part_4', 'Q21_Part_5']]

# salaries_Q24=Salaries[[ 'Q24_Part_1', 'Q24_Part_2', 'Q24_Part_3', 'Q24_Part_4', 'Q24_Part_5', 'Q24_Part_6', 'Q24_Part_7',
#                'Q24_Part_8', 'Q24_Part_9', 'Q24_Part_10', 'Q24_Part_11', 'Q24_Part_12']]

# salaries_Q25=Salaries[['Q25_Part_1', 'Q25_Part_2', 'Q25_Part_3', 'Q25_Part_4', 'Q25_Part_5', 'Q25_Part_6', 'Q25_Part_7',
#                 'Q25_Part_8']]
               
# salaries_Q26 = Salaries[['Q26_Part_1', 'Q26_Part_2', 'Q26_Part_3', 'Q26_Part_4', 'Q26_Part_5', 'Q26_Part_6', 'Q26_Part_7']]
               
# salaries_Q27 = Salaries[['Q27_Part_1', 'Q27_Part_2', 'Q27_Part_3', 'Q27_Part_4', 'Q27_Part_5', 'Q27_Part_6']]
               
# salaries_Q28 = Salaries[['Q28_Part_1', 'Q28_Part_2', 'Q28_Part_3', 'Q28_Part_4', 'Q28_Part_5', 'Q28_Part_6', 'Q28_Part_7',
#                  'Q28_Part_8', 'Q28_Part_9', 'Q28_Part_10', 'Q28_Part_11', 'Q28_Part_12']]

# salaries_Q29 =Salaries[['Q29_Part_1', 'Q29_Part_2', 'Q29_Part_3', 'Q29_Part_4', 'Q29_Part_5', 'Q29_Part_6', 'Q29_Part_7',
#                  'Q29_Part_8', 'Q29_Part_9', 'Q29_Part_10', 'Q29_Part_11', 'Q29_Part_12']]
               
# salaries_Q30 = Salaries[[ 'Q30_Part_1', 'Q30_Part_2', 'Q30_Part_3', 'Q30_Part_4', 'Q30_Part_5', 'Q30_Part_6', 'Q30_Part_7',
#                  'Q30_Part_8', 'Q30_Part_9', 'Q30_Part_10', 'Q30_Part_11', 'Q30_Part_12']  ]         

# salaries_Q31 =Salaries[['Q31_Part_1', 'Q31_Part_2', 'Q31_Part_3', 'Q31_Part_4', 'Q31_Part_5', 'Q31_Part_6', 'Q31_Part_7',
#                  'Q31_Part_8', 'Q31_Part_9', 'Q31_Part_10', 'Q31_Part_11', 'Q31_Part_12']]

# salaries_Q32 = Salaries[['Q32_Part_1', 'Q32_Part_2', 'Q32_Part_3', 'Q32_Part_4', 'Q32_Part_5', 'Q32_Part_6', 'Q32_Part_7',
#                  'Q32_Part_8', 'Q32_Part_9', 'Q32_Part_10', 'Q32_Part_11', 'Q32_Part_12']]

# salaries_Q33 = Salaries[['Q33_Part_1', 'Q33_Part_2', 'Q33_Part_3', 'Q33_Part_4', 'Q33_Part_5', 'Q33_Part_6', 'Q33_Part_7',
#                  'Q33_Part_8', 'Q33_Part_9', 'Q33_Part_10', 'Q33_Part_11', 'Q33_Part_12']]

# salaries_Q34 = Salaries[['Q34_Part_1', 'Q34_Part_2', 'Q34_Part_3', 'Q34_Part_4', 'Q34_Part_5', 'Q34_Part_6', 'Q34_Part_7',
#                  'Q34_Part_8', 'Q34_Part_9', 'Q34_Part_10', 'Q34_Part_11', 'Q34_Part_12']]







# # Since all the remaining questions are multiple choice we will binarze the value where if the choice was selected that value will be equal to 1 and all null vlaues will be 0


# part_col_list = ['Q9_Part_1', 'Q9_Part_2', 'Q9_Part_3', 'Q9_Part_4', 'Q9_Part_5', 'Q9_Part_6', 'Q9_Part_7',
#                 'Q9_Part_8','Q12_Part_1', 'Q12_Part_2', 'Q12_Part_3', 'Q12_Part_4', 'Q12_Part_5', 'Q12_Part_6','Q12_Part_7', 
#                 'Q12_Part_8','Q12_Part_9', 'Q12_Part_10', 'Q12_Part_11', 'Q12_Part_12','Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3', 
#                 'Q13_Part_4', 'Q13_Part_5', 'Q13_Part_6', 'Q13_Part_7', 'Q13_Part_8', 'Q13_Part_9', 'Q13_Part_10', 'Q13_Part_11',
#                  'Q13_Part_12','Q14', 'Q14_Part_1_TEXT', 'Q14_Part_2_TEXT', 'Q14_Part_3_TEXT', 'Q14_Part_4_TEXT', 'Q14_Part_5_TEXT',
#                 'Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4', 'Q16_Part_5', 'Q16_Part_6', 'Q16_Part_7',
#                  'Q16_Part_8','Q16_Part_9', 'Q16_Part_10', 'Q16_Part_11', 'Q16_Part_12','Q17_Part_1', 'Q17_Part_2', 'Q17_Part_3', 
#                  'Q17_Part_4', 'Q17_Part_5', 'Q17_Part_6', 'Q17_Part_7','Q17_Part_8', 'Q17_Part_9', 'Q17_Part_10', 'Q17_Part_11', 
#                  'Q17_Part_12', 'Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5', 'Q18_Part_6', 'Q18_Part_7',
#                  'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10', 'Q18_Part_11', 'Q18_Part_12','Q20_Part_1', 'Q20_Part_2', 'Q20_Part_3',
#                  'Q20_Part_4', 'Q20_Part_5','Q20_Part_6', 'Q20_Part_7','Q20_Part_8', 'Q20_Part_9', 'Q20_Part_10', 'Q20_Part_11', 
#                  'Q20_Part_12','Q21_Part_1', 'Q21_Part_2', 'Q21_Part_3', 'Q21_Part_4', 'Q21_Part_5','Q24_Part_1', 'Q24_Part_2', 
#                  'Q24_Part_3', 'Q24_Part_4', 'Q24_Part_5', 'Q24_Part_6', 'Q24_Part_7','Q24_Part_8', 'Q24_Part_9', 'Q24_Part_10',
#                  'Q24_Part_11', 'Q24_Part_12','Q25_Part_1', 'Q25_Part_2', 'Q25_Part_3', 'Q25_Part_4', 'Q25_Part_5', 'Q25_Part_6',
#                  'Q25_Part_7','Q25_Part_8','Q26_Part_1', 'Q26_Part_2', 'Q26_Part_3', 'Q26_Part_4', 'Q26_Part_5', 'Q26_Part_6', 
#                  'Q26_Part_7','Q27_Part_1', 'Q27_Part_2', 'Q27_Part_3', 'Q27_Part_4', 'Q27_Part_5', 'Q27_Part_6','Q28_Part_1', 
#                  'Q28_Part_2', 'Q28_Part_3', 'Q28_Part_4', 'Q28_Part_5', 'Q28_Part_6', 'Q28_Part_7','Q28_Part_8', 'Q28_Part_9', 
#                  'Q28_Part_10', 'Q28_Part_11', 'Q28_Part_12','Q29_Part_1', 'Q29_Part_2', 'Q29_Part_3','Q29_Part_4', 'Q29_Part_5',
#                  'Q29_Part_6', 'Q29_Part_7','Q29_Part_8', 'Q29_Part_9', 'Q29_Part_10', 'Q29_Part_11', 'Q29_Part_12','Q30_Part_1',
#                  'Q30_Part_2', 'Q30_Part_3', 'Q30_Part_4', 'Q30_Part_5', 'Q30_Part_6', 'Q30_Part_7',
#                  'Q30_Part_8', 'Q30_Part_9', 'Q30_Part_10', 'Q30_Part_11', 'Q30_Part_12','Q31_Part_1', 'Q31_Part_2', 'Q31_Part_3',
#                  'Q31_Part_4', 'Q31_Part_5', 'Q31_Part_6', 'Q31_Part_7','Q31_Part_8', 'Q31_Part_9', 'Q31_Part_10', 'Q31_Part_11',
#                  'Q31_Part_12','Q32_Part_1', 'Q32_Part_2', 'Q32_Part_3','Q32_Part_4', 'Q32_Part_5', 'Q32_Part_6', 'Q32_Part_7',
#                  'Q32_Part_8', 'Q32_Part_9', 'Q32_Part_10', 'Q32_Part_11','Q32_Part_12','Q33_Part_1', 'Q33_Part_2', 'Q33_Part_3',
#                  'Q33_Part_4', 'Q33_Part_5', 'Q33_Part_6', 'Q33_Part_7','Q33_Part_8', 'Q33_Part_9', 'Q33_Part_10', 'Q33_Part_11', 
#                  'Q33_Part_12','Q34_Part_1', 'Q34_Part_2', 'Q34_Part_3','Q34_Part_4', 'Q34_Part_5', 'Q34_Part_6', 'Q34_Part_7',
#                  'Q34_Part_8', 'Q34_Part_9', 'Q34_Part_10', 'Q34_Part_11', 'Q34_Part_12']



# Salary3 = Salaries.copy()



# def Binarize_encoding(row):
#     if isinstance(row,str):
#         return 1
#     else:
#         if row == 0 or math.isnan(row):
#             return 0
#         else:
#             return 1
# for col in part_col_list:
#         Salaries[col] = Salaries[col].apply(Binarize_encoding)






Salaries2 = load_data("Salaries_Cleaned.csv")




################# EXPLORATORY DATA ANALYSIS  and visualization###################################################

st.sidebar.subheader('Data Exploration Visualization')

EDA_option = st.sidebar.radio('Visualization Options', ('None','Respondent Country Count', 'Salary Buckets', 'Salary Vs Questions'))

if EDA_option == 'Respondent Country Count':
    '## Visualizing Countries Count Graph'
    # Visualizaing Questions:
    fig_2 = px.histogram(x = Salaries["Q3"], title = "Countries Count Graph - Total Response Per Country",
    labels={'x':'Countries', 'y':'Count','variable': 'Countries Count'}, height = 800, width = 800)
    st.plotly_chart(fig_2)

if EDA_option == 'Salary Buckets':
    '## Visualizing Salary Bucket Graph'
    salary_plot = pd.DataFrame()
    salary_plot =Salaries2.Q10_buckets.value_counts().rename_axis('unique_values').reset_index(name='counts')

    fig_3 = px.bar(salary_plot, x='unique_values', y='counts',
    labels={'unique_values': 'Salary Buckets', 'counts': 'Count'}, height = 600, width = 800)
    st.plotly_chart(fig_3)

    # plt.figure(figsize=(25,40))

    # fig_3 = sns.catplot(x='unique_values',
    #         y='counts',
    #         kind="bar",
    #         data=salary_plot)
    # fig_3.set_xticklabels(rotation=90)
    # plt.title('Visualization of of yearly salary')
    # plt.xlabel('Salary Bucket', fontsize=16)
    # plt.ylabel('Count in Salary', fontsize=16)
    # st.pyplot(fig_3)
    


  


if EDA_option == 'Salary Vs Questions':
    option = st.sidebar.selectbox('Choose an option',['Age','Degree', 'Profession', 'Gender'])


    if option == 'Age':
        '## Visualizing'
        # Visualizing salary Bucket by Age
        plt.figure(figsize=(30,15))
        fig_4 =sns.countplot(x='Q10_buckets', hue = 'Q1', data=Salaries2)
        plt.title('Visualization of of yearly salary by Age', fontsize=28)
        plt.xlabel('Salary Bucket', fontsize=24)
        plt.ylabel('Count in Salary', fontsize=24)
        st.pyplot(fig_4.figure)

    elif option == 'Degree':
        '## Visualizing'
        # Visualizing salary buckets by degree
        plt.figure(figsize=(30,15))
        fig_5 =sns.countplot(x='Q10_buckets', hue = 'Q4', data=Salaries2)
        plt.title('Visualization of of yearly salary by Degree', fontsize=28)
        plt.xlabel('Salary Bucket', fontsize=24)
        plt.ylabel('Count in Salary', fontsize=24)
        st.pyplot(fig_5.figure)


    elif option == ' Profession':
        '## Visualizing'
        # Visulization of Yearly Salary by Profession
        plt.figure(figsize=(30,15))
        fig_6 =sns.countplot(x='Q5', hue = 'Q10_buckets', data=Salaries2)
        plt.title('Visualization of of yearly salary by Profession', fontsize=28)
        plt.xlabel('Profession', fontsize=24)
        plt.ylabel('Count in Salary', fontsize=24)
        st.pyplot(fig_6.figure)


    # elif option == 'Gender':
    else:    
        '## Visualizing'
        # Visualization of Salary by Gender

        plt.figure(figsize=(30,15))
        fig_7 =sns.countplot(x='Q10_buckets', hue = 'Q2', data=Salaries2)
        plt.title('Visualization of of yearly salary by Gender', fontsize=28)
        plt.xlabel('Gender', fontsize=24)
        plt.ylabel('Count in Salary', fontsize=24)
        st.pyplot(fig_7.figure)




Salaries3 = Salaries2.copy()
Salaries3.drop(columns = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q10','Q11','Q15','Q22','Q23'], inplace = True)

st.sidebar.subheader("Feature Selection")

if st.sidebar.checkbox('Correlation Plot',False):

    corr = Salaries3.corr()
    # print(corr)


    plt.figure(figsize=(15,15))
    fig_10 = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap="RdYlGn",
        square=True,
        
    )
    plt.xticks(
        # fig_10.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    st.pyplot(fig_10.figure)




# # We will drop redundant columns which are already present in the dataframe






# Salaries3['Q14_Encoded'] = LabelEncoder().fit_transform(salaries_Q14['Q14'])
# Salaries3['Q14_Encoded'].value_counts()

# Salaries3.drop(columns = ['Q14','Q14_Part_1_TEXT','Q14_Part_2_TEXT','Q14_Part_3_TEXT','Q14_Part_4_TEXT','Q14_Part_5_TEXT'], inplace=True )




if st.sidebar.checkbox('Feature selection Steps', False):
    ' ## Feature Selection Steps'
    'For feature selection 3 Methods were used: \n'
    '* Chi Square Test'
    '* ExtraTree Classifier Methods'
    '* Recursive Feature Elimination'

    'Based on the common and best method we selected top 100 features.'



# # Correlation HeatMap of Salaries with other features:
# plt.figure(figsize=(15,40))
# plt.title("Heat Map for Correlation of Features with Salary",size=20)
# plt.ylabel("Features")
# plt.xlabel("Salary Feature")
# sns.heatmap(corr2[["Q10_Encoded"]],annot=True)


# # Rearranging DataFrame


# #A = Salaries.columns.tolist()
# #print(A)
# Salary5 = Salaries.copy()
# Salaries.drop(columns = ['Q10_buckets'], inplace=True )

# Salaries = Salaries[['Q1_Encoded', 'Q2_Encoded', 'Q3_Encoded', 'Q4_Encoded', 'Q5_Encoded', 'Q6_Encoded', 'Q7_Encoded',
#                      'Q8_Encoded', 'Q11_Encoded', 'Q15_Encoded', 'Q22_Encoded', 'Q23_Encoded', 'Q14_Encoded',
#                      'Q9_Part_1', 'Q9_Part_2', 'Q9_Part_3', 'Q9_Part_4', 'Q9_Part_5', 'Q9_Part_6', 'Q9_Part_7', 'Q9_Part_8',
#                      'Q12_Part_1', 'Q12_Part_2', 'Q12_Part_3', 'Q12_Part_4', 'Q12_Part_5', 'Q12_Part_6', 'Q12_Part_7',
#                      'Q12_Part_8', 'Q12_Part_9', 'Q12_Part_10', 'Q12_Part_11', 'Q12_Part_12', 'Q13_Part_1', 'Q13_Part_2',
#                      'Q13_Part_3', 'Q13_Part_4', 'Q13_Part_5', 'Q13_Part_6', 'Q13_Part_7', 'Q13_Part_8', 'Q13_Part_9',
#                      'Q13_Part_10', 'Q13_Part_11', 'Q13_Part_12', 'Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4',
#                      'Q16_Part_5', 'Q16_Part_6', 'Q16_Part_7', 'Q16_Part_8', 'Q16_Part_9', 'Q16_Part_10', 'Q16_Part_11',
#                      'Q16_Part_12', 'Q17_Part_1', 'Q17_Part_2', 'Q17_Part_3', 'Q17_Part_4', 'Q17_Part_5', 'Q17_Part_6',
#                      'Q17_Part_7', 'Q17_Part_8', 'Q17_Part_9', 'Q17_Part_10', 'Q17_Part_11', 'Q17_Part_12', 'Q18_Part_1',
#                      'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5', 'Q18_Part_6', 'Q18_Part_7', 'Q18_Part_8',
#                      'Q18_Part_9', 'Q18_Part_10', 'Q18_Part_11', 'Q18_Part_12', 'Q20_Part_1', 'Q20_Part_2', 'Q20_Part_3',
#                      'Q20_Part_4', 'Q20_Part_5', 'Q20_Part_6', 'Q20_Part_7', 'Q20_Part_8', 'Q20_Part_9', 'Q20_Part_10',
#                      'Q20_Part_11', 'Q20_Part_12', 'Q21_Part_1', 'Q21_Part_2', 'Q21_Part_3', 'Q21_Part_4', 'Q21_Part_5',
#                      'Q24_Part_1', 'Q24_Part_2', 'Q24_Part_3', 'Q24_Part_4', 'Q24_Part_5', 'Q24_Part_6', 'Q24_Part_7',
#                      'Q24_Part_8', 'Q24_Part_9', 'Q24_Part_10', 'Q24_Part_11', 'Q24_Part_12', 'Q25_Part_1', 'Q25_Part_2',
#                      'Q25_Part_3', 'Q25_Part_4', 'Q25_Part_5', 'Q25_Part_6', 'Q25_Part_7', 'Q25_Part_8', 'Q26_Part_1', 
#                      'Q26_Part_2', 'Q26_Part_3', 'Q26_Part_4', 'Q26_Part_5', 'Q26_Part_6', 'Q26_Part_7', 'Q27_Part_1',
#                      'Q27_Part_2', 'Q27_Part_3', 'Q27_Part_4', 'Q27_Part_5', 'Q27_Part_6', 'Q28_Part_1', 'Q28_Part_2',
#                      'Q28_Part_3', 'Q28_Part_4', 'Q28_Part_5', 'Q28_Part_6', 'Q28_Part_7', 'Q28_Part_8', 'Q28_Part_9',
#                      'Q28_Part_10', 'Q28_Part_11', 'Q28_Part_12', 'Q29_Part_1', 'Q29_Part_2', 'Q29_Part_3', 'Q29_Part_4',
#                      'Q29_Part_5', 'Q29_Part_6', 'Q29_Part_7', 'Q29_Part_8', 'Q29_Part_9', 'Q29_Part_10', 'Q29_Part_11',
#                      'Q29_Part_12', 'Q30_Part_1', 'Q30_Part_2', 'Q30_Part_3', 'Q30_Part_4', 'Q30_Part_5', 'Q30_Part_6',
#                      'Q30_Part_7', 'Q30_Part_8', 'Q30_Part_9', 'Q30_Part_10', 'Q30_Part_11', 'Q30_Part_12', 'Q31_Part_1',
#                      'Q31_Part_2', 'Q31_Part_3', 'Q31_Part_4', 'Q31_Part_5', 'Q31_Part_6', 'Q31_Part_7', 'Q31_Part_8',
#                      'Q31_Part_9', 'Q31_Part_10', 'Q31_Part_11', 'Q31_Part_12', 'Q32_Part_1', 'Q32_Part_2', 'Q32_Part_3',
#                      'Q32_Part_4', 'Q32_Part_5', 'Q32_Part_6', 'Q32_Part_7', 'Q32_Part_8', 'Q32_Part_9', 'Q32_Part_10',
#                      'Q32_Part_11', 'Q32_Part_12', 'Q33_Part_1', 'Q33_Part_2', 'Q33_Part_3', 'Q33_Part_4', 'Q33_Part_5',
#                      'Q33_Part_6', 'Q33_Part_7', 'Q33_Part_8', 'Q33_Part_9', 'Q33_Part_10', 'Q33_Part_11', 'Q33_Part_12',
#                      'Q34_Part_1', 'Q34_Part_2', 'Q34_Part_3', 'Q34_Part_4', 'Q34_Part_5', 'Q34_Part_6', 'Q34_Part_7',
#                      'Q34_Part_8', 'Q34_Part_9', 'Q34_Part_10', 'Q34_Part_11', 'Q34_Part_12', 'Q10_Encoded']]




# ######## Using Recursive Feature Elimination for Feature Selection #####################

# X = Salaries.iloc[:,0:215]  
# y = Salaries.iloc[:,-1]

# model = RandomForestClassifier(random_state = 42)
# rfe = RFE(model, step=1)
# fit = rfe.fit(X, y)
# rfe_mask = fit.get_support() #list of booleans for selected features
# new_features = [] 

# for bool, feature in zip(rfe_mask, X.columns):
#      if bool:
#              new_features.append(feature)


# new_features.append('Q10_Encoded')
# Salaries_new_features = Salaries[new_features]




# ####################### Looking at Tree Based Classifier ########################

# Salaries.iloc[:,0:215]  
# y = Salaries.iloc[:,-1]   

# model = ExtraTreesClassifier()
# model.fit(X,y)

# # print(model.feature_importances_) 

# # Looking at Feature Importance
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(15).plot(kind='barh')
# plt.show()





# ################# CHI-SQUARE TEST #####################

# X = Salaries.iloc[:,0:215]  
# y = Salaries.iloc[:,-1]  

# top_features = SelectKBest(score_func=chi2, k=15)
# feature_fit = top_features.fit(X,y)

# dfscores = pd.DataFrame(top_features.scores_)
# dfcolumns = pd.DataFrame(X.columns)

# #concat two dataframes for better visualization 

# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(15,'Score'))  #print 10 best features











# ################################## ML MODEL IMPLEMENTATION ####################################
# X = Salaries_new_features.drop(['Q10_Encoded'],axis=1)
# y = Salaries_new_features['Q10_Encoded']

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)


# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,\
#                         train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    
#     plt.figure(figsize=(10,6))
#     plt.title(title)
    
#     if ylim is not None:
#         plt.ylim(*ylim)
        
#     plt.xlabel("Training examples")
#     plt.ylabel(scoring)
    
#     train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
    
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,\
#                      train_scores_mean + train_scores_std, alpha=0.1, \
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,\
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
#     plt.legend(loc="best")
    
#     return plt




# # Calculating Bias and Variance in Data

# def bias(y, y_pred):
#     y = np.array(y)
#     y_pred = np.array(y_pred)
#     return (np.mean((y_pred-y)**2))

# def variance (y, y_pred):
#     y = np.array(y)
#     y_pred = np.array(y_pred)
#     y_avg = np.mean(y_pred)
#     return (np.mean((y_pred-y_avg)**2))




# import warnings
# #warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore")


# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()    
# model.fit(X_train, y_train)



# X3 = Salaries_new_features.iloc[:,0:107]  
# y3 = Salaries_new_features.iloc[:,-1]



# def crossv(model1, X1, y1):    
#     model = model1
#     scaler = StandardScaler()
#     kfold = KFold(n_splits=10)
#     kfold.get_n_splits(X1)

#     accuracy = np.zeros(10)
#     np_idx = 0
#     outcomes = []
#     score_r2=[]
#     bias_cv = []
#     var_cv = []
#     mse = []

#     for train_idx, test_idx in kfold.split(X):
#         X_train, X_test = X1.values[train_idx], X1.values[test_idx]
#         y_train, y_test = y1.values[train_idx], y1.values[test_idx]

#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         model.fit(X_train, y_train)

#         predictions = model.predict(X_test)
#         predict_probability = model.predict_proba(X_test)
        

#         TN = confusion_matrix(y_test, predictions)[0][0]
#         FP = confusion_matrix(y_test, predictions)[0][1]
#         FN = confusion_matrix(y_test, predictions)[1][0]
#         TP = confusion_matrix(y_test, predictions)[1][1]
#         total = TN + FP + FN + TP
#         ACC = (TP + TN) / float(total)

#         accuracy[np_idx] = ACC*100
#         np_idx += 1

#         outcomes.append(accuracy)
#         score_r2.append(r2_score(y_test, predictions))
#         bias_cv.append(bias(y_test, predictions)) 
#         var_cv.append(variance(y_test, predictions)) 
#         mse.append(abs(mean_absolute_error(y_test, predictions)))


#         print ("Fold {}: Accuracy: {}     Bias: {}   Variance: {}  R2 Score: {}   MSE: {}% ".format(np_idx, round(ACC,3), bias(y_test, predictions), variance(y_test, predictions), r2_score(y_test, predictions), abs(mean_absolute_error(y_test, predictions))))  

#     print ("\n Average Score: {}% ({}%)".format(round(np.mean(accuracy),3),round(np.std(accuracy),3)))
#     print("\n Variance: {}%".format(round(np.var(outcomes),3)))
#     print("\n Standard Deviation: {}%".format(np.std(outcomes)))
    
#     plot_learning_curve(model,'Logistic Regression', X1, y1, cv=10)



# model = LogisticRegression()
# crossv(model, X, y) 


# model2 = LogisticRegression(C=10,solver="newton-cg", multi_class="ovr", max_iter=1000, penalty = 'l2')
# crossv(model2, X, y) 





# salary_encode2 = { 
#     0: '0-9,999',
#     1:'10,000-19,999',
#     2:'15,000-29,999',
#     3:'30,000-39,999',
#     4:'40,000-49,999',
#     5:'50,000-59,999',
#     6:'60,000-69,999',
#     7:'70,000-79,999',
#     8:'80,000-89,999',
#     9:'90,000-99,999',
#     10:'100,000-124,999',
#     11:'125,000-149,999',
#     12:'150,000-199,999',
#     13:'200,000-249,999',
#     14: '>250,000'}
# keys_salary = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# values_salary = ['0-9,999','10,000-19,999','15,000-29,999','30,000-39,999','40,000-49,999','50,000-59,999','60,000-69,999','70,000-79,999',
#     '80,000-89,999','90,000-99,999','100,000-124,999','125,000-149,999','150,000-199,999','200,000-249,999','>250,000']




# X = Salaries_new_features.drop(['Q10_Encoded'],axis=1)
# y = Salaries_new_features['Q10_Encoded']

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# model= LogisticRegression()
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# predict_probability = model.predict_proba(X_test)





# def salary_prediction(i,prediction1 ,predict_probability1,y_test1):
#     print('Salary Class: ', y_test1.iloc[i], ' Salary Range: ', salary_encode2[y_test1.iloc[i]])
#     print('Predicted Class: ', prediction1[i],' Salary Range:', salary_encode2[prediction1[i]])
#     plt.figure(figsize=(20,10))
    
#     sns.barplot(keys_salary,predict_probability1[i])
#     plt.title("Probability distribution",size=35,color="black")
#     plt.ylabel("Probability")
#     plt.xlabel("Class Range")
#     plt.show()
    
#     plt.figure(figsize=(20,10))
#     sns.barplot(values_salary,predict_probability1[i])
#     plt.title("Probability distribution",size=35,color="black")
#     plt.ylabel("Probability")
#     plt.xlabel("Salary Range")
#     plt.show()



#  #Looking at probability prediction of line item 12
# salary_prediction(12,prediction ,predict_probability,y_test)




# accuracy_list = []
# f1_score_list = []
# recall_list = []
# precision_list = []


# cols=['newton-cg','lbfgs','liblinear','sag','saga']
# ind=["C=0.001","C=0.01","C=0.05","C=0.1","C=0.5","C=1","C=5","C=10","C=100"]
# #print(len(Recall))
# #print(len(F1score))
# for i in range (0,len(Recall),5):
    
#     accuracy_list.append((accuracies[i:i+5]))
#     f1_score_list.append(F1score[i:i+5])
#     recall_list.append(Recall[i:i+5])
#     precision_list.append(Precision[i:i+5])
    
# Col1=f1_score_list[0]
# Col2=f1_score_list[1]
# Col3=f1_score_list[2]
# Col4=f1_score_list[3]
# Col5=f1_score_list[4]
# Col6=f1_score_list[5]
# Col7=f1_score_list[6]
# Col8=f1_score_list[7]
# Col9=f1_score_list[8]
# f1scores_df=pd.DataFrame([Col1,Col2,Col3,Col4,Col5,Col6,Col7,Col8,Col9],columns=cols,index=ind)
# f1scores_df=100*f1scores_df
# # f1scores_df.plot(figsize=(10,5),title="F1-Score")



st.sidebar.subheader('Machine Learning Model')
st.sidebar.markdown('For this Project Logistic Regression was Selected and was trained using cross validation and grid search to find the optimum model')

option2 = st.sidebar.selectbox('Model Training Visualization ',['None','Model Training','Metrics'])

if option2 == 'Model Training':
    st.image('Model 1.PNG')
    st.image('Model 2.PNG')
    st.image('Model 3.PNG')

if option2 == 'Metrics':
    st.image('Accuracy.PNG')
    st.image('Precision.PNG')
    st.image('f1 Score.PNG')


st.sidebar.markdown('\n\n')


st.sidebar.subheader('Model Predictor')

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

regressor = model["model"]
le_country = model["le_country"]
le_education = model["le_education"]


def show_predict_page():
    " ## Salary Prediction for Data Scientist Using Few Features"

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")


if st.sidebar.checkbox('Predict Software Engineer Salaries'):
    show_predict_page()