#!/usr/bin/env python
# coding: utf-8

# # The Impact of Campaign Financing on the Outcomes of the 2016 U.S. House Races

# ## Introduction
# 
# Campaign financing is a critical factor in the landscape of United States elections. The funds raised and spent by political candidates can significantly influence their visibility, messaging, and overall competitiveness. This project focuses on analyzing the 2016 U.S. House of Representatives races to understand how campaign financing affected election outcomes, but as we gear up for another general election in just a few weeks, I was curious to explore the data and identify any trends that could be useful in determining potential 2024 house race outcomes. 
# 
# ## Project Goal
# 
# The primary goal of this analysis is to determine the extent to which financial factors impacted the results of the 2016 House races, and to identify whether other factors play a significant role in winning an election. By examining data on total contributions, expenditures, and other financial metrics for each candidate, we aim to:
# 
# - **Assess the relationship between campaign finances and electoral success.**
# - **Identify key financial predictors of winning candidates.**
# - **Utilize supervised machine learning models to predict election outcomes based on financial data.**
# 
# Through this investigation, we hope to gain insights into the influence of money in politics and how it shapes the democratic process.
# 
# ## Data Source
# Source: https://www.kaggle.com/datasets/danerbland/electionfinance
# 
# The dataset utilized in this project was assembled to investigate the potential of predicting congressional election results using campaign finance reports from the period leading up to the 2016 election (January 1, 2015, through October 19, 2016). Each entry in the dataset represents a candidate and includes comprehensive information about their campaign finances, such as total contributions, total expenditures, state, district, office, and election outcomes.
# 
# **Data Collection:**
# 
# - **Campaign Finance Data:** Obtained directly from the Federal Election Commission (FEC), ensuring official and accurate financial records of each candidate's campaign activities.
# - **Election Results and Vote Totals:** Sourced from CNN's election results page for the 2016 U.S. House races, providing verified outcomes and vote counts for winners of contested races.
# 
# **Public Access and Licensing:**
# 
# The dataset is publicly available and has been provided under the **CC0: Public Domain** license, allowing unrestricted use, distribution, and reproduction in any medium. This open access facilitates transparency and encourages further research into the effects of campaign financing on election outcomes.
# 
# ### Citation
# 
# *Federal Election Commission. (2016).* Candidate Summary File [Data set]. Retrieved from [http://www.fec.gov/finance/disclosure/metadata/metadataforcandidatesummary.shtml](http://www.fec.gov/finance/disclosure/metadata/metadataforcandidatesummary.shtml)
# 
# *CNN Politics. (2016).* Election Results: U.S. House. Retrieved from [http://www.cnn.com/election/2016/results](http://www.cnn.com/election/2016/results)
# 
# *Dataset compiled and made available on Kaggle:*
# 
# Unknown Author. (2017). *Campaign Finance and Election Results* [Data set]. Kaggle. [https://www.kaggle.com/datasets/benhamner/campaign-finance-and-election-results](https://www.kaggle.com/datasets/benhamner/campaign-finance-and-election-results)
# 
# ---

# ## Data Descriptions and Initial Exploration
# 
# 

# In[1]:


import pandas as pd

df = pd.read_csv('CandidateSummaryAction1.csv')


# In[2]:


#Observing the first few rows of data
df.head()

#Pulling a concise summary of the DataFrame
df.info()

#Pulling descriptive statistics for numerical columns
df.describe()

#Pulling descriptive statistics for all columns (including categorical)
df.describe(include='all')


# ## Data Overview
# 
# The dataset comprises campaign finance and election result information for candidates in the 2016 U.S. House races. It was assembled to investigate the potential of predicting election outcomes based on campaign finance reports leading up to the election.
# 
# ### Dataset Summary
# 
# - **Number of Samples (Rows):** 1,814 candidates
# - **Number of Features (Columns):** 51 features
# - **Data Size:** Approximately 722.9 KB
# 
# ### Data Structure
# 
# The dataset includes a mix of categorical and numerical features:
# 
# - **Categorical Features:** 48 columns
# - **Numerical Features:** 3 columns (`can_off_dis`, `can_zip`, `votes`)
# 
# ### Key Features Description
# 
# - **`can_id`**: Candidate identification number.
# - **`can_nam`**: Candidate's full name.
# - **`can_off`**: Office the candidate is running for (e.g., 'H' for House).
# - **`can_off_sta`**: State abbreviation where the candidate is running.
# - **`can_off_dis`**: Congressional district number.
# - **`can_par_aff`**: Candidate's party affiliation.
# - **`can_inc_cha_ope_sea`**: Candidate status (Incumbent, Challenger, Open Seat).
# - **`tot_con`**: Total contributions received by the candidate.
# - **`tot_dis`**: Total disbursements/expenditures made by the candidate.
# - **`winner`**: Indicates if the candidate won ('Y') or lost (blank).
# - **`votes`**: Number of votes received by the winning candidates in contested House races.
# 
# ### Data Types
# 
# - **Numerical Features (Float64):**
#   - `can_off_dis` (District number)
#   - `can_zip` (Candidate's ZIP code)
#   - `votes` (Vote counts for winners)
# 
# - **Categorical Features (Object):**
#   - Remaining 48 columns, including candidate information and financial data.
# 
# ### Missing Values
# 
# - **General Overview:**
#   - Some features have missing values, which need to be addressed during data cleaning.
#   
# - **Examples of Missing Data:**
#   - `can_off_dis`: Missing in 2 entries.
#   - `can_par_aff`: Missing in 1 entry.
#   - `votes`: Available for 379 entries (vote counts for winners only).
#   - `winner`: Only 471 entries have 'Y' (winners), the rest are blank (losers).
# 
# ### Data Source and Compilation
# 
# - **Single-Table Format:** The data is presented in a single table but compiled from multiple sources.
#   - **Campaign Finance Data:** Sourced directly from the Federal Election Commission (FEC) ([FEC Metadata](http://www.fec.gov/finance/disclosure/metadata/metadataforcandidatesummary.shtml)).
#   - **Election Results:** Vote totals and results for House races obtained from CNN's election results page.
# 

# ## Cleaning the Data and Preprocessing: 

# In[3]:


# Data Cleaning and Preprocessing

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('CandidateSummaryAction1.csv')

# Display initial data information
print("Initial Data Information:")
print(df.info())

# Replace empty strings in 'winner' with 'N' and fill NaN with 'N'
df['winner'] = df['winner'].fillna('N').replace('', 'N')

# Convert 'winner' to a categorical variable
df['winner'] = df['winner'].astype('category')

# Exclude 'votes' from being dropped
exclude_cols = ['votes']

# Calculate threshold for dropping columns
threshold = len(df) * 0.5

# Identify columns to drop (excluding 'votes')
cols_to_drop = [col for col in df.columns if col not in exclude_cols and df[col].isnull().sum() > threshold]

# Drop the identified columns
df = df.drop(columns=cols_to_drop)

# Convert financial columns to numeric, handling parentheses, dollar signs, and commas
financial_cols = ['tot_con', 'tot_dis', 'votes']
for col in financial_cols:
    # Check if column exists in DataFrame
    if col in df.columns:
        # Remove dollar signs, commas, and closing parentheses
        df[col] = df[col].replace('[\$,)]', '', regex=True)
        # Replace opening parenthesis with a negative sign
        df[col] = df[col].replace('\(', '-', regex=True)
        # Replace empty strings with NaN
        df[col] = df[col].replace('', np.nan)
        # Convert the column to float
        df[col] = df[col].astype(float)
    else:
        print(f"Column '{col}' not found in the DataFrame.")

# Impute missing values in 'tot_con' and 'tot_dis' with zeros
df[['tot_con', 'tot_dis']] = df[['tot_con', 'tot_dis']].fillna(0)

# Remove rows where 'tot_con' and 'tot_dis' are both zero (assumed non-competitive candidates)
df = df[~((df['tot_con'] == 0) & (df['tot_dis'] == 0))]

# Handle outliers in 'tot_con' using the IQR method
Q1 = df['tot_con'].quantile(0.25)
Q3 = df['tot_con'].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df['tot_con'] < (Q1 - 1.5 * IQR)) | (df['tot_con'] > (Q3 + 1.5 * IQR))
df = df[~outlier_condition]

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

# Display cleaned data information
print("\nCleaned Data Information:")
print(df.info())


# In[4]:


df.head()


# In[5]:


# Renaming Columns for Clarity

# Create a copy of the DataFrame to avoid modifying the original data
df_clean = df.copy()

# Define a dictionary mapping old column names to new, clearer names
column_renames = {
    'can_id': 'Candidate_ID',
    'can_nam': 'Candidate_Name',
    'can_off': 'Office',
    'can_off_sta': 'State',
    'can_off_dis': 'District',
    'can_par_aff': 'Party_Affiliation',
    'can_inc_cha_ope_sea': 'Candidate_Status',
    'can_str1': 'Street_Address',
    'can_cit': 'City',
    'can_sta': 'Address_State',
    'tot_con': 'Total_Contributions',
    'tot_dis': 'Total_Disbursements',
    'tot_rec': 'Total_Receipts',
    'ope_exp': 'Operating_Expenditures',
    'cas_on_han_clo_of_per': 'Cash_On_Hand_Close',
    'net_con': 'Net_Contributions',
    'net_ope_exp': 'Net_Operating_Expenditures',
    'cov_sta_dat': 'Coverage_Start_Date',
    'cov_end_dat': 'Coverage_End_Date',
    'winner': 'Winner',
    'votes': 'Votes'
}

# Rename the columns in the DataFrame
df_clean.rename(columns=column_renames, inplace=True)

# Display the first few rows of the updated DataFrame
df_clean.head()


# ### Summary of Cleaning
# 
# - **Created a copy of the original DataFrame** called `df_clean` to preserve the original data.
# - **Defined a `column_renames` dictionary** that maps the original column names to more descriptive ones.
# - **Used the `rename()` method** with `inplace=True` to update the column names in `df_clean`.
# - **Displayed the first few rows** using `df_clean.head()` to verify the changes.
# 
# This renaming improves the clarity of the data, making it easier to understand and work with in su

# ## Adding the Contribution-to-Expenditure Ratio

# In[6]:


# Adding the Contribution-to-Expenditure Ratio

# Avoid division by zero by replacing zeros in Total_Disbursements with NaN
df_clean['Total_Disbursements'].replace({0: np.nan}, inplace=True)

# Calculate the ratio
df_clean['Contrib_Exp_Ratio'] = df_clean['Total_Contributions'] / df_clean['Total_Disbursements']

# Fill any resulting NaN values with zero (if desired)
df_clean['Contrib_Exp_Ratio'].fillna(0, inplace=True)

# Display the first few rows to verify the new column
df_clean[['Candidate_Name', 'Total_Contributions', 'Total_Disbursements', 'Contrib_Exp_Ratio']].head()


# ### Explanation:
# 
# - **Handling Zero Disbursements:**
#   - Replaced zeros in `Total_Disbursements` with `NaN` to prevent division by zero errors.
# - **Calculating the Contribution-to-Expenditure Ratio:**
#   - Computed the ratio by dividing `Total_Contributions` by `Total_Disbursements`.
# - **Handling NaN Values:**
#   - Filled `NaN` values in the new `Contrib_Exp_Ratio` column with zeros, assuming that candidates with no disbursements have a ratio of zero.
# - **Verification:**
#   - Displayed relevant columns to verify that the new `Contrib_Exp_Ratio` has been added correctly.
# 
# ### Interpretation:
# 
# The **Contribution-to-Expenditure Ratio** indicates how much a candidate has raised in contributions relative to their expenditures. A ratio:
# 
# - **Greater than 1:** The candidate raised more money than they spent.
# - **Equal to 1:** The candidate's contributions exactly matched their expenditures.
# - **Less than 1:** The candidate spent more than they raised (possibly using loans, personal funds, or previous cash on hand).
# 
# This metric can provide insights into the financial efficiency and fundraising effectiveness of a campaign.

# ## Visual Exploration

# In[7]:


# Exploratory Data Analysis (EDA) 

# Ensure necessary libraries are imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visuals
sns.set_style('whitegrid')

# Using the cleaned DataFrame from previous steps (df_clean)

# Distribution of Total Contributions
plt.figure(figsize=(10,6))
sns.distplot(df_clean['Total_Contributions'], bins=30, kde=True, hist=True)
plt.title('Distribution of Total Contributions')
plt.xlabel('Total Contributions ($)')
plt.ylabel('Density')
plt.show()


# **Observation:**
# 
# - The distribution of **Total Contributions** appears to be right-skewed, indicating that most candidates receive lower amounts of contributions, while a few candidates receive very high amounts.
# - This skewness is typical in financial data, where a small number of candidates may dominate fundraising.

# In[8]:


# Distribution of Contribution-to-Expenditure Ratio using distplot
plt.figure(figsize=(10,6))
sns.distplot(df_clean['Contrib_Exp_Ratio'], bins=30, kde=True, hist=True, color='green')
plt.title('Distribution of Contribution-to-Expenditure Ratio')
plt.xlabel('Contribution-to-Expenditure Ratio')
plt.ylabel('Density')
plt.show()


# **Observation:**
# 
# - The **Contribution-to-Expenditure Ratio** distribution is concentrated around values less than 2, with a long tail extending to higher ratios.
# - A ratio greater than 1 indicates candidates raised more than they spent, while less than 1 indicates they spent more than they raised.
# 
# **Interpretation:**
# 
# - The majority of candidates have a ratio close to 1, suggesting that they spend amounts roughly equivalent to what they raise.
# 

# In[9]:


# Boxplot of Total Contributions by Winner
plt.figure(figsize=(10,6))
sns.boxplot(x='Winner', y='Total_Contributions', data=df_clean)
plt.title('Total Contributions by Election Outcome')
plt.xlabel('Election Outcome')
plt.ylabel('Total Contributions ($)')
plt.show()


# **Observation:**
# 
# - **Winners** tend to have higher total contributions compared to **losers**.
# - The median contribution for winners is significantly higher than that for losers.
# - There are more outliers (high contributions) among winners.
# 
# **Interpretation:**
# 
# - This suggests a positive relationship between the amount of money raised and the likelihood of winning an election.
# - Candidates who raise more funds may have better resources for campaigning, increasing their chances of success.

# In[10]:


# Scatter Plot of Total Contributions vs. Total Disbursements
plt.figure(figsize=(10,6))
sns.scatterplot(x='Total_Contributions', y='Total_Disbursements', hue='Winner', data=df_clean)
plt.title('Total Contributions vs. Total Disbursements by Election Outcome')
plt.xlabel('Total Contributions ($)')
plt.ylabel('Total Disbursements ($)')
plt.legend(title='Winner')
plt.show()


# **Observation:**
# 
# - There is a strong positive correlation between **Total Contributions** and **Total Disbursements**.
# - Winners generally have higher contributions and disbursements.
# - The points representing winners are clustered in the higher ranges of both contributions and expenditures.
# 
# **Interpretation:**
# 
# - Candidates typically spend what they raise.
# - Successful candidates are those who both raise and spend more money on their campaigns.

# In[11]:


# Calculate correlation matrix
corr_matrix = df_clean[['Total_Contributions', 'Total_Disbursements', 'Contrib_Exp_Ratio', 'Votes']].corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# **Interpretation:**
# 
# - The strong correlation between contributions and disbursements confirms that candidates spend what they raise.
# - The positive correlation between votes and financial variables indicates that higher fundraising and spending are associated with receiving more votes.

# In[12]:


# Countplot of Party Affiliation
plt.figure(figsize=(10,6))
sns.countplot(x='Party_Affiliation', data=df_clean, order=df_clean['Party_Affiliation'].value_counts().index)
plt.title('Distribution of Party Affiliation')
plt.xlabel('Party Affiliation')
plt.ylabel('Number of Candidates')
plt.xticks(rotation=45)
plt.show()


# In[13]:


# Stacked Bar Chart of Election Outcome by Party Affiliation
party_winner = df_clean.groupby(['Party_Affiliation', 'Winner']).size().unstack(fill_value=0)

# Plotting
party_winner.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Election Outcome by Party Affiliation')
plt.xlabel('Party Affiliation')
plt.ylabel('Number of Candidates')
plt.legend(title='Winner', labels=['Lost', 'Won'])
plt.xticks(rotation=45)
plt.show()


# ## Initial Modeling Using Random Forest

# In[14]:


# Feature Importance using Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

features = ['Total_Contributions', 'Total_Disbursements', 'Contrib_Exp_Ratio']
target = 'Winner'

# Encode target variable
le = LabelEncoder()
df_clean['Winner_Encoded'] = le.fit_transform(df_clean['Winner'])  # 'Y' -> 1, 'N' -> 0

# Split data into X and y
X = df_clean[features]
y = df_clean['Winner_Encoded']

# Handle missing values if any
X = X.fillna(0)

# Train a simple Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Plot feature importances
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False))
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# **Observation:**
# 
# - **Total_Contributions** is identified as the most important feature, followed by **Total_Disbursements** and **Contrib_Exp_Ratio**.
# 
# **Interpretation:**
# 
# - The amount a candidate raises is slightly more indicative of election success than the amount they spend.
# - This could be because raising money in house races is primarily driven by grassroots donors, indicating a higher level of support. 
# 

# ## Evaluating Whether Close Races (by Contribution Margin <10%) Can be Utilized

# In[15]:


df.head()


# In[16]:



import pandas as pd

# Step 1: Create 'Election_ID' by combining 'can_off_sta' (State) and 'can_off_dis' (District)
df['can_off_dis'] = df['can_off_dis'].astype(str)
df['Election_ID'] = df['can_off_sta'] + '-' + df['can_off_dis']

# Step 2: Convert 'net_con' (Net Contributions) to numeric
# Remove any commas or dollar signs and handle errors without dropping rows
df['net_con'] = df['net_con'].replace({'\$': '', ',': ''}, regex=True)
df['net_con'] = pd.to_numeric(df['net_con'], errors='coerce')

# Step 3: Fill missing 'net_con' values with zero to retain data
df['net_con'] = df['net_con'].fillna(0)

# Step 4: Calculate total net contributions per election
election_funds = df.groupby('Election_ID')['net_con'].sum().reset_index()
election_funds.rename(columns={'net_con': 'Total_Election_Net_Contributions'}, inplace=True)

# Step 5: Merge total contributions back into the main DataFrame
df = df.merge(election_funds, on='Election_ID', how='left')

# Step 6: Calculate the percentage of net contributions for each candidate
df['Contributions_Percentage'] = (df['net_con'] / df['Total_Election_Net_Contributions']) * 100

# Step 7: Sort candidates within each election by contributions percentage
df = df.sort_values(by=['Election_ID', 'Contributions_Percentage'], ascending=[True, False])

# Step 8: Assign rank within each election based on contributions percentage
df['Rank'] = df.groupby('Election_ID')['Contributions_Percentage'].rank(method='first', ascending=False)

# Step 9: Get the next candidate's contributions percentage within each election
df['Next_Contributions_Percentage'] = df.groupby('Election_ID')['Contributions_Percentage'].shift(-1)

# Step 10: Calculate the funding margin for the top candidate in each election
df['Funding_Margin'] = df['Contributions_Percentage'] - df['Next_Contributions_Percentage']

# Set 'Funding_Margin' to NaN for candidates who are not ranked 1
df.loc[df['Rank'] != 1, 'Funding_Margin'] = None

# Step 11: Identify close races based on funding margin (e.g., margin <= 10%)
close_races = df[(df['Funding_Margin'] <= 10) & (df['Funding_Margin'].notnull())]

# Step 12: Display the number of close races based on fundraising
num_close_races = close_races['Election_ID'].nunique()
print(f"Number of close races based on fundraising margin <= 10%: {num_close_races}")

# Display the first few close races
close_races.head()


# In[36]:


# Focusing on predicting the 'winner' based on fundraising and other available features for the close races

import pandas as pd
import numpy as np

# Select relevant features for modeling
features = [
    'net_con',             # Net contributions
    'tot_con',             # Total contributions
    'tot_dis',             # Total disbursements
    'ind_con',             # Individual contributions
    'oth_com_con',         # Other committee contributions
    'can_par_aff',         # Candidate party affiliation
    'can_inc_cha_ope_sea', # Candidate status (incumbent, challenger, etc.)
    # Add any other relevant features available in your data
]

# Ensure that these columns are in the DataFrame
model_data = close_races[features + ['winner']].copy()

# Handle missing values
model_data.dropna(inplace=True)

# Convert financial columns to numeric
financial_cols = ['net_con', 'tot_con', 'tot_dis', 'ind_con', 'oth_com_con']
for col in financial_cols:
    # Convert to string and remove dollar signs and commas
    model_data[col] = model_data[col].astype(str).str.replace(r'[\$,]', '', regex=True)
    # Convert back to numeric
    model_data[col] = pd.to_numeric(model_data[col], errors='coerce')

# Drop any remaining rows with missing financial data
model_data.dropna(subset=financial_cols, inplace=True)


# In[37]:


# Feature Engineering

# Create ratios and interaction terms
model_data['Contrib_Disburse_Ratio'] = model_data['tot_con'] / model_data['tot_dis']
model_data['Indiv_Total_Contrib_Ratio'] = model_data['ind_con'] / model_data['tot_con']
model_data['OtherCom_Total_Contrib_Ratio'] = model_data['oth_com_con'] / model_data['tot_con']

# Replace infinite values with zero (in case of division by zero)
model_data.replace([np.inf, -np.inf], 0, inplace=True)

# Log transformation to reduce skewness
for col in financial_cols:
    model_data['Log_' + col] = np.log1p(model_data[col])

# Encode categorical variables
model_data = pd.get_dummies(model_data, columns=['can_par_aff', 'can_inc_cha_ope_sea'], drop_first=True)


# In[38]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select numerical features
X = model_data.drop(['winner'], axis=1)
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Calculate VIF
vif_data = pd.DataFrame()
vif_data['Feature'] = numerical_features
vif_data['VIF'] = [variance_inflation_factor(X[numerical_features].values, i) for i in range(len(numerical_features))]

print(vif_data)


# In[39]:


# As'tot_con' and 'net_con' have high VIF, we can drop one of them or combine them

# Drop 'tot_con' if necessary
X.drop(['tot_con'], axis=1, inplace=True)

# Recalculate VIF
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
vif_data = pd.DataFrame()
vif_data['Feature'] = numerical_features
vif_data['VIF'] = [variance_inflation_factor(X[numerical_features].values, i) for i in range(len(numerical_features))]

print(vif_data)


# ### Interpretation of VIF Results
# VIF (Variance Inflation Factor) measures how much the variance of a regression coefficient is inflated due to multicollinearity
# Due to the very high rates of multicollinearity we need to leverage a larger data set with limited and diverse features.

# ## Model Comparisons

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[45]:


print(model_data.columns)


# In[46]:


# Define features and target
X = model_data.drop('winner', axis=1)
y = model_data['winner'].map({'Y': 1, 'N': 0})  # Encode target as binary

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)


# In[47]:


# Check class distribution before handling imbalance
print("\nClass Distribution Before Handling Imbalance:")
print(y_train.value_counts())


# In[51]:


from sklearn.utils import resample

# Combine training data
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority = train_data[train_data['winner'] == 0]
minority = train_data[train_data['winner'] == 1]

# Downsample majority class
majority_downsampled = resample(
    majority,
    replace=False,  # sample without replacement
    n_samples=len(minority),  # to match minority class
    random_state=42
)

# Combine minority class with downsampled majority class
train_downsampled = pd.concat([minority, majority_downsampled])

# Separate features and target
X_train_resampled = train_downsampled.drop('winner', axis=1)
y_train_resampled = train_downsampled['winner']

# Check class distribution after undersampling
print("\nClass Distribution After Random Undersampling:")
print(y_train_resampled.value_counts())


# In[56]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit on resampled training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)  # Use X_test directly

# Convert scaled arrays back to DataFrames for compatibility
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_resampled.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)  # Corrected to X_test.columns

# Now, X_train_scaled and X_test_scaled are ready for modeling


# In[57]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Define parameter grid for hyperparameter tuning
param_grid_lr = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear'],  # 'liblinear' supports both l1 and l2
    'max_iter': [100, 200, 500]
}

# Initialize Logistic Regression
lr = LogisticRegression(random_state=42)

# Initialize GridSearchCV
grid_search_lr = GridSearchCV(
    estimator=lr,
    param_grid=param_grid_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Perform Grid Search
grid_search_lr.fit(X_train_scaled, y_train_resampled)

# Best model from Grid Search
best_lr = grid_search_lr.best_estimator_

# Predictions on test set
y_pred_lr = best_lr.predict(X_test_scaled)

# Evaluation
print("Best Logistic Regression Parameters:", grid_search_lr.best_params_)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression ROC AUC Score:", roc_auc_score(y_test, best_lr.predict_proba(X_test_scaled)[:, 1]))


# In[58]:


from sklearn.ensemble import RandomForestClassifier

# Define parameter grid for hyperparameter tuning
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Perform Grid Search
grid_search_rf.fit(X_train_scaled, y_train_resampled)

# Best model from Grid Search
best_rf = grid_search_rf.best_estimator_

# Predictions on test set
y_pred_rf = best_rf.predict(X_test_scaled)

# Evaluation
print("Best Random Forest Parameters:", grid_search_rf.best_params_)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest ROC AUC Score:", roc_auc_score(y_test, best_rf.predict_proba(X_test_scaled)[:, 1]))


# In[59]:


from sklearn.ensemble import GradientBoostingClassifier

# Define parameter grid for hyperparameter tuning
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1],
    'min_samples_split': [2, 5]
}

# Initialize Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42)

# Initialize GridSearchCV
grid_search_gb = GridSearchCV(
    estimator=gb,
    param_grid=param_grid_gb,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Perform Grid Search
grid_search_gb.fit(X_train_scaled, y_train_resampled)

# Best model from Grid Search
best_gb = grid_search_gb.best_estimator_

# Predictions on test set
y_pred_gb = best_gb.predict(X_test_scaled)

# Evaluation
print("Best Gradient Boosting Parameters:", grid_search_gb.best_params_)
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))
print("Gradient Boosting ROC AUC Score:", roc_auc_score(y_test, best_gb.predict_proba(X_test_scaled)[:, 1]))


# ## Comparative Analysis
# 
# Random Forest Classifier outperforms both Gradient Boosting and Logistic Regression in terms of Accuracy and F1-Score, achieving a balanced performance across both classes.
# All models achieved a perfect ROC AUC Score of 1.0, which, given the  small test set, might not be indicative of true model performance and could be a result of **overfitting** or chance.
# 
# #### Model Effectiveness:
# 
# Random Forest demonstrated superior performance with balanced F1-Scores for both classes and higher accuracy, making it the preferred model among the three.
# 
# Gradient Boosting and Logistic Regression showed similar performance, with challenges in predicting the majority class effectively.
# 
# #### Feature Importance:
# 
# While not explicitly provided in the output, feature importance analysis from the Random Forest model likely highlighted key fundraising metrics (e.g., log_net_con) and candidate attributes (e.g., can_inc_cha_ope_sea_INCUMBENT) as significant predictors of election outcomes.
# 
# #### Class Imbalance Handling:
# 
# The use of Random Undersampling may have contributed to Random Forest's better performance by balancing the dataset, allowing the model to learn equally from both classes.
# Logistic Regression and Gradient Boosting might not have handled class imbalance as effectively, leading to poorer performance on the majority class.
# 
# #### ROC AUC Score Consideration:
# 
# The perfect ROC AUC Score across all models is unusual and likely a result of the small test set, which doesn't provide a reliable assessment of model generalizability.
# 

# In[ ]:




