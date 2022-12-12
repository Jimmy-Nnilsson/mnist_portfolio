import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix

class TitanicDataPreparation(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        all_columns = []

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if "Age" in X.columns:
            X = self.age(X)
        if "SibSp" in X.columns and "Parch" in X.columns:
            X = self.family(X)
        if "Ticket" in X.columns:
            X = self.Ticket(X)
        if "Cabin" in X.columns:
            X = self.Cabin(X)
        if "Name" in X.columns:
            X = self.Title(X)
        if "Sex" in X.columns:
            X = self.Sex(X)
        # X = pd.get_dummies(X)
        return X

    def age(self, X):
        age_lims = [12, 18, 45, 65]
        age_classes = ['child', "teen", "young", "mid", "senior"]

        X.loc[pd.isna(X["Age"]), "Aclass"] = 'unknown'
        for i,a in enumerate(age_lims):
            if i==0:
                print(age_classes[i])
                X.loc[(~pd.isna(X["Age"])) & (X["Age"] <= age_lims[i]), "Aclass"] = age_classes[i]
            else:
                X.loc[(~pd.isna(X["Age"])) & (X["Age"] <= age_lims[i]) & (X["Age"] > age_lims[i-1]), "Aclass"] = age_classes[i]
        X.loc[(~pd.isna(X["Age"])) & (X["Age"] > age_lims[i]), "Aclass"] = age_classes[i+1]
        X.drop('Age', axis=1, inplace=True)
        return X

    def family(self, X):
        X["Family"] = X['SibSp'] + X['Parch'] + 1
        X['Alone'] = 0
        X.loc[X['Family'] <= 1, "Alone"] = 1
        X.drop(['SibSp', 'Parch'], axis=1, inplace=True)
        return X

    def Ticket(self, X):
        if "Ticket" in X.columns:
            X[['Ticketlab', 'Ticket_no']] = X['Ticket'].str.split(" ", 1, expand=True)
            X.loc[(X['Ticketlab'].str.isnumeric()) & pd.isna(X['Ticket_no']), 'Ticket_no'] = X['Ticketlab']
            X.loc[X['Ticketlab'] == X['Ticket_no'], "Ticketlab"] = np.nan
            X.drop('Ticket', axis=1, inplace=True)
            X['Ticket_no'] = X['Ticket_no'].str.replace(". ", "").replace("Basl541", "8651541")
            X['Ticket_no']= X['Ticket_no'].astype('Int64')
            # X.drop("Ticketlab", axis=1, inplace=True)
            return X

    def Cabin(self, X):
        X['Cabin_class'] = X['Cabin'].apply(clean_cabin, ret_char=True)
        X['Cabin_no'] = X['Cabin'].apply(clean_cabin)
        X.drop("Cabin", axis=1, inplace=True)
        return X
    def Title(self, X):
        X['Title'] = X['Name'].apply(self.get_title)
        X['Title'].value_counts()
        X.drop('Name', axis=1, inplace=True)
        return X

    def get_title(self, st):
        st = st.split(' ')
        for s in st:
            if "." in s:
                return s
        return st
    def Sex(self, X):
        X['Female'] = 0
        X.loc[X['Sex'] == "female", 'Female'] = 1
        X.drop('Sex', axis=1, inplace=True)
        return X


def clean_cabin(x, ret_char=False):
    no_list = []
    char_list = []
    no = str("")
    if pd.isna(x):
        return(x)
    for l in x:
        if l.isnumeric():
            no = (no + str(l))
        elif not l.isnumeric() and l != " ":
            char_list.append(l)
        if l == " ":
            if no != "":
                no_list.append(int(no))
                no=""
    if not ret_char :
        if no != "":
            no_list.append(int(no))
        return(np.mean(no_list))
    else:
        char_list = str(set(char_list)).replace("{", "").replace("}", "").replace("'", "")
        char_list = char_list.replace(",", "").replace(" ", "")
        return(char_list)