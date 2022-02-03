import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (plot_roc_curve, precision_score, recall_score, 
                             confusion_matrix, accuracy_score)
import matplotlib.pyplot as plt
from features import Feature_Selection

class Logistic_Prep:
    """This class is for building a Logistic Regression model."""

    def __init__(self, data, main_cols, cat_cols, candidate_cols_new):
        """Initiate the class

        Parameters
        ----------
        data : pandas dataframe
            [trainining dataset]
        main_cols : list
            [main columns for model training]
        cat_cols : list
            [categorical columns for model training]
        candidate_cols_new : list
            [numerical columns for model trianing]
        """
        
        self.data = data
        self.main_cols = main_cols
        self.cat_cols = cat_cols
        self.candidate_cols_new = candidate_cols_new
        
    def split_and_resampling(self):
        """train and test split and oversampling to balance the dataset

        Returns
        -------
        [dataframe]
            [return xtrain & ytrain after balancing dataset and xtest & ytest]
        """
        try:

            x_train, x_test, y_train, y_test = train_test_split(
                self.data[self.main_cols], self.data[['class']], test_size = .3,
                random_state = 42, stratify = self.data['class']
                )

            smt_1 = SMOTENC(random_state = 42, 
                            categorical_features = list(range(
                                len(self.cat_cols)
                                ))
                            )
            x_res, y_res = smt_1.fit_resample(x_train, y_train)

            self.x_res = x_res
            self.y_res = y_res            
            self.x_test = x_test
            self.y_test = y_test
            
            return self.x_res, self.y_res, self.x_test, self.y_test
        
        except Exception as err:
            print(err)
    
    def combined(self):
        """This function is for x and y combined.
        
        Returns
        -------
        [dataframe]
            [return train_combined & test_combined]
        """
        try:
            train_combined = self.x_res.merge(self.y_res, how = 'left', 
                                         right_index = True, 
                                         left_index = True
                                         )
            test_combined  = self.x_test.merge(self.y_test, how = 'left', 
                                          right_index = True, 
                                          left_index = True
                                          )
            
            self.train_combined = train_combined
            self.test_combined = test_combined

            return self.train_combined, self.test_combined
        
        except Exception as err:
            print(err)
           
    def feature_engineering(self):
        """This function is for feature engineering.

        Returns
        -------
        [dataframe & list]
            [train_combined & test_combined and a list of new features after 
            feature engineering]
        """
        try:
            self.train_combined['aq_000 + ag_003'] = self.train_combined.apply(
                lambda x: x['aq_000'] + x['ag_003'] ,axis = 1
                )
            self.train_combined['ci_000 + cj_000'] = self.train_combined.apply(
                lambda x: x['ci_000'] + x['cj_000'] ,axis = 1
                )
            self.test_combined['aq_000 + ag_003']  = self.test_combined.apply(
                lambda x: x['aq_000'] + x['ag_003'] ,axis = 1
                )
            self.test_combined['ci_000 + cj_000']  = self.test_combined.apply(
                lambda x: x['ci_000'] + x['cj_000'] ,axis = 1
                )

            self.train_combined.drop(
                ['aq_000', 'ag_003', 'ci_000', 'cj_000'], axis = 1, 
                inplace = True
                )
            self.test_combined.drop(
                ['aq_000', 'ag_003', 'ci_000', 'cj_000'], axis = 1, 
                inplace = True
                )

            # redefine the candidate cols with new engineered features:

            feature_eng_cols = ['aq_000', 'ag_003', 'ci_000', 'cj_000']
            candidate_cols_new_1 = list(
                set(self.candidate_cols_new) - set(feature_eng_cols)
                )
            new_feature_cols = ['aq_000 + ag_003', 'ci_000 + cj_000']
            candidate_cols_new_2 = candidate_cols_new_1 + new_feature_cols
            
            self.candidate_cols_new_2 = candidate_cols_new_2
            
            return self.train_combined, self.test_combined, self.candidate_cols_new_2
        
        except Exception as err:
            print(err)
            

    def scaling(self):
        """scaling for Logistic Regression
        
        Returns
        -------
        [dataframe]
            [scaled train & test numerical column dataframes]
        """
        try:
            scaler = MinMaxScaler()
            scaler.fit(self.train_combined[self.candidate_cols_new_2])
            train_scaled_df = pd.DataFrame(
                scaler.transform(
                    self.train_combined[self.candidate_cols_new_2]
                    ), columns = self.candidate_cols_new_2, 
                index = self.train_combined.index
                )
            test_scaled_df  = pd.DataFrame(
                scaler.transform(self.test_combined[self.candidate_cols_new_2]
                                 ), columns = self.candidate_cols_new_2, 
                index = self.test_combined.index
                )
            
            self.train_scaled_df = train_scaled_df
            self.test_scaled_df = test_scaled_df
            
            return self.train_scaled_df, self.test_scaled_df
        
        except Exception as err:
            print(err)

    
    def OneHotEncoding(self):
        """One Hot Encoding for categorical columns

        Returns
        -------
        [dataframe]
            [return train_dummies and test_dummies]
        """
        try:
            train_dummies = pd.get_dummies(
                self.train_combined, columns = self.cat_cols
                )
            test_dummies  = pd.get_dummies(
                self.test_combined, columns = self.cat_cols)

            test_dummies  = test_dummies.reindex(
                columns = train_dummies.columns, fill_value = 0
                )
            test_dummies  = test_dummies[train_dummies.columns]

            self.train_dummies = train_dummies
            self.test_dummies = test_dummies
            
            return self.train_dummies, self.test_dummies
        
        except Exception as err:
            print(err)
            
    def merge_scaled_df(self):
        """merge dummies & scaled_df together

        Returns
        -------
        [dataframe]
            [return train_combined_1 & test_combined_1]
        """
        try:
            # drop candidate_cols_new_2
            self.train_dummies.drop(
                self.candidate_cols_new_2, axis = 1, inplace = True
                )
            self.test_dummies.drop(
                self.candidate_cols_new_2, axis = 1, inplace = True
                )
            
            # merge scaled df & dummies:

            train_combined_1 = self.train_scaled_df.merge(
                self.train_dummies, how = 'left', left_index = True, 
                right_index = True
                )
            test_combined_1  = self.test_scaled_df.merge(
                self.test_dummies, how = 'left', left_index = True, 
                right_index = True
                )
        
            self.train_combined_1 = train_combined_1
            self.test_combined_1 = test_combined_1
            
            return self.train_combined_1, self.test_combined_1
        
        except Exception as err:
            print(err)
            
class Logistic_Modeling:
    """This class is for Logistic Regression, model evaluation and prediction."""           
        
    def __init__(self, xtrain, xtest, ytrain, ytest):
        """initialize the class

        Parameters
        ----------
        xtrain : [dataframe]
            [depenedent features in training dataset]
        xtest : [dataframe]
            [depenedent features in test dataset]
        ytrain : [dataframe/series]
            [indepenedent features in training dataset]
        ytest : [dataframe/series]
            [indepenedent features in test dataset]
        """
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest           
    
    def training(self):
        """Model training (Logistic Regression)

        Returns
        -------
        [model]
            [trained Logistic Regression Model]
        """
        try:
            model = LogisticRegression(
                random_state=0, penalty = 'l2', C = 110, solver = 'liblinear'
                ).fit(self.xtrain, self.ytrain)
            
            self.model = model
            
            return self.model
        except Exception as err:
            print(err)
        
    def model_evaluation(self):  
        """plot ROC and show AUC for both train & test dataset
        """
        try:
            
            fig, ax = self.plotting()
            plot_roc_curve(self.model, self.xtrain, self.ytrain, ax = ax)  
            fig.savefig('roc_train.png', bbox_inches = 'tight') 
            
            fig, ax = self.plotting() 
            plot_roc_curve(self.model, self.xtest, self.ytest, ax = ax)
            fig.savefig('roc_test.png', bbox_inches = 'tight') 
            
        except Exception as err:
            print(err)
    
    def plotting(self):
        """Generate fig & ax""" 
        
        try:
            fig, ax = plt.subplots(1, 1, figsize = (10, 6), dpi = 120)
            
            return fig, ax
        
        except Exception as err:
            print(err)
    

    def prediction(self, x, y):
        """Make a prediction

        Parameters
        ----------
        x : [dataframe]
            [xtest or x_actual_test data]
        y : [dataframe]
            [ytest or y_actual_test data]

        Returns
        -------
        [array]
            [binary & probability prediction]
        """
        try:
            pred = self.model.predict(x)
            pred_prob = self.model.predict_proba(x)
            print(f"accuracy score: {accuracy_score(y, pred)}")
            print(f"precision score: {precision_score(y, pred)}")
            print(f"recall score: {recall_score(y, pred)}")
            print(confusion_matrix(y, pred))

            return pred, pred_prob
        
        except Exception as err:
            print(err)



    