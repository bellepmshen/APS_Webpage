import pandas as pd
from features import Feature_Selection
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (plot_roc_curve, precision_score, recall_score, 
                             confusion_matrix, accuracy_score)


class Tree_Model_Prep:
    """This class is for data prep for gradient boosting trees and 
    random forest models."""
    
    def __init__(self, data, main_cols, cat_cols, candidate_cols_new):
        """initiate the class

        Parameters
        ----------
        data : [dataframe]
            trainining dataset
        main_cols : [list]
            main columns for model training
        cat_cols : [list]
            categorical columns for model training
        candidate_cols_new : [list]
            numerical columns for model trianing
        """
        self.data = data
        self.main_cols = main_cols
        self.cat_cols = cat_cols
        self.candidate_cols_new = candidate_cols_new
        
    def split(self):
          
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                self.data[self.main_cols], self.data[['class']], 
                test_size = .3, random_state = 42, stratify = self.data['class']
                )

            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
            
            return self.x_train, self.x_test, self.y_train, self.y_test 
                   
        except Exception as err:
            print(err)
    
    def combine(self):
        """This function is for x and y combined.

        Returns
        -------
        [dataframe]
            return train_combined & test_combined
        """
        try:
            train_combined = self.x_train.merge(self.y_train, how = 'left', 
                                         right_index = True, 
                                         left_index = True
                                         )
            test_combined = self.x_test.merge(self.y_test, how = 'left', 
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
        [dataframe]
            train_combined & test_combined after feature engineering
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
            
            return self.train_combined, self.test_combined
        
        except Exception as err:
            print(err)          
            
    def one_hot_encoding(self):
        """One Hot Encoding for categorical columns

        Returns
        -------
        [dataframe]
            return dummies & numerical datasets of train & test 
        """
        try:
            train_dummies = pd.get_dummies(
                self.train_combined, columns = self.cat_cols
                )
            test_dummies  = pd.get_dummies(
                self.test_combined, columns = self.cat_cols
                )

            test_dummies  = test_dummies.reindex(
                columns = train_dummies.columns, fill_value = 0
                )
            test_dummies  = test_dummies[train_dummies.columns]
            
            self.train_dummies = train_dummies
            self.test_dummies = test_dummies
            
            return self.train_dummies, self.test_dummies
        
        except Exception as err:
            print(err)
    
    
class Gradient_Boosting_Model:
    """This class is for building a gradient boosting tree model."""
    
    def __init__(self, xtrain, xtest, ytrain, ytest, data):
        """initialize the class

        Parameters
        ----------
        xtrain : [dataframe]
            depenedent features in training dataset
        xtest : [dataframe]
            depenedent features in test dataset
        ytrain : [dataframe/series]
            indepenedent features in training dataset
        ytest : [dataframe/series]
            indepenedent features in test dataset 
        data : [dataframe/series]
            original dataset without train and test split         
        """
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest   
        self.data = data   

    def training(self):
        """model training for gradient boosting tree

        Returns
        -------
        [model]
            trained gradient boosting tree model
        """
        try:
            model = xgb.XGBClassifier(
                random_state = 42, learning_rate = .3, max_depth = 9, 
                colsample_bytree = 0.7, gamma = 0.0, min_child_weight = 3,
                scale_pos_weight = 56, n_estimators = 900              
            )
            model.fit(self.xtrain, self.ytrain)
            
            self.model = model
            
            return self.model
        
        except Exception as err:
            print(err)
    
    def plotting(self):
        """Generate fig & ax""" 
        
        try:
            fig, ax = plt.subplots(1, 1, figsize = (10, 6), dpi = 120)
            
            return fig, ax
        
        except Exception as err:
            print(err)
            
    def model_evaluation(self, x, y, fig_name):  
        """plot ROC and show AUC for both train & test dataset

        Parameters
        ----------
        x : [dataframe]
            xtrain or xtest
        y: [dataframe/series]
            ytrain or ytest
        fig_name : [str]
            output figure name with extension (ex: .png)
        """
        try:
            
            fig, ax = self.plotting()
            plot_roc_curve(self.model, x, y, ax = ax)  
            fig.savefig(fig_name, bbox_inches = 'tight') 
            
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

class Random_Forest_Model(Gradient_Boosting_Model):
    """This class is a child class of Gradient_Boosting_Model and
    is to build a random forest model"""
    
    def __init__(self, xtrain, xtest, ytrain, ytest, data):
        super().__init__(xtrain, xtest, ytrain, ytest, data)
    
    def training(self):
        """model training for random forest 

        Returns
        -------
        [model]
            trained random forest model
        """
        try:
            model = xgb.XGBRFClassifier(
                random_state = 42, n_estimator = 400, min_samples_split = 2,
                min_samples_leaf = 4, max_features = 'sqrt', max_depth = 40,
                bootstrap = False, scale_pos_weight = 
                self.data['class'].value_counts()[0]/
                self.data['class'].value_counts()[1]          
            )
            model.fit(self.xtrain, self.ytrain)
            
            self.model = model
            
            return self.model
        
        except Exception as err:
            print(err)
              