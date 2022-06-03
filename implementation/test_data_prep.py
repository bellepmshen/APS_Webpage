import pandas as pd
from features import Feature_Selection

class Logist_Data_Prep:
    """This class is for data prep for logistic regression model
    """
    def __init__(self, scaler, candidate_cols_new_2, cat_cols, train_dummies):
        """Initialize the class

        Parameters
        ----------
        scaler : scaler
            the scaler from training dataset
        candidate_cols_new_2 : list 
            candidate columns 
        cat_cols : list
            categorical columns
        train_dummies : list
            the dummy columns from training dataset for alignment
        """
        self.scaler = scaler
        self.candidate_cols_new_2 = candidate_cols_new_2
        self.cat_cols = cat_cols
        self.train_dummies = train_dummies
    
    
    def get_data(self):
        """This function is for get the test data

        Returns
        -------
        dataframe
            the test data
        """
        try:
            cols_selec = Feature_Selection(
            '/Users/belleshen/Documents/VS_Code/Project/'
            'AirPressureSystemFailureDetection/Data/archive/', 
            'aps_failure_test_set.csv'
            )
            test_data = cols_selec.read_files()
            test_data = cols_selec.replace_data()
            test_data = cols_selec.make_numerical()
            
            self.test_data = test_data
            
            return self.test_data
        except Exception as err:
            print(err)

    def feature_engineering(self):
        """This function is for doing feature engineering

        Returns
        -------
        dataframe
            the dataframe with feature engineered columns
        """
        try: 
            self.test_data['aq_000 + ag_003'] = self.test_data.apply(
                lambda x: x['aq_000'] + x['ag_003'] ,axis = 1
                )
            self.test_data['ci_000 + cj_000'] = self.test_data.apply(
                lambda x: x['ci_000'] + x['cj_000'] ,axis = 1
                )

            self.test_data.drop(
                ['aq_000', 'ag_003', 'ci_000', 'cj_000'], 
                axis = 1, 
                inplace = True)
            
            return self.test_data
        
        except Exception as err:
            print(err)
    
    def scaling(self):
        """This function is for scaling

        Returns
        -------
        dataframe
            the scaled df
        """
        try:
            test_data_scaled_df = pd.DataFrame(
                self.scaler.transform(self.test_data[self.candidate_cols_new_2]), 
                columns = self.candidate_cols_new_2, 
                index = self.test_data.index
                )
            
            self.test_data_scaled_df = test_data_scaled_df
            
            return self.test_data_scaled_df
        except Exception as err:
            print(err)
    
    def OneHotEncoding(self):
        """This function is for one hot encoding

        Returns
        -------
        dataframe
            the df with one hot encoding
        """
        try: 
            test_data_dummies = pd.get_dummies(
                self.test_data, columns = self.cat_cols
                )
            test_data_dummies.drop(
                self.candidate_cols_new_2, axis = 1, inplace = True
                )
            
            test_data_dummies = test_data_dummies.reindex(
                columns = self.train_dummies, fill_value = 0
                )
            test_data_dummies = test_data_dummies[self.train_dummies]
            
            self.test_data_dummies = test_data_dummies
            
            return self.test_data_dummies
        
        except Exception as err:
            print(err)
    
    def merge_scaled_df(self):
        """This function is for merging scaling & dummy dataframe

        Returns
        -------
        dataframe
            merged dataframe
        """
        try:
            test_data_combined = self.test_data_scaled_df.merge(
                self.test_data_dummies, how = 'left', left_index = True, 
                right_index = True)
            
            self.test_data_combined = test_data_combined
            
            return self.test_data_combined
        
        except Exception as err:
            print(err)
            

class Tree_Data_Prep:
    """This class is for data prep for tree models
    """
    def __init__(self, cat_cols, train_dummies_xgb):
        """Initialize the class

        Parameters
        ----------
        cat_cols : list
            categorical columns
        train_dummies_xgb : list
            dummy columns from training dataset for tree model
        """
        self.cat_cols = cat_cols
        self.train_dummies_xgb = train_dummies_xgb
        
        
    def get_data(self):
        """This function is for get the test data

        Returns
        -------
        dataframe
            the test data
        """
        try:
            cols_selec = Feature_Selection(
            '/Users/belleshen/Documents/VS_Code/Project/'
            'AirPressureSystemFailureDetection/Data/archive/', 
            'aps_failure_test_set.csv'
            )
            test_data = cols_selec.read_files()
            test_data = cols_selec.replace_data()
            test_data = cols_selec.make_numerical()
            
            self.test_data = test_data
            
            return self.test_data
        except Exception as err:
            print(err)

    def feature_engineering(self):
        """This function is for doing feature engineering

        Returns
        -------
        dataframe
            the dataframe with feature engineered columns
        """        
        try:
            self.test_data['aq_000 + ag_003'] = self.test_data.apply(
                lambda x: x['aq_000'] + x['ag_003'] ,axis = 1
                )
            self.test_data['ci_000 + cj_000'] = self.test_data.apply(
                lambda x: x['ci_000'] + x['cj_000'] ,axis = 1
                )

            self.test_data.drop(['aq_000', 'ag_003', 'ci_000', 'cj_000'], 
                                axis = 1, inplace = True
                                )
            
            return self.test_data
        
        except Exception as err:
            print(err)
            
    def OneHotEncoding(self):
        """This function is for one hot encoding

        Returns
        -------
        dataframe
            the df with one hot encoding
        """       
        try: 
            test_data_dummies_xgb = pd.get_dummies(
                self.test_data, columns = self.cat_cols)

            test_data_dummies_xgb = test_data_dummies_xgb.reindex(
                columns = self.train_dummies_xgb, fill_value = 0
                )
            test_data_dummies_xgb = test_data_dummies_xgb[self.train_dummies_xgb]
            
            self.test_data_dummies_xgb = test_data_dummies_xgb
            
            return self.test_data_dummies_xgb
        
        except Exception as err:
            print(err)             

    