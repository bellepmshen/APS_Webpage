import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

class Feature_Selection:
    """This class is for feature selection.
    """
    
    def __init__(self, path, filename):
        """Initiate the class

        Parameters
        ----------
        path : str
            file path
        filename : str
            filename
        """
        
        self.path = path
        self.filename = filename
    
    def read_files(self):
        """reading files

        Returns
        -------
        [dataframe]
            [data]
        """
        try:
            
            f = self.path + self.filename
            data = pd.read_csv(f)
            
            self.data = data
            
            return self.data
        
        except Exception as err:
            print(err)
        

    def replace_data(self):
        """replace 'na' with -1 in the dataset and make target feature into
        binary

        Returns
        -------
        [dataframe]
            [data]
        """
        
        try:
            self.data = self.data.applymap(lambda x: -1 if x == 'na' else x)

            self.data['class'] = self.data['class'].map(
                lambda x: 1 if x == 'pos' else 0
                )
            
            return self.data
        
        except Exception as err:
            print(err)
    
    def make_numerical(self):
        """Make the whole dataset into numerical columns

        Returns
        -------
        [dataframe]
            [data]
        """
        try:
            for i in self.data.columns:
                self.data[i] = pd.to_numeric(self.data[i])
            
            return self.data
        
        except Exception as err:
            print(err)

    def get_cat_nu_cols(self):
        """define cat_cols & nu_cols

        Returns
        -------
        [list]
            [categorical & numerical column lists]
        """
        try:

            feature_importance_df = \
            pd.read_csv(
                '/Users/belleshen/Documents/VS_Code/Project/AirPressureSystemFailureDetection/feature_importance_df.csv'
                )
            feature_importance_df_sort = feature_importance_df.sort_values(
                by = 'feature_importance', ascending=False)

            data_nunique_df = pd.DataFrame(self.data.nunique(), 
                                           columns= ['nunique_values'])
            data_cat_df = data_nunique_df[data_nunique_df.nunique_values < 100]
            
            cat_cols = list(
                set(feature_importance_df_sort.head(65)['columns']) & 
                set(data_cat_df.index)
                )
            nu_cols = list(
                set(feature_importance_df_sort.head(65)['columns']) - 
                set(data_cat_df.index)
                )
            
            self.cat_cols = cat_cols
            self.nu_cols = nu_cols
            
            return self.cat_cols, self.nu_cols
        
        except Exception as err:
            print(err)
    
    def high_colinearity_cols(self):
        """get high colineariy columns

        Returns
        -------
        [list]
            [return high colinearity columns list]
        """
        try:
            data_nu_corr_matrix = self.data[self.nu_cols].corr()
            
            high_colinearity = []
            for i in data_nu_corr_matrix.columns:
                a_series = data_nu_corr_matrix[i].sort_values(ascending=False)
                idx = list(a_series[a_series > .9].index)
                high_colinearity.append(idx)

            print(f"high_colinearity length: {len(high_colinearity)}")
            
            self.high_colinearity = high_colinearity
            
            return self.high_colinearity
        
        except Exception as err:
            print(err)
    
    def no_colinearity_cols(self):
        """get no colinearity columns and get new high colinearity columns

        Returns
        -------
        [list]
            [return high_colinearity & no_colinearity columns list]
        """
        try:
            no_colinearity = [i for i in self.high_colinearity if len(i) == 1]

            for i in no_colinearity:
                if i in self.high_colinearity:
                    self.high_colinearity.remove(i)

            print(f"high_colinearity length: {len(self.high_colinearity)}, \
                  no_colinearity length:{len(no_colinearity)}"
                  )    
            self.no_colinearity = no_colinearity
            
            return self.no_colinearity, self.high_colinearity  
        
        except Exception as err:
            print(err)
    
    def scaling(self):
        """scaling the data before doing linear regression

        Returns
        -------
        [dataframe]
            [return a dataframe with scaled numerical columns]
        """
        try:
            scaler = MinMaxScaler()
            scaler.fit(self.data[self.nu_cols])
            data_scaled_nu = pd.DataFrame(
                scaler.transform(self.data[self.nu_cols]), 
                columns = self.nu_cols
                )
            self.data_scaled_nu = data_scaled_nu
            
            return self.data_scaled_nu
        
        except Exception as err:
            print(err)
 
    def dimension_reduced(self):
        """Find p-values of these high colinearity columns by doing linear 
        regression of them and get p-value of each column

        Returns
        -------
        [list]
            [return a list of p-value of each column after linear regression]
        """
        try:
            pvalues_box = []
            for i in self.high_colinearity:
                model = sm.OLS(self.data['class'], self.data_scaled_nu[i])
                res = model.fit()
                pvalues_box.append(res.pvalues)
                
                self.pvalues_box = pvalues_box
            return self.pvalues_box   
        
        except Exception as err:
            print(err)        
    
    def get_candidate_nu_cols(self):
        """get candidate numerical columns

        Returns
        -------
        [list]
            [return a list of candidate (numerical) columns]
        """
        try:
            # find the minimum of pvalues of each high colinearity combination:

            pvalues_df = pd.concat(self.pvalues_box, axis = 1)

            candidate_cols_nonzero = []
            candidate_cols_zero = []

            for i in range(len(pvalues_df)):
                if pvalues_df.iloc[:, i].dropna().min() != 0:
                    nonzero_col = pvalues_df.iloc[:, i].dropna().idxmin()
                    candidate_cols_nonzero.append(nonzero_col)
                else:
                    zero_col = sorted(
                        pvalues_df.iloc[:, i].dropna()
                        [pvalues_df.iloc[:, i].dropna() == 0].index.to_list()
                        ) 
                    candidate_cols_zero.append(zero_col[0]) 
                    # select the first element in an alphabetically sorted list

            candidate_cols = candidate_cols_nonzero.copy()
            for i in candidate_cols_zero:
                candidate_cols.append(i)

            print(f"candidate_cols length (before): {len(candidate_cols)}")

            for i in self.no_colinearity:
                for j in i:
                    candidate_cols.append(j)

            print(
                f"candidate_cols length (after): {len(candidate_cols)}, \
                candidate_cols length (w/o duplicates): \
                {len(set(candidate_cols))}"
                )
            
            self.candidate_cols = candidate_cols
            
            return self.candidate_cols
        
        except Exception as err:
            print(err)
    
    def final_columns(self):
        """define main_cols

        Returns
        -------
        [list]
            [return main_cols list and candidate_cols_new list]
        """
        try:
            
            candidate_cols_new = list(set(self.candidate_cols))
            main_cols          = self.cat_cols + candidate_cols_new 

            print(f"candidate_cols_new length: {len(candidate_cols_new)}, \
                  main_cols length: {len(main_cols)}")
            
            self.main_cols = main_cols
            self.candidate_cols_new = candidate_cols_new
            
            return self.main_cols, self.candidate_cols_new
        
        except Exception as err:
            print(err)               

    



