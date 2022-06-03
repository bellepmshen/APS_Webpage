import pickle
import pandas as pd
from test_data_prep import Logist_Data_Prep, Tree_Data_Prep
from sklearn.metrics import (precision_score, recall_score, 
                             confusion_matrix, accuracy_score)

def get_pickle():
    """This function is for getting pickle objects.

    Returns
    -------
    scaler: the scaler for logistic model prep
    candidate_cols_new_2: the list for candidate columns from training
    cat_cols: the list for categorical columns
    train_dummies: the list for logistic model dummy variables alignment
    train_dummies_1: the list for tree models dummy variables alignment
    logist_mode: the trained logistic model
    gb_model: the trained gradient boosting model
    rf_model: the trained random forest model
    
    """
    try:
        #load scaler for logistic regression model:
        with open('scaler.pickle', 'rb') as f:
            scaler = pickle.load(f)
        
        #load candidate_cols_new_2:
        with open('candidate_cols_new_2.pickle', 'rb') as f_1:
            candidate_cols_new_2 = pickle.load(f_1)   
        
        #load cat_cols:
        with open('cat_cols', 'rb') as f_2:
            cat_cols = pickle.load(f_2)
        
        #load train_dummies feature names for logistic regression dummies variables alignment:
        with open('train_dummies_t.pickle', 'rb') as f_3:
            train_dummies = pickle.load(f_3)    
        
        #load train_dummies_1 feature names for tree model dummies variables alignment:
        with open('train_dummies_4.pickle', 'rb') as f_4:
            train_dummies_1 = pickle.load(f_4)   
        
        #load logistic model:
        with open('logistic_model.pickle', 'rb') as f_5:
            logist_model = pickle.load(f_5)
        
        #load GB model:
        with open('gb_model_3.pickle', 'rb') as f_6:
            gb_model = pickle.load(f_6)
        
        #load RF model:
        with open('rf_model_1.pickle', 'rb') as f_7:
            rf_model = pickle.load(f_7)
           
        return (scaler, candidate_cols_new_2, cat_cols, train_dummies, 
                train_dummies_1, logist_model, gb_model, rf_model) 
        
    except Exception as err:
        print(err)                    


def test_prep(scaler, candidate_cols_new_2, cat_cols, train_dummies, train_dummies_1):
    """This function is for test data prep for prediction

    Parameters
    ----------
    scaler : scaler
        the scaler for logistic model 
    candidate_cols_new_2 : list
        the list for candidate columns from training
    cat_cols : list
        the list for categorical columns
    train_dummies :list
        the list for logistic model dummy variables alignment
    train_dummies_1 : list
        the list for tree models dummy variables alignment

    Returns
    -------
    pandas dataframe
        test_data_combined, the preped datafor logistic model
        test_data_dummies_xgb, the preped data for tree models
        test_data, the orginal test data
        
    """
    
    try:
        logist_test_p = Logist_Data_Prep(
            scaler, candidate_cols_new_2, cat_cols, train_dummies 
            )
        test_data = logist_test_p.get_data()
        test_data = logist_test_p.feature_engineering()
        test_data_scaled_df = logist_test_p.scaling()
        test_data_dummies = logist_test_p.OneHotEncoding()
        test_data_combined = logist_test_p.merge_scaled_df()

        tree_test_p = Tree_Data_Prep(cat_cols, train_dummies_1)
        test_data_tree = tree_test_p.get_data()
        test_data_tree = tree_test_p.feature_engineering()
        test_data_dummies_xgb = tree_test_p.OneHotEncoding()
        
        return test_data_combined, test_data_dummies_xgb, test_data

    except Exception as err:
        print(err)

def inference(model, data, tree = False, ytree_test = False):
    """This function is for model prediction

    Parameters
    ----------
    model : model
        the trained model, ex: logistic model or tree model
    data : dataframe 
        the preped dataframe with target feature, when the dataframe is for logistic model
        the preped dataframe without target feature, when the dataframe is for tree model
    tree : bool, optional
        for separating is it for logistic or tree model, by default False
    ytree_test : bool, optional
        the y-true array or series for tree model prediction, by default False

    Returns
    -------
    array
        prediction, the prediction labels
        prediction_prob, the prediction probability
    """
    
    try:
        if tree == False: 
            prediction = model.predict(data.drop(['class'], axis = 1))
            prediction_prob = model.predict_proba(data.drop(['class'], axis = 1))
            
            print(f"accuracy score: {accuracy_score(data['class'], prediction)}")
            print(f"precision score: {precision_score(data['class'], prediction)}, recall score: {recall_score(data['class'], prediction)}")
            print(confusion_matrix(data['class'], prediction))
            
            return prediction, prediction_prob 
        else:
            prediction = model.predict(data)
            prediction_prob = model.predict_proba(data)
        
            print(f"accuracy score: {accuracy_score(ytree_test, prediction)}")
            print(f"precision score: {precision_score(ytree_test, prediction)}, recall score: {recall_score(ytree_test, prediction)}")
            print(confusion_matrix(ytree_test, prediction))
        
            return prediction, prediction_prob            

    except Exception as err:
            print(err) 
    
      

def ensembled(logist_pred_prob, gb_pred_prob, rf_pred_prob,  test_data):
    """This function is for ensembling the model prediction results

    Parameters
    ----------
    logist_pred_prob : array
        the logistic model prediction probability
    gb_pred_prob : array
       the GB model prediction probability
    rf_pred_prob : array
        the RF model prediction probability
    test_data : dataframe
        the orginal dataframe

    Returns
    -------
    dataframe
        the ensembled prediction dataframe 
    """
    try:
        logistc_pred_test_1 = pd.DataFrame(
            logist_pred_prob, 
            columns = ['Logistic_pred_0', 'Logistic_pred_1'], 
            index = test_data_combined.index
            )
        
        xgb_pred_test_2 = pd.DataFrame(
            gb_pred_prob, columns = ['xgb_pred_0', 'xgb_pred_1'], 
            index = test_data_dummies_xgb.index
            )
        
        xgb_rf_pred_test_2 = pd.DataFrame(
            rf_pred_prob, 
            columns = ['xgb_rf_pred_0', 'xgb_rf_pred_1'], 
            index = test_data_dummies_xgb.index
            )
        
        test_data_prediction_df = logistc_pred_test_1.merge(
            xgb_pred_test_2, how = 'left', left_index = True, 
            right_index = True).merge(xgb_rf_pred_test_2, how = 'left', 
                                      left_index = True, right_index = True
                                      )

        test_data_prediction_df['ensemble_pred_1'] = \
            test_data_prediction_df.apply(
                lambda x: (x['xgb_pred_1'] * .5) + (x['xgb_rf_pred_1'] * .25) +\
                    (x['Logistic_pred_1'] * .25), axis = 1
                    )

        test_data_prediction_df['ensemble_pred_binary_1'] = \
            test_data_prediction_df['ensemble_pred_1'].map(
                lambda x: 1 if x > .8 else 0
                )

        print(f"accuracy score: {accuracy_score(test_data['class'], test_data_prediction_df['ensemble_pred_binary_1'] )}")
        print(f"precision score: {precision_score(test_data['class'], test_data_prediction_df['ensemble_pred_binary_1'] )}, recall score: {recall_score(test_data['class'], test_data_prediction_df['ensemble_pred_binary_1'] )}")
        print(confusion_matrix(test_data['class'], test_data_prediction_df['ensemble_pred_binary_1']))
    
        return test_data_prediction_df
    
    except Exception as err:
        print(err)

if __name__ == "__main__":
    
    scaler, candidate_cols_new_2, cat_cols, train_dummies, train_dummies_1, logist_model, gb_model, rf_model = get_pickle()
    test_data_combined, test_data_dummies_xgb, test_data = test_prep(scaler, candidate_cols_new_2, cat_cols, train_dummies, train_dummies_1)
    
    #get prediction:
    logist_pred, logist_pred_prob = inference(logist_model, test_data_combined)
    gb_pred, gb_pred_prob = inference(gb_model, test_data_dummies_xgb, True, test_data['class'])
    # print(test_data_dummies_xgb)
    rf_pred, rf_pred_prob = inference(rf_model, test_data_dummies_xgb, True, test_data['class'])
    
    # ensemble
    test_data_prediction_df = ensembled(logist_pred_prob, gb_pred_prob, rf_pred_prob,  test_data)
    
    
    