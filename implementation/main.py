import pandas as pd
from features import Feature_Selection
from tree_models import (
    Tree_Model_Prep, Gradient_Boosting_Model,
    Random_Forest_Model)
from logist import Logistic_Prep, Logistic_Modeling

def features_defined():
    """This function is for executing feature selections.

    Returns
    -------
    dataframe and lists
        training dataset, main/categorical/numerical columns 
        for model training
    """
    try: 
        cols_selec = Feature_Selection(
        '/Users/belleshen/Documents/VS_Code/Project/'
        'AirPressureSystemFailureDetection/Data/archive/', 
        'aps_failure_training_set.csv'
        )
        data = cols_selec.read_files()
        data = cols_selec.replace_data()
        data = cols_selec.make_numerical()
        cat_cols, nu_cols = cols_selec.get_cat_nu_cols()
        high_colinearity = cols_selec.high_colinearity_cols()
        no_colinearity, high_colinearity = cols_selec.no_colinearity_cols()
        data_scaled_nu = cols_selec.scaling()
        pvalues_box = cols_selec.dimension_reduced()
        candidate_cols = cols_selec.get_candidate_nu_cols()
        main_cols, candidate_cols_new = cols_selec.final_columns()
        
        return data, main_cols, cat_cols, candidate_cols_new
    
    except Exception as err:
        print(err)

def data_prep_logistic(data, main_cols, cat_cols, candidate_cols_new):
    """This function is data prep for logistic regression.

    Parameters
    ----------
    data : dataframe
        training dataset
    main_cols : list
        main columns for model training
    cat_cols : list
        categorical columns for model training
    candidate_cols_new : list
        numerical columns for model trianing

    Returns
    -------
    dataframe
        return train_combined_1 & test_combined_1
    """
    
    try:
        prep = Logistic_Prep(data, main_cols, cat_cols, candidate_cols_new)
        x_res, y_res, x_test, y_test = prep.split_and_resampling()
        train_combined, test_combined = prep.combined()
        train_combined, test_combined, candidate_cols_new_2 = prep.feature_engineering()
        train_scaled_df, test_scaled_df = prep.scaling()
        train_dummies, test_dummies = prep.OneHotEncoding()
        train_combined_1, test_combined_1 = prep.merge_scaled_df()
        
        return train_combined_1, test_combined_1, train_dummies
    
    except Exception as err:
        print(err)

def build_logistic_model(train_combined_1, test_combined_1):
    """This function is for building a Logistic Regression model.

    Parameters
    ----------
    train_combined_1 : dataframe
        prep-training dataset with target feature
    test_combined_1 : dataframe
        prep-testing dataset with target feature
        
    Returns
    -------
    model, array
        return trained logistic model, prediction with labels
        and prediction with probability
    """
    try: 
        build_model = Logistic_Modeling(
            train_combined_1.drop(['class'], axis = 1),
            test_combined_1.drop(['class'], axis = 1),
            train_combined_1['class'],
            test_combined_1['class']
            )
        lg_model = build_model.training()
        lg_pred, lg_pred_prob = build_model.prediction(
            test_combined_1.drop(['class'], axis = 1),
            test_combined_1['class']
        )
        build_model.model_evaluation()
        
        return lg_model, lg_pred, lg_pred_prob
    
    except Exception as err:
        print(err)

def data_prep_tree(data, main_cols, cat_cols, candidate_cols_new):
    """This function is data prep for building tree models.

    Parameters
    ----------
    data : dataframe
        training dataset
    main_cols : list
        main columns for model training
    cat_cols : list
        categorical columns for model training
    candidate_cols_new : list
        numerical columns for model trianing

    Returns
    -------
    dataframe
        return train_dummies_1 & test_dummies_1
    """
    try: 
        tree_prep = Tree_Model_Prep(data, main_cols, cat_cols, candidate_cols_new)
        x_train_1, x_test_1, y_train_1, y_test_1 = tree_prep.split()
        train_combined_2, test_combined_2 = tree_prep.combine()
        train_combined_2, test_combined_2 = tree_prep.feature_engineering()
        train_dummies_1, test_dummies_1 = tree_prep.one_hot_encoding()
        
        return train_dummies_1, test_dummies_1
    
    except Exception as err:
        print(err)

def build_gradient_bst_model(train_dummies_1, test_dummies_1, data):  
    """This function is for building gradient boosting tree model.

    Parameters
    ----------
    train_dummies_1 : dataframe
        prep-training dataset with target feature
    test_dummies_1 : dataframe
        prep-testing dataset with target feature
    data: dataframe
        training dataset without data prep

    Returns
    -------
    model, array
        return trained gradient boosting model, prediction with labels
        and prediction with probability
    """
    try: 
        build_gb_model = Gradient_Boosting_Model(
            train_dummies_1.drop(['class'], axis = 1),
            test_dummies_1.drop(['class'], axis = 1),
            train_dummies_1['class'],
            test_dummies_1['class'],
            data
            )
        gb_model = build_gb_model.training()
        gb_pred, gb_pred_prob = build_gb_model.prediction(
            test_dummies_1.drop(['class'], axis = 1),
            test_dummies_1['class']
        )
        build_gb_model.model_evaluation(
            train_dummies_1.drop(['class'], axis = 1),
             train_dummies_1['class'],
            'gb_train.png'
        )
        build_gb_model.model_evaluation(
            test_dummies_1.drop(['class'], axis = 1),
            test_dummies_1['class'],
            'gb_test.png'
        )
        return gb_model, gb_pred, gb_pred_prob

    except Exception as err:
        print(err)

def build_random_forest_model(train_dummies_1, test_dummies_1, data):
    """This function is for building random forest model.

    Parameters
    ----------
    train_dummies_1 : dataframe
        prep-training dataset with target feature
    test_dummies_1 : dataframe
        prep-testing dataset with target feature
    data: dataframe
        training dataset without data prep

    Returns
    -------
    model, array
        return trained gradient boosting model, prediction with labels
        and prediction with probability
    """
    try: 
        build_rf_model = Random_Forest_Model(
            train_dummies_1.drop(['class'], axis = 1),
            test_dummies_1.drop(['class'], axis = 1),
            train_dummies_1['class'],
            test_dummies_1['class'],
            data
        )
        rf_model = build_rf_model.training()
        rf_pred, rf_pred_prob = build_rf_model.prediction(
            test_dummies_1.drop(['class'], axis = 1),
            test_dummies_1['class']
        )     
        build_rf_model.model_evaluation(
            train_dummies_1.drop(['class'], axis = 1),
                train_dummies_1['class'],
            'rf_train.png'
        )
        build_rf_model.model_evaluation(
            test_dummies_1.drop(['class'], axis = 1),
            test_dummies_1['class'],
            'rf_test.png'
        )
        return rf_model, rf_pred, rf_pred_prob
    
    except Exception as err:
        print(err)


if __name__ == '__main__':

    # feature selection:
    data, main_cols, cat_cols, candidate_cols_new = features_defined()
    
    # logistic regression model:
    train_combined_1, test_combined_1, train_dummies = data_prep_logistic(
        data, main_cols, cat_cols, candidate_cols_new
        )
    lg_model, lg_pred, lg_pred_prob = build_logistic_model(
        train_combined_1, test_combined_1
    )   

    # tree model data prep:
    train_dummies_1, test_dummies_1 = data_prep_tree(
        data, main_cols, cat_cols, candidate_cols_new
        )

    # gradient boosting tree model:
    gb_model, gb_pred, gb_pred_prob = build_gradient_bst_model(
        train_dummies_1, test_dummies_1, data
    ) 
    
    # random forest model:
    rf_model, rf_pred, rf_pred_prob = build_random_forest_model(
        train_dummies_1, test_dummies_1, data
    )
    