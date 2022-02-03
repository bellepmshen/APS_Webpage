from features import Feature_Selection
from logist import Logistic_Prep, Logistic_Modeling

if __name__ == '__main__':

    # feature selections:
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
    
    # data prep for Logistic Regression:
    prep = Logistic_Prep(data, main_cols, cat_cols, candidate_cols_new)
    x_res, y_res, x_test, y_test = prep.split_and_resampling()
    train_combined, test_combined = prep.combined()
    train_combined, test_combined, candidate_cols_new_2 = prep.feature_engineering()
    train_scaled_df, test_scaled_df = prep.scaling()
    train_dummies, test_dummies = prep.OneHotEncoding()
    train_combined_1, test_combined_1 = prep.merge_scaled_df()
    
    # Logistic Model:
    
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