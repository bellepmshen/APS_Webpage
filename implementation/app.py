# Python Standard Library Modules

# 3rd-party installed modules
from flask import Flask, render_template, request
import datetime as dt
import pandas as pd
import pickle
# import sys
# sys.path.append('/Users/belleshen/Documents/VS_Code/LFZ/Assignment/solution/W6/')
from HomeCreditDefaultImplementation import read_pickles, dummy_maintable_merge

# Custom Project Modules

app = Flask(__name__)

@app.route("/")
def index_page():
    return render_template("HTML_Exercise0825.html")

@app.route("/prediction")
def form_page():
    return render_template("Home_Default_Risk_Prediction.html")

@app.route("/result", methods = ["POST"])
def make_prediction():
    #print(request.form['CODE_GENDER']) # to access the value of the CODE_GENDER consider it is a dictionary even though it is a list with tuples from
    print(request.form) # this request.form will let you see collect the key in results from the form
    data = pd.DataFrame([request.form])

    # any empty string from request.form (the return values from the request.form are all string format), it will be converted into None
    #for column in data.columns:
    #    data[column] = data[column].map(lambda x: x if len(x) > 0 else None) 
    # if you have any datetime information in the form, the HTML format (YYYY-MM-DD) might be different from your training dataset
    # so you have to do the conversion too to let the model know this value 

    # data["date"] = data["date"].map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").strftime("%Y%m%d"))
    #numerical_columns = ["AMT_INCOME_TOTAL"]
    #for column in numerical_columns:
    #    data[column] = pd.to_numeric(data[column]) # using 'to_numeric' so you dont have to worry about it should be a int64 or float64 cause it will convert it itself
    category_columns  = [
                         'CODE_GENDER', 'NAME_INCOME_TYPE', 'FLAG_OWN_REALTY', 'OCCUPATION_TYPE'
                        ]
    numerical_columns = [
                         'AMT_INCOME_TOTAL'
                        ]

    for column in category_columns:
        data[column] = data[column].astype("string")
    for column in numerical_columns:
        data[column] = data[column].astype("float64")

    
    # default_values = read_pickles('default_values_pickles')

    # for columns, value in default_values.items():
    #     for i in list(data.columns):
    #         if columns == i:
    #             data[i][0] = value
    
    print(data)
    print(data.info())


    # put the pipeline here: from data cleaning & feature selection & data preprocessing & model predict


    main_cols                = read_pickles('main_cols_pickle')
    cat_cols                 = read_pickles('cat_cols_pickle')
    train_dummies            = read_pickles('train_dummies_cols_pickle')
    model                    = read_pickles('my_model_pickle')

    main_table_test          = data[main_cols]

    test_dummies             = pd.get_dummies(main_table_test[cat_cols])
    test_dummies             = test_dummies.reindex(columns = train_dummies, fill_value = 0)
    test_dummies             = test_dummies[train_dummies]

    main_table_test_1        = dummy_maintable_merge(main_table_test, test_dummies, cat_cols)
    #print(main_table_test_1.info())
    prediction               = model.predict_proba(main_table_test_1)


    return render_template("make_prediction.html", prediction = f"{prediction[0][1]*100: .2f}", 
            gender = data['CODE_GENDER'][0], income_type = data['NAME_INCOME_TYPE'][0],
            clien_income = data["AMT_INCOME_TOTAL"][0], own_realty = data['FLAG_OWN_REALTY'][0],
            occupation_type = data['OCCUPATION_TYPE'][0])
