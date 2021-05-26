import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import key_dict
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

key_dict = key_dict.key_dict
def return_dict_mean_value(query_feature):
    '''
    'key_dict' is dictionary object which has all the Categorical variable names store as KEY and its mean as VALUE.
    This is function is used to return mean value for query_feature.

    KEY ==>
    Value ==> Mean value response to that key

    query_feature ==>  Desired key
    Return ==> Categorical feature and their corresponding mean values.
    '''

    result_dict = dict()

    for feature_name, values in key_dict.items():
        if feature_name == query_feature:
            for key in values:
                result_dict.update([(key, values[key])])

                print(key + ':', values[key])
    return result_dict


# return_dict_mean_value('dish_liked')

@app.route('/')
def home():
    return render_template('index.html')





@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features=[]
    for i in request.form.values():
        print("i :-",i)
        features.append(i)
    #print(key_dict.key_dict)
    df = pd.DataFrame([features],columns=['votes', 'cost', 'online_order', 'book_table', 'rest_type', 'location', 'cuisines',
                               'dish_liked'])
    #features = [int(x) for x in request.form.values()]
    #features = [for x in request.form.values()]
    ###############################################################


    dict_online = return_dict_mean_value('online_order')

    dict_book_table = return_dict_mean_value('book_table')
    dict_rest_type = return_dict_mean_value('rest_type')
    dict_location = return_dict_mean_value('location')
    dict_cuisines = return_dict_mean_value('cuisines')
    dict_dish_liked = return_dict_mean_value('dish_liked')

    df['mean_online_order'] = df['online_order'].map(dict_online)
    #print("checking :-", df['mean_online_order'])

    df['mean_book_table'] = df['book_table'].map(dict_book_table)

    df['mean_rest_type'] = df['rest_type'].map(dict_rest_type)

    df['mean_location'] = df['location'].map(dict_location)

    df['mean_cuisines'] = df['cuisines'].map(dict_cuisines)

    df['mean_dish_liked'] = df['dish_liked'].map(dict_dish_liked)
    ###################################################################

    df = df[['votes', 'cost', 'mean_online_order', 'mean_book_table', 'mean_rest_type', 'mean_location', 'mean_cuisines',
         'mean_dish_liked']]



    print("df :- \n",df)


    #final_features = [np.array(df)]

    final_features = df.to_numpy()
    print('\n')
    print('final_feature',final_features)
    print(type(final_features))
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)