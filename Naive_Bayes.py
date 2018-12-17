Using Naive Bayes  
def nb_classifier(features_np_array, labels_np_array):
    """
    Train a model using features_np_array and labels_np_array
    :param features_np_array:
    :param labels_np_array:
    :return: nb
    """ 
    nbc = GaussianNB()
    nbc.fit(features_np_array, labels_np_array)
    return nbc

nb_model = nb_classifier(train_features, train_labels)

#Predictions for test using Naive Bayes

nb_pred_test = model_prediction(nb_model,test_features_scaled)

#Generating a dataframe for the predicted test labels
nb_pred_test_df = pred_dataframe(nb_pred_test, 'Prediction')

#Generating submission csv
generate_pred_csv(test_data, nb_pred_test_df, 'Prediction', 'nb_submit.csv')