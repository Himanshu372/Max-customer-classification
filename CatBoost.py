#Using Cat Boost  
def cb_classifier(features_np_array, labels_np_array):
    """
    Train a model using features_np_array and labels_np_array
    :param features_np_array:
    :param labels_np_array:
    :return: nb
    """ 
    cbc = CatBoostClassifier(iterations=2, depth=2, learning_rate=0.5, loss_function='Logloss', custom_loss = ['F1'])
    cbc.fit(train_features, train_labels)
    return cbc

#Predictions for test
cbc_pred_test = model_prediction(nb_model,test_features_scaled)

#Generating a dataframe for the predicted test labels
cbc_pred_test_df = pred_dataframe(cbc_pred_test, 'Prediction')

#Generating submission csv
generate_pred_csv(test_data, cbc_pred_test_df, 'Prediction', 'cb_submit.csv')