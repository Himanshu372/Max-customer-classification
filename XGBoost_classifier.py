#Using XGBoost 
def xgb_classifier(features_np_array, labels_np_array):
    """
    Train a model using features_np_array and labels_np_array
    :param features_np_array:
    :param labels_np_array:
    :return: xgc
    """ 
    xgc = xgb.XGBClassifier(max_depth = 3, learning_rate = .3, n_estimators = 10, objective = 'binary:logistic')
    xgc.fit(features_np_array, labels_np_array)
    return xgc

xgc_model = xgb_classifier(train_features, train_labels)

#Feature importance for XGBoost
fearures_importance(xgc_model, features_dataframe.columns)

#Predicting on validation set using xgb trained model
xgc_pred_val_labels = model_prediction(xgc_model, validation_features)

#Genenrating pred_dataframe for xgc_pred_lable
xgc_pred_val_df = pred_dataframe(val_pred_labels, 'xgc_val_pred')

#F-score for xgb model 
xgb_fscore = f_score(validation_labels, xgc_pred_val_labels)
xgb_fscore

#Predictions for test
xgc_pred_test = model_prediction(xgc_model, test_features_scaled)

#Generating dataframe for predicted test label
xgb_pred_test_df = pred_dataframe(val_pred_labels, 'Prediction')

#Generating submission csv
generate_pred_csv(test_data, xgb_pred_test_df, 'Prediction', 'xgb_submit.csv')