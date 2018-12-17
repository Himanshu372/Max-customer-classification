#Training Random forest classifier

def random_forest_classifier(features_np_array, labels_np_array):
    """
    Train a model using features_np_array and labels_np_array
    :param features_np_array:
    :param labels_np_array:
    :return: rfc
    """ 
    rfc = RandomForestClassifier(n_estimators = 10)
    rfc.fit(features_np_array, labels_np_array)
    return rfc    
    

rfc_model = random_forest_classifier(train_features, train_labels)

#Feature Importance

def fearures_importance(model, trainset_columns_list):
    """
    Features importance for a trained model
    :param model:
    :param trainset_columns_list:
    :return: imp_df
    """ 
    imp_df = pd.DataFrame(model.feature_importances_,index = trainset_columns_list,columns=['Imp']).sort_values('Imp',ascending=False)
    return imp_df    
    
    
fearures_importance(rfc_model, features_dataframe.columns)

#Predicting on validation set using trained model

def model_prediction(trained_model, features_np_array):
    """
    Predictions based on trained model
    :param trained_model:
    :param features_np_array:
    :return: pred_np_array
    """ 
    pred_np_array = trained_model.predict(features_np_array)
    return pred_np_array    
    
val_pred_labels = model_prediction(rfc_model, validation_features)

#Converting pred_labels into a dataframe with given column name

def pred_dataframe(pred_labels_array, column_name):
    """
    Converting pred_labels to dataframe
    :param pred_labels_array:
    :param pred_labels_array:
    :return: pred_df
    """ 
    pred_df = pd.DataFrame(pred_labels_array, columns = [column_name])
    return pred_df

val_labels_df = pred_dataframe(val_pred_labels, 'val_labels')

#Calculating F-score 
def f_score(y_true, y_pred):
    """
    Returns f-score for a binary classified arrays
    :param y_true:
    :param y_pred:
    :return: f_score
    """ 
    #Generating confusion matrix for given true and pred labels
    conf = confusion_matrix(y_true, y_pred)
    #Precision
    p = conf[1][1]/(conf[1][1] + conf[0][1])
    #Recall
    r = conf[1][1]/(conf[1][1] + conf[1][0])
    return (2*p*r)/(p + r)

rfc_fscore = f_score(validation_labels, val_pred_labels)

#Generating submission csv by adding pred_labels columns to test with required column_name
def generate_pred_csv(testset, pred_labels_df, column_name, csv_name):
    """
    Converting pred_labels to dataframe
    :param testset:
    :param pred_labels_df:
    :param column_name:
    :param csv_name:
    :return: submission_csv
    """ 
    testset[column_name] = pred_labels_df
    return testset.to_csv(csv_name, index = False)
  
generate_pred_csv(test, test_pred_df, 'Prediction', 'rfc_submit.csv')