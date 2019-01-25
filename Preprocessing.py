Adding columns to the dataframe
def add_headers(dataset, headers):
    """
    Add the headers to the dataset
    :param dataset:
    :param headers:
    :return: dataset 
    """
    dataset.columns = headers
    return dataset
    
    
add_headers(train, ['index','customer_id','store_code','prediction','income_range','job_type','marital_status','gender','state','lang','loyalty_status','age','points','total_cust_rev','total_items_sold','state_eq_region','sales_per_day','store_size_sq_ft','cust_count','store_total_rev','loyalty_status_encoded','gender_encoded','income_range_encoded','marital_status_encoded','rev_ratio'])

#Encoding funtion
def dataset_encoding(dataset, column_name, encoding_dict):
    """
    Encode dataset column as per encoding_dict
    :param dataset:
    :param column_name:
    :param encoding_dict:
    :return: encoded_dataset_column 
    """ 
    return dataset[column_name].map(encoding_dict)

#Preprocessing
 
#Encoding categorical features
#Encoding loyalty_status
train['loyalty_status_encoded'] = dataset_encoding(train, 'loyalty_status', {'Gold': 0, 'Silver': 1})

#Encoding gender 
train['gender_encoded'] = dataset_encoding(train, 'gender', {'M': 1, 'F': 0.87, 'Unspecified':0})

#Encoding income_range 
train['income_range_encoded'] = dataset_encoding(train, 'income_range', {'Below 5000': 1, '5001 to 10000': 2, '10001 to 20000':3, '20001 to 30000':4, '30001 & Above':5,'Unspecified':0, 'Unknown':0})

#Encoding income_range 
train['marital_status_encoded'] = dataset_encoding(train, 'marital_status', {'Married': 1, 'Single': 0, 'Divorsed': 0, 'Widowed': 0, 'Unspecified': 0,'Separated': 0, 'Others':0})

#Ratio of total_cust_rev to store_total_rev
train['rev_ratio'] = train['rev_ratio']*10000

#Imputation
def dataset_fillna(dataset, column_name, val):
    """
    Encode dataset column as per encoding_dict
    :param dataset:
    :param column_name:
    :param val:
    :return: None 
    """ 
    dataset[column_name].fillna(value = val,  inplace = True)

dataset_fillna(train, 'points', 150.0)

#Sampling dataset
def sample_dataset(train_df, fraction):
    """
    Sample the dataset
    :param train_df:
    :param fraction:
    :return: sampled_train_df
    """ 
    #Using random sampling to reduce size of train
    sampled_train_df = train_df.sample(frac = fraction, random_state = 42)
    return sampled_train_df

train = sample_dataset(train, .7)

#Extract features from dataset
def extract_features(train_df, column_list):
    """
    Sample the dataset
    :param train_df:
    :param column_list:
    :return: features_df
    """ 
    #Using random sampling to reduce size of train
    featues_df = train_df.drop(column_list, axis = 1)
    return featues_df

features_dataframe = extract_features(train, ['index','customer_id','store_code','prediction','income_range','job_type','marital_status','gender','state','lang','loyalty_status','age'])

#Converting target values to np array
labels = np.array(train['prediction'])

#Splitting train and test
def normalize_dataset(features_df):
    """
    Normalize the dataset
    :param features_dataframe:
    :return: normalized_features_array
    """ 
    #Using minmaxscaler from sklearn to normalize dataset
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(features_df)

features_scaled_array = normalize_dataset(features_dataframe)

features_scaled_array

#Sampling and balancing dataset using ADAYSN
def balancing_dataset(features_array, labels_array):
    """
    Balancing classes and extracting sample from balanced dataset
    :param features_array:
    :param labels_array:
    :return: balanced_features, balanced_labels
    """ 
    #Using ADASYN to balance dataset by up sampling data using n = 5
    sm = ADASYN(sampling_strategy = .7, random_state = 42, n_neighbors = 5)
    balanced_features, balanced_labels = sm.fit_resample(features_array, labels_array)
    return balanced_features, balanced_labels

balanced_features_array, balanced_labels_array = balancing_dataset(features_scaled_array, labels)

#Splitting train and test
def split_dataset(features_np_array, target_np_array, test_percentage, random_state):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param test_percentage:
    :param feature_array:
    :param target_array:
    :return: train_x, test_x, train_y, test_y
    """ 
    #Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(features_np_array, target_np_array, test_size = test_percentage, random_state = random_state)
    return train_x, test_x, train_y, test_y


train_features,validation_features, train_labels, validation_labels = split_dataset(features_scaled_array, labels, .3, 42)
