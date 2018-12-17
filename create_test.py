#Creating test dataset 
def create_test_dataset():
    
    test_set = []
    #
    date_format = '%Y/%m/%d'
    #
    for row in range(len(test_data)):
        #
        customer = test_data[row:row+1]['Customer_ID'].values[0]
        nationality = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Nationality'].unique()[0]
        income_range = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Income_Range'].unique()[0]
        job_type = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Job_Type'].unique()[0]
        marital_status =  customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Marital_Status'].unique()[0]
        gender = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Gender'].unique()[0]
        state = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['State'].unique()[0]
        lang = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Language'].unique()[0]
        loyalty_status = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Loyalty_Status'].unique()[0]
        age = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Age'].unique()[0]
        points = customer_info.loc[customer_info['Customer_ID'] == test_data[row:row+1]['Customer_ID'].values[0]]['Points'].unique()[0]
        total_cust_rev = transactions_agg['total_revenue'][transactions_agg['customer_id'] == test_data[row:row+1]['Customer_ID'].values[0]].sum()
        total_items_sold = transactions_agg['total_units_sold'][transactions_agg['customer_id'] == test_data[row:row+1]['Customer_ID'].values[0]].sum()
        #
        region = test_store_master.loc[test_store_master['store_code'] == test_data[row:row+1]['Store_Code'].values[0]]['region'].unique()[0][5:]
        sales_per_day = test_store_master.loc[test_store_master['store_code'] == test_data[row:row+1]['Store_Code'].values[0]]['sales_per_day'].unique()[0]
        test_store_code = test_data[row:row+1]['Store_Code'].values[0]
        store_size = test_store_master.loc[test_store_master['store_code'] == test_data[row:row+1]['Store_Code'].values[0]]['sales_per_day'].unique()[0]
        cust_count = test_store_master.loc[test_store_master['store_code'] == test_data[row:row+1]['Store_Code'].values[0]]['customer_count'].unique()[0]
        store_total_rev = test_store_master.loc[test_store_master['store_code'] == test_data[row:row+1]['Store_Code'].values[0]]['total_rev'].unique()[0]
        #
        state_eq_region = 1 if state == region else 0
        #    
        test_set.append((customer,test_store_code,income_range,job_type,marital_status,gender,state,lang,loyalty_status,age,points,total_cust_rev,total_items_sold,state_eq_region,sales_per_day,store_size,cust_count,store_total_rev))    
    test_set = pd.DataFrame(test_set)
    return test_set

test = create_test_dataset()

#Fetching data from drive and converting into required dataframe
test_data_drive = drive.CreateFile({'id': '1oWi93HW3YkUOC1GvOlJA4HGet8ckqJ2m'})
test_data_drive.GetContentFile('test_data.csv')
test = pd.read_csv('test_data.csv', index_col = None)

#Adding headers to test_set
add_headers(test, ['index','customer_id','store_code','income_range','job_type','marital_status','gender','state','lang','loyalty_status','age','points','total_cust_rev','total_items_sold','state_eq_region','sales_per_day','store_size_sq_ft','cust_count','store_total_rev'])

#Preprocessing
 
#Encoding categorical features
#Encoding loyalty_status
'loyalty_status', {'Gold': 0, 'Silver': 1})
test['loyalty_status_encoded'] = dataset_encoding(test,'loyalty_status', {'Gold': 0, 'Silver': 1})

#Encoding gender 
test['gender_encoded'] = dataset_encoding(test,'gender', {'M': 1, 'F': 0.87, 'Unspecified':0})

#Encoding income_range 
test['income_range_encoded'] = dataset_encoding(test,'income_range', {'Below 5000': 1, '5001 to 10000': 2, '10001 to 20000':3, '20001 to 30000':4, '30001 & Above':5,'Unspecified':0, 'Unknown':0})

#Encoding income_range 
test['marital_status_encoded'] = dataset_encoding(test,'marital_status', {'Married': 1, 'Single': 0.5, 'Divorsed': 0.5, 'Widowed': 0.5, 'Unspecified': 0,'Separated': 0.5, 'Others':0})

#Imputation
#Imputing NA's in points by median 
dataset_fillna(test, 'points', 215.0)

#Extracting features
test_features_dataframe = extract_features(test, ['index','customer_id','store_code','income_range','job_type','marital_status','gender','state','lang','loyalty_status','age'],axis = 1)

#Normalizing test features dataframe
test_features_scaled = normalize_dataset(test_features_dataframe)

#Predictions for random forest model
test_pred_labels = model_prediction(rfc_model, test_features_scaled)

#Generating test_labels dataframe
test_pred_df['Prediction'] = pred_dataframe(test_pred_labels, 'Prediction')

