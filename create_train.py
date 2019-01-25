def create_train_dataset():
    
    train_set = []
    
    #
    for customer in customer_info['Customer_ID'].unique():
        #
        customer_records = customer_info.loc[customer_info['Customer_ID'] == customer]
        income_range = customer_records['Income_Range'].unique()[0]
        job_type = customer_records['Job_Type'].unique()[0]
        marital_status =  customer_records['Marital_Status'].unique()[0]
        gender = customer_records['Gender'].unique()[0]
        state = customer_records['State'].unique()[0]
        loyalty_status = customer_records['Loyalty_Status'].unique()[0]
        points = customer_records['Points'].unique()[0]
        total_cust_rev = transactions_agg['reg_rev'][transactions_agg['customer_id'] == customer].sum()
        total_items_sold = transactions_agg['reg_units_sold'][transactions_agg['customer_id'] == customer].sum()
        required_records = customer_trans.loc[customer_trans['Customer_ID'] == customer]
        #
        for row in range(len(store_master)):
            #
            region = store_master[row:row+1]['region'].values[0]
            sales_per_day = store_master[row:row+1]['sales_per_day'].values[0]
            store_master_store_code = store_master[row:row+1]['store_code'].values[0]
            store_size = store_master[row:row+1]['store_size_sq_ft'].values[0]
            cust_count = store_master[row:row+1]['customer_count'].values[0]
            store_total_rev = store_master[row:row+1]['total_rev'].values[0]
            region_rev_list = transactions_agg.loc[(transactions_agg['region'] == region) & (transactions_agg['customer_id'] == customer),'reg_rev'].unique()
            region_cust_rev = region_rev_list[0] if len(region_rev_list) != 0 else 0
            region_units_sold_list = transactions_agg.loc[(transactions_agg['region'] == region) & (transactions_agg['customer_id'] == customer),'reg_units_sold'].unique()
            region_units_sold = region_units_sold_list[0] if len(region_rev_list) != 0 else 0
            region_trans_list = transactions_agg.loc[(transactions_agg['region'] == region) & (transactions_agg['customer_id'] == customer),'reg_trans'].unique()
            region_trans = region_trans_list[0] if len(region_rev_list) != 0 else 0
            #
            prediction = 0
            state_eq_region = 1 if state == region[5:] else 0
            if store_master_store_code in list(required_records['Store_Code']):
                prediction = 1 
                train_set.append((customer,store_master_store_code,prediction,income_range,job_type,marital_status,gender,state,loyalty_status,points,total_cust_rev,total_items_sold,state_eq_region,sales_per_day,store_size,cust_count,store_total_rev))
            train_set.append((customer,store_master_store_code,prediction,income_range,job_type,marital_status,gender,state,loyalty_status,points,total_cust_rev,total_items_sold,state_eq_region,sales_per_day,store_size,cust_count,store_total_rev))
    train_set = pd.DataFrame(train_set)
    return train_set

train = create_train_dataset() 



def add_features(train_set):
    train_set['region_rev'] = 0
    train_set['region_units_sold'] = 0
    train_set['region_trans'] = 0
    #
    for row in range(len(transactions_agg)):
        customer = transactions_agg[row:row+1]['customer_id'].values[0]
        region = transactions_agg[row:row+1]['region'].values[0]
        region_rev = transactions_agg[row:row+1]['reg_rev'].values[0]
        region_units_sold = transactions_agg[row:row+1]['reg_units_sold'].values[0]
        region_trans = transactions_agg[row:row+1]['reg_trans'].values[0]
        
        for store_code in store_master['store_code'].unique():
            store_region = store_master.loc[store_master['store_code'] == store_code, 'region'].unique()[5:]
            train_set.loc[(train_set['store_code'] == store_code) & (train_set['customer_id'] == customer), 'region_rev'] = region_rev if region == store_region else 0
            train_set.loc[(train_set['store_code'] == store_code) & (train_set['customer_id'] == customer), 'region_units_sold'] = region_units_sold if region == store_region else 0
            train_set.loc[(train_set['store_code'] == store_code) & (train_set['customer_id'] == customer), 'region_trans'] = region_units_sold if region == store_region else 0
    return train_set

expanded_train = add_features(train)
    
