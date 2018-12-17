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
