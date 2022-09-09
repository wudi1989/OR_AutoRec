import pandas as pd
from scipy.sparse import csr_matrix


def load_data(path1='../../Data/ml100k/temp/ml100k_train.csv',
                           path2='../../Data/ml100k/temp/ml100k_test.csv',
                           header=['user_id', 'item_id', 'rating','category']):
    train_data = pd.read_csv(path1, usecols=header)
    test_data = pd.read_csv(path2, usecols=header)

    n_users = max(train_data.user_id) if max(train_data.user_id) > max(test_data.user_id) else max(test_data.user_id)
    n_items = max(train_data.item_id) if max(train_data.item_id) > max(test_data.item_id) else max(test_data.item_id)


    train_row = []
    train_col = []
    train_rating = []
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1.0
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    print("Load Data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), n_users, n_items




