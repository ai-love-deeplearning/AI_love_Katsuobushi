
from operator import itemgetter

import random
import pandas as pd
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

def convert_sparse_matrix_to_sparse_tensor(sparse_mx):
    # if not sp.isspmatrix_coo(sparse_mx):
    #     sparse_mx = sparse_mx.tocoo()
    # coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    return tf.SparseTensor(indices, sparse_mx.data, sparse_mx.shape)

# 特徴行列の正規化
def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # inf（無限大）にゼロを設定して、ゼロによる除算を回避します
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('エラー：正規化された隣接行列のエントリはゼロです')
        exit

    return feat_norm

def preprocess_user_item_features(u_features, v_features):

    """
    ユーザー特徴量とアニメ作品特徴量から1つの大きな特徴行列を作成します。
    ユーザー特徴量の下にアニメ作品特徴量をスタックします。
    """

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')



    return u_features, v_features

def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):

    """ 二部隣接行列のセットをグローバルに正規化します
        Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj（対称的に正規化する二部調整）')
    # degree_u and degree_v are row and column sums of adj+I
    # degree_u（ユーザーがアニメ作品を見た数）
    # degree_v（アニメ作品がユーザーに見られた数）

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # infにゼロを設定して、ゼロによる除算を回避します
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric: # symmetric normalization
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else: # left normalization
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm

def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def map_data(data):
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    n = len(uniq)

    return data, id_dict, n

def load_data(seed=1234, verbose=True):

    u_features = None
    v_features = None

    filename = './csv/ratings_data.csv'

    dtypes = {'u_nodes': np.int64, 'v_nodes': np.int64, 'ratings': np.float32}

    headers = ['u_nodes', 'v_nodes', 'ratings']
    data = pd.read_csv(filename, header=None, names=headers, converters=dtypes, engine='python')

    data_array = data.as_matrix().tolist()
    random.seed(seed)
    random.shuffle(data_array)
    data_array = np.array(data_array)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
    ratings = ratings.astype(np.float32)

    # Load anime features
    animes_file = './csv/animes_data.csv'

    animes_headers = ['anime_id', 'title', 'company', 'genre']
    animes_df = pd.read_csv(animes_file, header=None, names=animes_headers, engine='python')

    # すべての制作会社を抽出する
    companies = []
    for s in animes_df['company'].values:
        companies.append(s.split(' / ')[0])

    companies = list(set(companies))
    num_companies = len(companies)

    companies_dict = {g: idx for idx, g in enumerate(companies)}

    # Creating 0 or 1 valued features for all companies
    v_features = np.zeros((num_items, num_companies), dtype=np.float32)
    for anime_id, s in zip(animes_df['anime_id'].values.tolist(), animes_df['company'].values.tolist()):
        # Check if anime_id was listed in ratings file and therefore in mapping dictionary
        if anime_id in v_dict.keys():
            com = s.split(' / ')[0]
            v_features[v_dict[anime_id], companies_dict[com]] = 1.

    # Load user features
    u_features = np.zeros((num_users, 1), dtype=np.float32)

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features

def create_trainvaltest_split():

    num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_data(seed=1234, verbose=True)

    neutral_rating = -1
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])

    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    num_train = ratings.shape[0] - num_val - num_test

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    train_idx = idx_nonzero[0:num_train]
    val_idx = idx_nonzero[num_train:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    train_pairs_idx = pairs_nonzero[0:num_train]
    val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    # train隣接行列を作成する
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    # val隣接行列を作成する
    rating_mx_val = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_val[val_idx] = labels[val_idx].astype(np.float32) + 1.
    rating_mx_val = sp.csr_matrix(rating_mx_val.reshape(num_users, num_items))

    # test隣接行列を作成する
    rating_mx_test = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_test[test_idx] = labels[test_idx].astype(np.float32) + 1.
    rating_mx_test = sp.csr_matrix(rating_mx_test.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))

    return u_features, v_features, rating_mx_train, rating_mx_val, rating_mx_test,\
    train_labels, u_train_idx, v_train_idx, val_labels, u_val_idx, v_val_idx,\
    test_labels, u_test_idx, v_test_idx, class_values

# if __name__ == '__main__':
#     u_features, v_features, rating_mx_train, train_labels, u_train_idx,
#     v_train_idx, val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx,
#     v_test_idx, class_values = create_trainvaltest_split()
