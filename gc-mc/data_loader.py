#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

from preprocessing import *

NUMCLASSES = 41
SYM = True

def get_loader():
	u_features, v_features, class_values,\
	train_adj, train_labels, train_u_indices, train_v_indices,\
	val_labels,   val_u_indices,   val_v_indices,\
	test_labels,  test_u_indices,  test_v_indices = create_trainvaltest_split()

	num_users, num_items = train_adj.shape
	num_side_features = 0

	print("Normalizing feature vectors...")
	u_features_side = normalize_features(u_features)
	v_features_side = normalize_features(v_features)

	u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

	# (2863, 273), (3472, 273)
	u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
	v_features_side = np.array(v_features_side.todense(), dtype=np.float32)
	# 補助情報の数
	num_side_features = u_features_side.shape[1]

	# ノードの入力特徴量のID
	# node id's for node input features
	id_csr_u = sp.identity(num_users, format='csr')
	id_csr_v = sp.identity(num_items, format='csr')

	# (2863, 6335) (3472, 6335)
	u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

	# global normalization
	normalized = []
	normalized_t = []
	train_adj_int = sp.csr_matrix(train_adj, dtype=np.int32)

	for i in range(NUMCLASSES):
		unnormalized = sp.csr_matrix(train_adj_int == i + 1, dtype=np.float32)

		unnormalized_t = unnormalized.T
		normalized.append(unnormalized)
		normalized_t.append(unnormalized_t)

	normalized = globally_normalize_bipartite_adjacency(normalized, symmetric=SYM)
	normalized_t = globally_normalize_bipartite_adjacency(normalized_t, symmetric=SYM)

	num_normalized = len(normalized)
	normalized = sp.hstack(normalized, format='csr')
	normalized_t = sp.hstack(normalized_t, format='csr')

	# Collect all user and item nodes for test set
	test_u = list(set(test_u_indices))
	test_v = list(set(test_v_indices))
	test_u_dict = {n: i for i, n in enumerate(test_u)}
	test_v_dict = {n: i for i, n in enumerate(test_v)}

	test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
	test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])

	test_normalized = normalized[np.array(test_u)]
	test_normalized_t = normalized_t[np.array(test_v)]

	# Collect all user and item nodes for validation set
	val_u = list(set(val_u_indices))
	val_v = list(set(val_v_indices))
	val_u_dict = {n: i for i, n in enumerate(val_u)}
	val_v_dict = {n: i for i, n in enumerate(val_v)}

	val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
	val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

	val_normalized = normalized[np.array(val_u)]
	val_normalized_t = normalized_t[np.array(val_v)]

	# Collect all user and item nodes for train set
	train_u = list(set(train_u_indices))
	train_v = list(set(train_v_indices))
	train_u_dict = {n: i for i, n in enumerate(train_u)}
	train_v_dict = {n: i for i, n in enumerate(train_v)}

	train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
	train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

	train_normalized = normalized[np.array(train_u)]
	train_normalized_t = normalized_t[np.array(train_v)]

	# feature_sideを分割
	test_u_features_side = u_features_side[np.array(test_u)]
	test_v_features_side = v_features_side[np.array(test_v)]

	val_u_features_side = u_features_side[np.array(val_u)]
	val_v_features_side = v_features_side[np.array(val_v)]

	train_u_features_side = u_features_side[np.array(train_u)]
	train_v_features_side = v_features_side[np.array(train_v)]

	return num_users, num_items, class_values, num_side_features, num_normalized, u_features, v_features,\
	train_u_features_side, train_v_features_side, train_normalized, train_normalized_t,\
	train_u_indices, train_v_indices, train_labels,\
	val_u_features_side, val_v_features_side, val_normalized, val_normalized_t,\
	val_u_indices, val_v_indices, val_labels,\
	test_u_features_side, test_v_features_side, test_normalized, test_normalized_t, \
	test_u_indices, test_v_indices, test_labels
