#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

from utils_env import *

NUMCLASSES = 41

def get_loader():
	"""Dataloaderをビルドして返す"""

	# u_features: ユーザーの特徴行列
	# v_features: アニメ作品の特徴行列
	# adj_train: 隣接行列（訓練）
	# train_labels: 評価点のラベル（訓練）
	# train_u_indices: ユーザーID（訓練）
	# train_v_indices: アニメ作品ID（訓練）
	# val_labels: 評価点のラベル（検証）
	# val_u_indices: ユーザーID（検証）
	# val_v_indices: アニメ作品ID（検証）
	# test_labels: 評価点のラベル（テスト）
	# test_u_indices: ユーザーID（テスト）
	# test_v_indices: アニメ作品ID（テスト）
	# class_values: 評価点
	u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
		val_labels, val_u_indices, val_v_indices, test_labels, \
		test_u_indices, test_v_indices, class_values = create_trainvaltest_split()

	num_users, num_items = adj_train.shape

	print("Normalizing feature vectors...")
	u_features_side = normalize_features(u_features)
	v_features_side = normalize_features(v_features)

	u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

	# 3472×544, 3472×544
	u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
	v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

	num_side_features = u_features_side.shape[1]

	# ノードの入力特徴量のID
	# node id's for node input features
	id_csr_u = sp.identity(num_users, format='csr')
	id_csr_v = sp.identity(num_items, format='csr')

	# (2863, 6335) (3472, 6335)
	u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

	u_features = u_features.toarray()
	v_features = v_features.toarray()

	num_features = u_features.shape[1]

	support = []
	support_t = []
	adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)

	for i in range(NUMCLASSES):
		support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)

		support_unnormalized_transpose = support_unnormalized.T


	return num_users, num_items, len(class_values), num_side_features, num_features, \
		   u_features, v_features, u_features_side, v_features_side, \

if __name__ == '__main__':
	num_users, num_items, num_classes, num_side_features, num_features,\
	u_features, v_features, u_features_side, v_features_side = get_loader()

	# print(num_users) # ユーザー数 2863
	# print(num_items) # 作品数 3472
	# print(num_classes) # 評価点の数 41
	# print(num_side_features) # 特徴量の数 273
	# print(num_features) # （ユーザー数＋作品数）6335
	# print(u_features)
	# print(v_features)
	# print(u_features_side)
	# print(v_features_side)
