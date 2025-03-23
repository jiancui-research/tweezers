import os
import random
import numpy as np
from datetime import datetime,timezone
from tqdm import tqdm
from collections import defaultdict, Counter
from itertools import permutations, combinations
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy
from collections import defaultdict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from utils.preprocess_text import preprocess_text
from utils.early_stopping import EarlyStopping
from model.tweetembedder import TweetEmbedder

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
import swifter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import logging
logging.set_verbosity_error()

from pdb import set_trace

from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

# 将发文的时间以2021-12-30为起点，计算时间差 并换算成相差的年数和天数。
def extract_time_feature(t_str):
    # t = datetime.fromisoformat(str(t_str))
    str_value = str(t_str).strip()
    if str_value.isdigit():
        # 将时间戳转为秒（处理毫秒/微秒）
        timestamp = int(str_value) / 1000  # 假设输入是毫秒级
        # 明确时区为 UTC
        t = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        t = datetime.fromisoformat(str_value)
        # 无时区时默认添加 UTC 时区
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    # 以2021-12-30为起点
    OLE_TIME_ZERO = datetime(2021, 12, 30, tzinfo=timezone.utc)
    delta = t - OLE_TIME_ZERO
    # 86,400 seconds in day
    return [(float(delta.days)/366.), (float(delta.seconds)/86400)]

# 获取tweetebdder的嵌入
def get_embeddings(
    tweet_df,
    gnn_type = 'gcn', 
    init_feat_fname=None,
    init_feat_type=None,
    connect_type = 'promptNER',
    time_=False, 
    rs=123, 
    hop=1,
    ):
    # GPU  
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    g = make_dgl_graph(tweet_df, connect_type=connect_type)

    # Load initial features from file
    # 加载使用 BERTweet 模型生成的推文特征（每个推文对应一个特征向量）
    feats_all = torch.load(init_feat_fname)

    # extract time feature
    if time_:
        tweet_df['time_feats'] = tweet_df['created_at'].apply(lambda x: extract_time_feature(x))
        time_feats = np.array(tweet_df['time_feats'].tolist())
        time_feats = time_feats.astype(np.float32)
        time_feats = torch.tensor(time_feats)
        #讲时间特征拼接到原始特征中
        feats_all = torch.cat([feats_all, time_feats], dim=1)
    
    if hop==1:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(hop)
    elif hop==2:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(hop)
    # 将节点特征 feats_all 存储到图对象 g 的节点数据（ndata）中，键名为 'f'。
    g.ndata['f'] = feats_all

    test_indices = [i for i in range(tweet_df.shape[0])]

    # Hyper-parameters batch_size
    bs = 256 

    test_dataloader = dgl.dataloading.DataLoader(
        g.to('cpu'), test_indices, sampler,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    # in_dim=feats_all.shape[1]  输入的数据维度 
    model = TweetEmbedder(
        in_dim=feats_all.shape[1], h1_dim=256, h2_dim=64, 
        com_proj_dim=64, proj_t1_dim=64, proj_t2_dim=64, 
        num_heads=4, conv_type=gnn_type, 
        )
    # Model save name 
    model_name = f'tweetembedder_{init_feat_type}'
    if time_:
        model_name+='_time'
    if hop > 1:
        model_name+=f'_{hop}hop'
    model_name+=f'_{connect_type}_{gnn_type}_{rs}.pt'
    model_fname = f'./trained_models/{model_name}'

    print(f"load model from {model_fname}")
    model.load_state_dict(torch.load(model_fname, map_location=device), strict=False)
    model.to(device)
    
    test_embeddings = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in test_dataloader:
            blocks = [b.to(device) for b in blocks]
            input_features = blocks[0].srcdata['f'].to(device)

            b_feats = feats_all[input_nodes].to(device)
            
            b_h, _ = model(blocks, b_feats, hop=hop)

            test_embeddings.append(b_h)
    
    test_embeddings_ = torch.cat(test_embeddings).to('cpu')            
    
    return test_embeddings_

def make_pairwise_matrix(eids):
    pair_matrix = []
    for i in range(len(eids)):
        row = []
        for j in range(len(eids)):
            row.append(eids[i] == eids[j])
        pair_matrix.append(row)

    return torch.Tensor(pair_matrix)

def extract_pairwise(indices, eids, sample_num=1000):
    # print(len(indices), len(eids))
    eid2indices = defaultdict(list)

    for idx, eid in zip(indices, eids):
        if eid == -1:
            continue
        eid2indices[eid].append(idx)

    pn_pairs = []
    for eid, indices in eid2indices.items():
        if len(indices) == 1:
            continue

        p_pairs = list(combinations(indices, 2))
        rest_eids = list(set(eid2indices.keys()) - set([eid]))

        n_pairs = []
        for e1, e2 in list(combinations(rest_eids, 2)):
            e1_indices = eid2indices[e1]
            e2_indices = eid2indices[e2]

            for e1_i in e1_indices:
                for e2_i in e2_indices:
                    n_pairs.append((e1_i, e2_i))

        for p_pair in p_pairs:
            for n_pair in n_pairs:
                pn_pairs.append((p_pair, n_pair))

    pn_pairs_ = random.sample(pn_pairs, min(sample_num, len(pn_pairs)))
    p1_list, p2_list, n1_list, n2_list = [], [], [], []
    for (p1, p2), (n1, n2) in pn_pairs_:
        p1_list.append(p1)
        p2_list.append(p2)
        n1_list.append(n1)
        n2_list.append(n2)

    return (p1_list, p2_list, n1_list, n2_list), pn_pairs
# 寻找邻域范围的最佳值
def find_best_eps(eps, test_embeddings, test_eids,_min_samples=3):
    ami_list, ari_list, nmi_list = [], [], []
    for eps_ in range(eps-5, eps+5):
        db = DBSCAN(eps=eps_, min_samples=_min_samples).fit(test_embeddings)
        ami_score, ari_score, nmi_score = get_clustering_scores(np.array(test_eids), db.labels_, digit=5)
        ami_list.append((eps_, ami_score))
        ari_list.append((eps_, ari_score))
        nmi_list.append((eps_, nmi_score))
    
    max_tuple = max(nmi_list, key=lambda x: x[1])
    max_eps, _ = max_tuple
    max_idx = nmi_list.index(max_tuple)
    return max_eps, (ami_list[max_idx][1], ari_list[max_idx][1], nmi_list[max_idx][1]), (ami_list, ari_list, nmi_list)

#三元组损失函数 用于训练模型用
def extract_apn(indices, eids):

    eid2indices = defaultdict(list)

    for idx, eid in zip(indices, eids):
        if eid == -1:
            continue
        eid2indices[eid].append(idx)

    apn_list = []
    for eid, indices in eid2indices.items():
        if len(indices) == 0:
            continue
        neg_event_set = list(
            set([idx for l in eid2indices.values() for idx in l]) - set(eid2indices[eid]))

        ap_list = list(permutations(indices, 2))

        for a, p in ap_list:
            for n in neg_event_set:
                apn_list.append((a, p, n))

    a_list, p_list, n_list = [], [], []
    for a, p, n in apn_list:
        a_list.append(a)
        p_list.append(p)
        n_list.append(n)

    return a_list, p_list, n_list

def get_clustering_scores(true_labels, predict_labels, digit=5, print_excel=False):
    ami_score = adjusted_mutual_info_score(true_labels, predict_labels)
    ari_score = adjusted_rand_score(true_labels, predict_labels)
    nmi_score = normalized_mutual_info_score(true_labels, predict_labels)

    ami_score= round(ami_score, digit)
    ari_score= round(ari_score, digit)
    nmi_score= round(nmi_score, digit)

    if print_excel:
        print(f'{ami_score}\t{ari_score}\t{nmi_score}')
    return ami_score, ari_score, nmi_score

'''
cluster_result: DBSCAN result
tweet_df: dataframe 
indices: original dataframe index
 '''
def print_cluster_text(
        cluster_result, tweet_df, 
        cluster_id, indices
):
    labels = cluster_result.labels_
    cluster_counter = Counter(labels)
    sample_indices = list(np.where(cluster_result.labels_==cluster_id)[0])

    print(f"# of samples in cluster {cluster_id}: {len(sample_indices)}")
    for i, idx in enumerate(sample_indices): 
        print(tweet_df.iloc[indices[idx]]['created_at'], tweet_df.iloc[indices[idx]]['id'])
        print(tweet_df.iloc[indices[idx]]['text'])  

'''
Return:  
dataframe containing tweet_ids, user_ids, event_types, etc,.  
'''
def get_cluster_info(
   cluster_result, tweet_df, indices, 
   window_start, window_end,
   fitler_noise = True,
   most_common_n = 20, 
):
    # cluster2tid, cluster2uid, cluster2ets = defaultdict(list), defaultdict(list), defaultdict(list)
    labels = cluster_result.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(f"# cluster: {n_clusters_}; # noise: {n_noise_}")

    cluster_counter = Counter(labels)
    if fitler_noise: 
        del cluster_counter[-1] # remove noise cluster 

    cluster2score = defaultdict(float)

    
    df = pd.DataFrame(columns=['cluster_id', 'tweet_ids', 'user_ids', 'event_types'])
    for rank, (cluster_id, num_samples) in enumerate(cluster_counter.most_common(most_common_n)):
        sample_indices = list(np.where(cluster_result.labels_==cluster_id)[0])

        tid_list, uid_list, et_list = [], [], []
        for i, idx in enumerate(sample_indices): 
            tid_list.append(tweet_df.iloc[idx]['id'])
            uid_list.append(tweet_df.iloc[idx]['user_id'])
            et_list.append(tweet_df.iloc[idx]['event_type'])

        score = len(set(uid_list)) / len(tid_list)
        df = df.append({
            'cluster_id': cluster_id, 
            'tweet_ids': tid_list, 
            'user_ids':uid_list, 
            'event_types':et_list,
            'score': score,
            'window_start':window_start,
            'window_end':window_end,
        }, ignore_index=True)
    
    return df
    # return cluster2tid, cluster2uid, cluster2ets

def print_cluster(
        cluster_result, tweet_df, indices, 
        print_noise=False, print_num=20, score_thred=0.7,
        most_common_n = 20,
):
    labels = cluster_result.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(f"# cluster: {n_clusters_}; # noise: {n_noise_}")

    cluster_counter = Counter(labels)
    if not print_noise: 
        del cluster_counter[-1] # remove noise cluster 

    cluster2score = defaultdict(float)

    # calculate score for each cluster
    for rank, (cluster_id, num_samples) in enumerate(cluster_counter.most_common(most_common_n)):
        sample_indices = list(np.where(cluster_result.labels_==cluster_id)[0])
        
        cluster_user_set = set()
        for i, idx in enumerate(sample_indices): 
            cluster_user_set.add(tweet_df.iloc[idx]['user_id'])    

        # cluster2score[cluster_id] = len(sample_indices)
        cluster2score[cluster_id] = (len(cluster_user_set) / len(sample_indices)), len(sample_indices)
    
    # filter cluters with score less than threshold
    cluster2score_f = dict(filter(lambda elem: elem[1][0]>score_thred, cluster2score.items()))

    # print sample in cluster
    for rank, (cluster_id, (score, num_samples)) in enumerate(sorted(cluster2score_f.items(), key=lambda item: -item[1][0])):

        sample_indices = list(np.where(cluster_result.labels_==cluster_id)[0])
        print('='*120)
        print(f"ranking: {rank} | cluster ID: {cluster_id}; score: {score}; # samples: {len(sample_indices)} ")
        print('-'*40)

        for i, idx in enumerate(sample_indices): 
            print(tweet_df.iloc[indices[idx]]['created_at'], tweet_df.iloc[idx]['user_screen_name'])
            print(tweet_df.iloc[indices[idx]]['text'])

            if i == (print_num-1): break

# 根据connect_type对tweet_df中的文本进行连接，构建图。只是将图中不同的点连起来了 并没有边和节点的数据
def make_dgl_graph(tweet_df, connect_type):
    if connect_type=='noun':
        connect_elems = tweet_df.nouns.tolist()
    elif connect_type=='keybert':
        connect_elems = tweet_df.keybert.tolist()
    elif connect_type=='entites': 
        connect_elems = tweet_df.entities.tolist()
    elif connect_type=='promptNER':
        #使用promptNER提取出来的实体进行连接
        connect_elems = tweet_df['promptNER entities'].tolist() 
    else:
        logging.info(f"no such connect type: {connect_type}")
    num_tweets = tweet_df.shape[0]

    # Connect with entitiies
    src_nids_, dst_nids_, e_weights_ = [], [], []
    for i in range(num_tweets):
        for j in range(i+1, num_tweets):
            num_common = len(set(connect_elems[i]).intersection(set(connect_elems[j])))
            # if num_common > 0:
            for _ in range(num_common):
                src_nids_.append(i)
                dst_nids_.append(j)

    src_nids = src_nids_ + dst_nids_
    dst_nids = dst_nids_ + src_nids_

    g = dgl.graph((torch.LongTensor(src_nids), torch.LongTensor(dst_nids)),  num_nodes=num_tweets)
    
    return g