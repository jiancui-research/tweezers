import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from nltk.corpus import stopwords

import os 
import sys 
sys.path.append(os.path.abspath('..'))
from utils.clustering_utils import *

# Initialize a dictionary to store results
results = {}

# Define a function to store results
def store_results(period, embedding_name, max_ami, max_ari, max_nmi):
    key = f"{period}_{embedding_name}"
    results[key] = (max_ami, max_ari, max_nmi)

# Define a function to print all results
def print_all_results(results):
    print("\nFinal Results:")
    for key, (max_ami, max_ari, max_nmi) in results.items():
        period, model = key.split('_', 1)
        print(f'{period} {model} Results:')
        print(f'{max_ami:.4f} & {max_ari:.4f} & {max_nmi:.4f} ')

# BERT, BERTweet, SecureBERT
device = 'cpu'
emb_folder = 'tweet_features/'

def save_or_load_embeddings(tweet_df, device, emb_folder, plm_type, date):
    os.makedirs(emb_folder, exist_ok=True)
    emb_path = os.path.join(emb_folder, f"{date}_{plm_type}_embs.pt")

    if os.path.exists(emb_path):
        embeddings = torch.load(emb_path)
    else:
        embeddings = get_tweet_lm_embs(tweet_df, device, plm_type)
        torch.save(embeddings, emb_path)

    return embeddings

def evaluate_dataset(fname, period):
    print(f"\nProcessing {period} dataset...")
    
    # Load Data
    tweet_df_test = pd.read_json(fname, orient='records', lines=True)
    print(f"Dataset shape: {tweet_df_test.shape}")
    
    # Get test EIDs
    test_eids = tweet_df_test.eid.tolist()

    print("Processing dataset use BERTweet...")
    bertweet_embs = save_or_load_embeddings(tweet_df_test, device, emb_folder, 'bertweet', period)

    # Uncomment if you want BERTweet baseline results
    # max_eps, (max_ami, max_ari, max_nmi), _ = find_best_eps(6, bertweet_embs, test_eids)
    # store_results(period, 'BERTweet', max_ami, max_ari, max_nmi)

    print("Processing Our Embeddings...")
    # Load pre-computed initial features 加载bertweet生成的特征
    init_feats_path = f'tweet_features/{period}_bertweet_embs.pt'
    if not os.path.exists(init_feats_path):
        print(f"Error: Initial features not found at {init_feats_path}")
        return

    our_embs = get_embeddings(
        tweet_df=tweet_df_test,
        gnn_type='gat',
        init_feat_type='bertweet',
        init_feat_fname=init_feats_path,
        connect_type='promptNER',
        time_=True,
        rs=123
    )

    max_eps, (max_ami, max_ari, max_nmi), _ = find_best_eps(6, our_embs, test_eids)
    store_results(period, 'TweetEmbedder', max_ami, max_ari, max_nmi)

# Evaluate January 2024
jan_fname = 'data/202401_tweets_eval.json'
evaluate_dataset(jan_fname, '202401')

# Evaluate February 2024
feb_fname = 'data/202402_tweets_eval.json'
evaluate_dataset(feb_fname, '202402')

# Print all results at the end
print_all_results(results)