"""
Chapter 4.5: ALS vs NCF vs Baseline (global mean) on the same train/test split.
Run on master: python3 ~/projects/recommend/compare_models.py
"""
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import torch, torch.nn as nn
import pandas as pd, numpy as np
import subprocess
from io import StringIO

# ─── Spark setup ───
spark = (SparkSession.builder
    .appName('ModelCompare')
    .master('yarn')
    .config('spark.executor.memory', '1g')
    .config('spark.driver.memory', '1g')
    .config('spark.dynamicAllocation.enabled', 'false')
    .getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

ratings = spark.read.csv('/data/movielens/ratings.csv', header=True, inferSchema=True)
train, test = ratings.randomSplit([0.8, 0.2], seed=42)
train.cache(); test.cache()
print(f'Train: {train.count()},  Test: {test.count()}')

# ─── Baseline: global mean ───
global_mean = train.agg(F.mean('rating')).collect()[0][0]
baseline_pred = test.withColumn('prediction', F.lit(global_mean))
ev_rmse = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
ev_mae  = RegressionEvaluator(metricName='mae',  labelCol='rating', predictionCol='prediction')
baseline_rmse = ev_rmse.evaluate(baseline_pred)
baseline_mae  = ev_mae.evaluate(baseline_pred)
print(f'Baseline (mean={global_mean:.3f}) RMSE={baseline_rmse:.4f}  MAE={baseline_mae:.4f}')

# ─── ALS ───
als = ALS(maxIter=10, regParam=0.1, rank=50,
          userCol='userId', itemCol='movieId', ratingCol='rating',
          coldStartStrategy='drop', nonnegative=True)
als_model = als.fit(train)
als_pred = als_model.transform(test)
als_rmse = ev_rmse.evaluate(als_pred)
als_mae  = ev_mae.evaluate(als_pred)
print(f'ALS RMSE={als_rmse:.4f}  MAE={als_mae:.4f}')

# ─── NCF: read saved model and evaluate on the same test split ───
test_pdf = test.toPandas()

class NCF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
    def forward(self, u, i):
        return self.mlp(torch.cat([self.user_embed(u), self.item_embed(i)], dim=1)).squeeze()

subprocess.run(['hdfs', 'dfs', '-get', '-f',
                '/models/ncf_model/ncf_checkpoint.pt', '/tmp/ncf_checkpoint.pt'])
ckpt = torch.load('/tmp/ncf_checkpoint.pt', map_location='cpu', weights_only=False)
m = NCF(ckpt['n_users'], ckpt['n_items'], ckpt['embed_dim'])
m.load_state_dict(ckpt['model_state']); m.eval()
u2i, i2i = ckpt['user2idx'], ckpt['item2idx']

errs, abs_errs = [], []
with torch.no_grad():
    for _, row in test_pdf.iterrows():
        uid, mid, actual = int(row['userId']), int(row['movieId']), float(row['rating'])
        if uid in u2i and mid in i2i:
            pred = m(torch.tensor([u2i[uid]]), torch.tensor([i2i[mid]])).item()
            errs.append((actual - pred) ** 2)
            abs_errs.append(abs(actual - pred))
ncf_rmse = float(np.sqrt(np.mean(errs)))
ncf_mae  = float(np.mean(abs_errs))
print(f'NCF RMSE={ncf_rmse:.4f}  MAE={ncf_mae:.4f}  (samples={len(errs)})')

# ─── Final ranking ───
print('\n' + '=' * 50)
print(f'{"Model":15s}{"RMSE":>10s}{"MAE":>10s}')
print('-' * 50)
ranked = sorted([
    ('Baseline', baseline_rmse, baseline_mae),
    ('ALS',      als_rmse,      als_mae),
    ('NCF',      ncf_rmse,      ncf_mae),
], key=lambda x: x[1])
for i, (name, rmse, mae) in enumerate(ranked, 1):
    marker = '   ★ WINNER' if i == 1 else ''
    print(f'{name:15s}{rmse:>10.4f}{mae:>10.4f}{marker}')
print('=' * 50)
spark.stop()
