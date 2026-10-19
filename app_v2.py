"""
Upgraded Flask web app for the recommendation system.
Adds: NCF realtime prediction (/api/ncf_recommend/<uid>), model health (/model_health), monitor dashboard (/monitor).
"""
from flask import Flask, render_template, request, jsonify
import happybase, json, time, csv, subprocess, os
from collections import deque
from io import StringIO

import numpy as np
import pandas as pd
from confluent_kafka import Producer

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError as _e:
    print(f'torch not installed; NCF features disabled: {_e}')
    torch = None
    nn = None
    TORCH_AVAILABLE = False

app = Flask(__name__)
HBASE_HOST = 'master'
HBASE_PORT = 9090
kafka_conf = {'bootstrap.servers': 'worker1:9092,worker2:9092,worker3:9092'}
producer = Producer(kafka_conf)

# ─── Movies (HDFS) ───
MOVIES = {}
try:
    proc = subprocess.run(['hdfs', 'dfs', '-cat', '/data/movielens/movies.csv'],
                          capture_output=True, text=True, timeout=30)
    for row in csv.DictReader(proc.stdout.strip().split('\n')):
        MOVIES[int(row['movieId'])] = {'title': row['title'], 'genres': row['genres']}
    print(f'Loaded {len(MOVIES)} movies')
except Exception as e:
    print(f'Movies load warning: {e}')
    MOVIES = {1: {'title': 'Toy Story', 'genres': 'Animation'}}


# ─── NCF model (HDFS → /tmp → torch) ───
NCF_OK = False
ncf_model = None
user2idx = {}
item2idx = {}
idx2item = {}
ncf_train_rmse = None

if TORCH_AVAILABLE:
    class NCF(nn.Module):
        def __init__(self, n_users, n_items, embed_dim=64):
            super().__init__()
            self.user_embed = nn.Embedding(n_users, embed_dim)
            self.item_embed = nn.Embedding(n_items, embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1))

        def forward(self, user, item):
            u = self.user_embed(user)
            i = self.item_embed(item)
            return self.mlp(torch.cat([u, i], dim=1)).squeeze()

    def load_ncf_model():
        subprocess.run(['hdfs', 'dfs', '-get', '-f',
                        '/models/ncf_model/ncf_checkpoint.pt',
                        '/tmp/ncf_checkpoint.pt'], check=True)
        ckpt = torch.load('/tmp/ncf_checkpoint.pt', map_location='cpu', weights_only=False)
        m = NCF(ckpt['n_users'], ckpt['n_items'], ckpt['embed_dim'])
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        return m, ckpt['user2idx'], ckpt['item2idx'], ckpt['idx2item'], ckpt.get('test_rmse')

    try:
        ncf_model, user2idx, item2idx, idx2item, ncf_train_rmse = load_ncf_model()
        print(f'NCF model loaded. Training test RMSE was: {ncf_train_rmse}')
        NCF_OK = True
    except Exception as e:
        print(f'NCF load failed: {e}')


# ─── Model scorer (sliding window MAE/RMSE) ───
class ModelScorer:
    def __init__(self, window_size=500, alert_threshold=1.0):
        self.errors = deque(maxlen=window_size)
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.total_scored = 0
        self.alert_count = 0

    def score(self, user_id, item_id, actual_rating):
        if not NCF_OK or user_id not in user2idx or item_id not in item2idx:
            return None
        with torch.no_grad():
            pred = ncf_model(
                torch.tensor([user2idx[user_id]]),
                torch.tensor([item2idx[item_id]])
            ).item()
        err = abs(actual_rating - pred)
        self.errors.append(err)
        self.total_scored += 1
        return pred, err

    def get_metrics(self):
        if not self.errors:
            return {'status': 'NO_DATA', 'mae': None, 'rmse': None,
                    'window_size': 0, 'total_scored': self.total_scored,
                    'alert_count': self.alert_count, 'threshold': self.alert_threshold}
        errs = np.array(self.errors)
        mae = float(errs.mean())
        rmse = float(np.sqrt((errs ** 2).mean()))
        if mae < 0.7: status = 'EXCELLENT'
        elif mae < self.alert_threshold: status = 'OK'
        elif mae < 1.3: status = 'WARNING'
        else: status = 'ALERT_RETRAIN'
        if status in ('WARNING', 'ALERT_RETRAIN'):
            self.alert_count += 1
        return {
            'status': status,
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'window_size': len(self.errors),
            'total_scored': self.total_scored,
            'alert_count': self.alert_count,
            'threshold': self.alert_threshold,
        }


scorer = ModelScorer(window_size=500, alert_threshold=1.0)


# ─── Baseline rating distribution (for PSI / Data Drift) ───
BASELINE_DIST = {}
BASELINE_MEAN = 3.5
try:
    proc = subprocess.run(['hdfs', 'dfs', '-cat', '/data/movielens/ratings.csv'],
                          capture_output=True, text=True, timeout=30)
    base_df = pd.read_csv(StringIO(proc.stdout))
    BASELINE_DIST = {float(k): float(v) for k, v in
                     base_df['rating'].value_counts(normalize=True).to_dict().items()}
    BASELINE_MEAN = float(base_df['rating'].mean())
    print(f'Baseline rating mean: {BASELINE_MEAN:.3f}')
except Exception as e:
    print(f'Baseline load warning: {e}')
    BASELINE_DIST = {1.0: 0.06, 2.0: 0.11, 3.0: 0.27, 4.0: 0.35, 5.0: 0.21}

recent_ratings_window = deque(maxlen=1000)


def calc_psi(baseline_dist, recent_ratings, bins=(1.0, 2.0, 3.0, 4.0, 5.0)):
    if len(recent_ratings) < 10:
        return 0.0
    eps = 1e-4
    psi = 0.0
    n = len(recent_ratings)
    for r in bins:
        b = max(baseline_dist.get(r, eps), eps)
        q = max(sum(1 for x in recent_ratings if x == r) / n, eps)
        psi += (q - b) * np.log(q / b)
    return psi


# ─── HBase / Kafka helpers ───
def get_recommendations(uid):
    try:
        conn = happybase.Connection(HBASE_HOST, port=HBASE_PORT)
        row = conn.table('recommendations').row(str(uid).encode())
        conn.close()
        if row and b'recs:top20' in row:
            return json.loads(row[b'recs:top20'])
    except Exception as e:
        print(f'HBase error: {e}')
    return []


def send_rating(uid, mid, r):
    uid_i, mid_i, r_f = int(uid), int(mid), float(r)
    evt = json.dumps({'userId': uid_i, 'movieId': mid_i,
                      'rating': r_f, 'timestamp': int(time.time())})
    producer.produce('user-ratings', evt.encode())
    producer.flush()
    scorer.score(uid_i, mid_i, r_f)
    recent_ratings_window.append(r_f)


# ─── Routes ───
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/recommend')
def recommend():
    uid = request.args.get('user_id', '1')
    return render_template('recommend.html', user_id=uid, recs=get_recommendations(uid))


@app.route('/rate', methods=['GET', 'POST'])
def rate():
    if request.method == 'POST':
        uid = request.form['user_id']
        mid = request.form['movie_id']
        r = request.form['rating']
        send_rating(uid, mid, r)
        return render_template('rate_done.html', user_id=uid,
                               movie=MOVIES.get(int(mid), {}).get('title', mid),
                               rating=r)
    return render_template('rate.html', movies=sorted(MOVIES.items())[:50])


@app.route('/api/recommend/<int:uid>')
def api_rec(uid):
    return jsonify({'userId': uid, 'recommendations': get_recommendations(uid)})


@app.route('/api/ncf_recommend/<int:uid>')
def ncf_recommend(uid):
    """NCF realtime Top-20 (computed on the fly)."""
    if not NCF_OK:
        return jsonify({'error': 'NCF model not loaded'}), 503
    if uid not in user2idx:
        return jsonify({'error': f'User {uid} not in NCF training set (cold start)'}), 404
    uidx = user2idx[uid]
    n = len(item2idx)
    all_items = torch.arange(n)
    user_tensor = torch.full_like(all_items, uidx)
    with torch.no_grad():
        scores = ncf_model(user_tensor, all_items).numpy()
    top_idx = scores.argsort()[::-1][:20]
    recs = [{'movieId': int(idx2item[i]),
             'title': MOVIES.get(int(idx2item[i]), {}).get('title', '?'),
             'score': float(scores[i])} for i in top_idx]
    return jsonify({'userId': uid, 'model': 'NCF', 'recommendations': recs})


@app.route('/model_health')
def model_health():
    return jsonify(scorer.get_metrics())


@app.route('/simulate_ratings', methods=['POST'])
def simulate_ratings():
    """Inject N random ratings into the system. Sends to Kafka + scores directly."""
    import random
    try:
        n = int(request.form.get('n', 100))
    except ValueError:
        n = 100
    n = max(1, min(n, 500))  # cap [1, 500] to prevent abuse

    if not user2idx or not item2idx:
        return jsonify({'sent': 0, 'error': 'NCF model not loaded; cannot score'}), 503

    user_pool = list(user2idx.keys())
    item_pool = list(item2idx.keys())
    sent = 0
    for _ in range(n):
        uid = random.choice(user_pool)
        mid = random.choice(item_pool)
        r = random.choice([1.0, 2.0, 3.0, 4.0, 5.0])
        evt = json.dumps({'userId': int(uid), 'movieId': int(mid),
                          'rating': float(r), 'timestamp': int(time.time())})
        producer.produce('user-ratings', evt.encode())
        scorer.score(int(uid), int(mid), float(r))
        recent_ratings_window.append(float(r))
        sent += 1
    producer.flush()
    return jsonify({'sent': sent, 'metrics': scorer.get_metrics()})


@app.route('/monitor')
def monitor():
    m = scorer.get_metrics()
    psi = calc_psi(BASELINE_DIST, recent_ratings_window)
    if psi < 0.1:   drift_status = 'STABLE'
    elif psi < 0.2: drift_status = 'WARNING'
    else:           drift_status = 'ALERT'
    recent_dist = {}
    if recent_ratings_window:
        n = len(recent_ratings_window)
        for r in (1.0, 2.0, 3.0, 4.0, 5.0):
            recent_dist[r] = sum(1 for x in recent_ratings_window if x == r) / n
    metrics = {
        'mae': m.get('mae'),
        'rmse': m.get('rmse'),
        'status': m.get('status', 'NO_DATA'),
        'total_scored': m.get('total_scored', 0),
        'psi': round(float(psi), 4),
        'drift_status': drift_status,
        'baseline_dist': BASELINE_DIST,
        'recent_dist': recent_dist,
        'recent_count': len(recent_ratings_window),
        'need_retrain': (psi > 0.2) or (m.get('status') in ('WARNING', 'ALERT_RETRAIN')),
    }
    return render_template('monitor.html', metrics=metrics)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
