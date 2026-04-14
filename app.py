from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'models', 'clean_data.csv')

df = pd.read_csv(DATA_PATH)

# Ensure required columns exist with safe defaults
for col in ['name','main_category','sub_category','image','link','ratings','no_of_ratings','discount_price','actual_price']:
    if col not in df.columns:
        df[col] = '' if col in ['name','main_category','sub_category','image','link'] else 0

# Clean numeric columns
for col in ['ratings','no_of_ratings','discount_price','actual_price']:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace('[₹,]','',regex=True).str.strip(),
        errors='coerce'
    ).fillna(0)

df.reset_index(drop=True, inplace=True)
if 'id' not in df.columns:
    df.insert(0, 'id', range(1, len(df)+1))
df['id'] = df['id'].astype(str)

# ─────────────────────────────────────────────
# DSA 1: HASH MAP — O(1) product lookup
# ─────────────────────────────────────────────
product_hash = {}
for _, row in df.iterrows():
    pid = str(row['id'])
    product_hash[pid] = {
        'id':            pid,
        'name':          str(row.get('name', '')),
        'category':      str(row.get('main_category', row.get('category', 'General'))),
        'sub_category':  str(row.get('sub_category', '')),
        'image':         str(row.get('image', '')),
        'link':          str(row.get('link', '#')),
        'ratings':       float(row.get('ratings', 0)),
        'no_of_ratings': int(row.get('no_of_ratings', 0)),
        'discount_price':float(row.get('discount_price', 0)),
        'actual_price':  float(row.get('actual_price', 0)),
    }

all_products = list(product_hash.values())

# ─────────────────────────────────────────────
# DSA 2: LINEAR SEARCH — O(n)
# ─────────────────────────────────────────────
def linear_search(products, query):
    q = query.lower()
    return [p for p in products if q in p['name'].lower() or q in p['category'].lower()]

# ─────────────────────────────────────────────
# DSA 3: BINARY SEARCH — O(log n) on sorted list
# ─────────────────────────────────────────────
def binary_search_by_rating(products, target_rating):
    sorted_p = sorted(products, key=lambda x: x['ratings'])
    lo, hi = 0, len(sorted_p) - 1
    result_idx = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_p[mid]['ratings'] == target_rating:
            result_idx = mid
            break
        elif sorted_p[mid]['ratings'] < target_rating:
            lo = mid + 1
            result_idx = mid
        else:
            hi = mid - 1
    start = max(0, result_idx - 10)
    return sorted_p[start: start + 20]

# ─────────────────────────────────────────────
# DSA 4: SORTING — O(n log n) Timsort
# ─────────────────────────────────────────────
def sort_products(products, by='ratings', reverse=True):
    valid = {'ratings', 'discount_price', 'actual_price', 'no_of_ratings'}
    if by not in valid:
        by = 'ratings'
    return sorted(products, key=lambda x: x.get(by, 0), reverse=reverse)

# ─────────────────────────────────────────────
# ML 1: CONTENT-BASED — TF-IDF + Cosine Similarity
# ─────────────────────────────────────────────
df['content'] = (
    df['name'].fillna('') + ' ' +
    df.get('main_category', df.get('category', pd.Series([''] * len(df)))).fillna('') + ' ' +
    df.get('sub_category', pd.Series([''] * len(df))).fillna('')
)

tfidf   = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['content'])

def content_based_recommend(product_name, top_n=8):
    matches = df[df['name'].str.lower().str.contains(product_name.lower(), na=False)]
    if matches.empty:
        # Try category match as fallback
        matches = df[df['main_category'].str.lower().str.contains(product_name.lower(), na=False)]
    if matches.empty:
        return []
    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1: top_n + 1]
    recs = []
    for i in sim_indices:
        pid = str(df.iloc[i]['id'])
        p   = product_hash.get(pid, all_products[i % len(all_products)])
        recs.append({**p, 'similarity': round(float(sim_scores[i]), 3), 'method': 'content'})
    return recs

# ─────────────────────────────────────────────
# ML 2: HYBRID — TF-IDF + SVD (Collaborative-style)
# Solves Cold Start Problem by combining both
# ─────────────────────────────────────────────
n_components = min(50, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
svd   = TruncatedSVD(n_components=n_components, random_state=42)
svd_matrix = normalize(svd.fit_transform(tfidf_matrix))

def hybrid_recommend(product_name, top_n=8):
    matches = df[df['name'].str.lower().str.contains(product_name.lower(), na=False)]
    if matches.empty:
        matches = df[df['main_category'].str.lower().str.contains(product_name.lower(), na=False)]
    if matches.empty:
        return content_based_recommend(product_name, top_n)

    idx = matches.index[0]

    # Content-based scores (TF-IDF)
    content_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # SVD latent scores
    svd_scores = svd_matrix @ svd_matrix[idx]

    # Weighted hybrid: 60% content + 40% SVD
    hybrid_scores = 0.6 * content_scores + 0.4 * svd_scores

    sim_indices = hybrid_scores.argsort()[::-1][1: top_n + 1]
    recs = []
    for i in sim_indices:
        pid = str(df.iloc[i]['id'])
        p   = product_hash.get(pid, all_products[i % len(all_products)])
        recs.append({**p, 'similarity': round(float(hybrid_scores[i]), 3), 'method': 'hybrid'})
    return recs

# ─────────────────────────────────────────────
# ANALYTICS HELPERS
# ─────────────────────────────────────────────
def get_category_col():
    return 'main_category' if 'main_category' in df.columns else 'category'

def get_category_stats():
    col = get_category_col()
    counts = df[col].value_counts().head(10)
    return [{'category': str(k), 'count': int(v)} for k, v in counts.items()]

def get_top_rated(n=10):
    top = df.nlargest(n, 'ratings')
    result = []
    for _, row in top.iterrows():
        pid = str(row['id'])
        p   = product_hash.get(pid, {})
        result.append({
            'name':     str(row.get('name', ''))[:60],
            'ratings':  float(row.get('ratings', 0)),
            'category': str(row.get('main_category', row.get('category', ''))),
            'image':    p.get('image', ''),
        })
    return result

def get_trending(n=8):
    """Trending = high ratings AND high review count"""
    tmp = df.copy()
    tmp['score'] = tmp['ratings'] * np.log1p(tmp['no_of_ratings'])
    top = tmp.nlargest(n, 'score')
    result = []
    for _, row in top.iterrows():
        pid = str(row['id'])
        p   = product_hash.get(pid, {})
        result.append(p)
    return result

def get_price_distribution():
    if 'discount_price' not in df.columns:
        return []
    prices = df['discount_price'].dropna()
    prices = prices[prices > 0]
    bins   = [0, 500, 1000, 2000, 5000, 10000, 50000]
    labels = ['<500', '500–1K', '1K–2K', '2K–5K', '5K–10K', '>10K']
    cut    = pd.cut(prices, bins=bins, labels=labels)
    dist   = cut.value_counts().reindex(labels).fillna(0)
    return [{'range': str(k), 'count': int(v)} for k, v in dist.items()]

def get_summary_stats():
    col = get_category_col()
    return {
        'total_products':    len(df),
        'total_categories':  int(df[col].nunique()),
        'avg_rating':        round(float(df['ratings'].mean()), 2) if 'ratings' in df.columns else 0,
        'top_category':      str(df[col].value_counts().idxmax()) if col in df.columns else 'N/A',
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('main.html')

@app.route('/api/products')
def api_products():
    page     = max(1, int(request.args.get('page', 1)))
    per_page = min(50, max(1, int(request.args.get('per_page', 20))))
    sort_by  = request.args.get('sort', 'ratings')
    order    = request.args.get('order', 'desc') == 'desc'
    category = request.args.get('category', '').strip()

    products = all_products.copy()
    if category:
        products = [p for p in products if category.lower() in p['category'].lower()]

    products = sort_products(products, by=sort_by, reverse=order)

    total = len(products)
    start = (page - 1) * per_page
    return jsonify({
        'products': products[start: start + per_page],
        'total':    total,
        'page':     page,
        'per_page': per_page,
        'pages':    (total + per_page - 1) // per_page
    })

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    algo  = request.args.get('algo', 'linear')
    if not query:
        return jsonify({'results': [], 'count': 0, 'algo': algo})
    if algo == 'binary':
        try:
            results = binary_search_by_rating(all_products, float(query))
        except ValueError:
            results = linear_search(all_products, query)
    else:
        results = linear_search(all_products, query)
    return jsonify({'results': results[:40], 'count': len(results), 'algo': algo})

@app.route('/api/recommend')
def api_recommend():
    name   = request.args.get('name', '').strip()
    mode   = request.args.get('mode', 'hybrid')  # 'hybrid' or 'content'
    if not name:
        return jsonify({'recommendations': [], 'query': '', 'mode': mode})
    recs = hybrid_recommend(name) if mode == 'hybrid' else content_based_recommend(name)
    return jsonify({'recommendations': recs, 'query': name, 'mode': mode})

@app.route('/api/trending')
def api_trending():
    return jsonify(get_trending())

@app.route('/api/product/<pid>')
def api_product(pid):
    p = product_hash.get(str(pid))
    if not p:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(p)

@app.route('/api/categories')
def api_category_list():
    col  = get_category_col()
    cats = sorted(df[col].dropna().unique().tolist())
    return jsonify(cats)

@app.route('/api/analytics/summary')
def api_summary():
    return jsonify(get_summary_stats())

@app.route('/api/analytics/categories')
def api_categories():
    return jsonify(get_category_stats())

@app.route('/api/analytics/top-rated')
def api_top_rated():
    return jsonify(get_top_rated())

@app.route('/api/analytics/price-distribution')
def api_price_dist():
    return jsonify(get_price_distribution())

if __name__ == "__main__":
    app.run(
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 5000))
    )
