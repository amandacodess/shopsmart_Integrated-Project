"""
Run this script once to generate models/clean_data.csv
from a raw Kaggle Amazon dataset, OR to create synthetic demo data.

Usage:
    python generate_data.py                        # creates synthetic data
    python generate_data.py --input raw.csv        # cleans a real Kaggle CSV
"""

import os, sys, argparse
import pandas as pd
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
OUT = os.path.join(MODELS_DIR, 'clean_data.csv')

CATEGORIES = {
    'Electronics':   ['Smartphones', 'Laptops', 'Headphones', 'Cameras', 'Tablets'],
    'Clothing':      ['Men Shirts', 'Women Dresses', 'Footwear', 'Accessories', 'Jackets'],
    'Home & Kitchen':['Cookware', 'Furniture', 'Bedding', 'Cleaning', 'Storage'],
    'Books':         ['Fiction', 'Self-Help', 'Tech', 'History', 'Children'],
    'Sports':        ['Fitness', 'Cricket', 'Football', 'Cycling', 'Yoga'],
    'Beauty':        ['Skincare', 'Haircare', 'Makeup', 'Fragrance', 'Tools'],
    'Toys':          ['Action Figures', 'Board Games', 'Educational', 'Outdoor', 'Puzzles'],
    'Grocery':       ['Snacks', 'Beverages', 'Dairy', 'Spices', 'Organic'],
}

ADJECTIVES = ['Premium', 'Pro', 'Ultra', 'Smart', 'Essential', 'Classic', 'Elite', 'Lite', 'Max', 'Mini']
BRANDS     = ['Samsung', 'Sony', 'Apple', 'boAt', 'Philips', 'Prestige', 'Puma', 'Lakme',
               'Hasbro', 'Nestle', 'Nike', 'Canon', 'Lenovo', 'Xiaomi', 'HP', 'Dell']

IMAGE_PLACEHOLDERS = [
    'https://via.placeholder.com/200x200/1a1a2e/ffffff?text=Product',
    'https://via.placeholder.com/200x200/16213e/ffffff?text=Item',
    'https://via.placeholder.com/200x200/0f3460/ffffff?text=Shop',
]

def make_synthetic(n=500):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        main_cat = rng.choice(list(CATEGORIES.keys()))
        sub_cat  = rng.choice(CATEGORIES[main_cat])
        brand    = rng.choice(BRANDS)
        adj      = rng.choice(ADJECTIVES)
        name     = f"{brand} {adj} {sub_cat} {i+1}"
        actual   = round(float(rng.integers(200, 80000)), 2)
        discount = round(actual * rng.uniform(0.5, 0.95), 2)
        rating   = round(float(rng.uniform(2.5, 5.0)), 1)
        n_rating = int(rng.integers(10, 50000))
        img      = IMAGE_PLACEHOLDERS[i % len(IMAGE_PLACEHOLDERS)]
        rows.append({
            'id':            i + 1,
            'name':          name,
            'main_category': main_cat,
            'sub_category':  sub_cat,
            'image':         img,
            'link':          '#',
            'ratings':       rating,
            'no_of_ratings': n_rating,
            'discount_price': discount,
            'actual_price':   actual,
        })
    return pd.DataFrame(rows)

def clean_kaggle(path):
    df = pd.read_csv(path)
    # Standardise column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Try to map known Kaggle column variants
    renames = {
        'product_name': 'name',
        'category':     'main_category',
        'rating':       'ratings',
        'rating_count': 'no_of_ratings',
        'discounted_price': 'discount_price',
        'product_id':   'id',
    }
    df.rename(columns={k: v for k, v in renames.items() if k in df.columns}, inplace=True)
    if 'ratings' in df.columns:
        df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
    if 'discount_price' in df.columns:
        df['discount_price'] = pd.to_numeric(
            df['discount_price'].astype(str).str.replace('[₹,]', '', regex=True), errors='coerce')
    if 'actual_price' in df.columns:
        df['actual_price'] = pd.to_numeric(
            df['actual_price'].astype(str).str.replace('[₹,]', '', regex=True), errors='coerce')
    df.dropna(subset=['name'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if 'id' not in df.columns:
        df.insert(0, 'id', range(1, len(df)+1))
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, help='Path to raw Kaggle CSV')
    args = parser.parse_args()
    if args.input:
        print(f"Cleaning {args.input} ...")
        df = clean_kaggle(args.input)
    else:
        print("Generating synthetic demo data (500 products)...")
        df = make_synthetic(500)
    df.to_csv(OUT, index=False)
    print(f"Saved → {OUT}  ({len(df)} rows)")

if __name__ == '__main__':
    main()