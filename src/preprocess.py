"""Complete Data Preprocessing for IEEE-CIS Fraud Detection"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DATA PREPROCESSING PIPELINE")
print("="*60)

# ========== SETUP PATHS ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Create processed directory if it doesn't exist
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data dir: {DATA_RAW}")
print(f"Processed dir: {DATA_PROCESSED}")

# ========== LOAD DATA ==========
print("\n📂 Loading data from parquet file...")

# Load the single parquet file
df = pd.read_parquet(DATA_RAW / 'ieee_fraud_detection.parquet')

print(f"✓ Data loaded!")
print(f"  - Shape: {df.shape}")
print(f"  - Columns: {len(df.columns)}")

# Check if this contains both train and test
if 'isFraud' in df.columns:
    train_df = df[df['isFraud'].notna()].copy()
    test_df = df[df['isFraud'].isna()].copy() if df['isFraud'].isna().any() else None
    
    print(f"\n📊 Data split:")
    print(f"  - Training samples: {len(train_df):,}")
    if test_df is not None:
        print(f"  - Test samples: {len(test_df):,}")
    print(f"  - Fraud rate: {train_df['isFraud'].mean():.4f}")
else:
    print("\n⚠️ No 'isFraud' column found. Assuming all data is training data.")
    train_df = df
    test_df = None

# ========== 1. INITIAL DATA INSPECTION ==========
print("\n" + "="*60)
print("1. INITIAL DATA INSPECTION")
print("="*60)

# Check missing values
missing_pct = (train_df.isnull().sum() / len(train_df) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 50]
print(f"\n🔍 Missing values analysis:")
print(f"  - Columns with >50% missing: {len(high_missing)}")
if len(high_missing) > 0:
    print(f"  - Example: {high_missing.head(3).index.tolist()}")

# ========== 2. IDENTIFY COLUMN TYPES ==========
print("\n" + "="*60)
print("2. IDENTIFYING COLUMN TYPES")
print("="*60)

v_cols = [c for c in train_df.columns if c.startswith('V')]
d_cols = [c for c in train_df.columns if c.startswith('D') and c[1:].isdigit()]
c_cols = [c for c in train_df.columns if c.startswith('C')]
id_cols = [c for c in train_df.columns if c.startswith('id_')]
card_cols = [c for c in train_df.columns if c.startswith('card')]
addr_cols = [c for c in train_df.columns if c.startswith('addr')]

print(f"  - V columns (anonymized): {len(v_cols)}")
print(f"  - D columns (datetime): {len(d_cols)}")
print(f"  - C columns (binned): {len(c_cols)}")
print(f"  - ID columns: {len(id_cols)}")
print(f"  - Card columns: {len(card_cols)}")
print(f"  - Address columns: {len(addr_cols)}")

# ========== 3. HANDLE MISSING VALUES ==========
print("\n" + "="*60)
print("3. HANDLING MISSING VALUES")
print("="*60)

def handle_missing(df):
    """Handle missing values based on column type"""
    df_clean = df.copy()
    
    # Fill V columns with -999 (anonymized)
    for col in v_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(-999)
    
    # Fill C columns with -1 (binned values)
    for col in c_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(-1)
    
    # Fill D columns with 0 (day differences)
    for col in d_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # Fill card and address columns with -1
    for col in card_cols + addr_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(-1)
    
    # Fill ID columns with -1
    for col in id_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(-1)
    
    # Fill categorical columns with 'missing'
    cat_cols = ['ProductCD', 'card4', 'card5', 'card6', 'P_emaildomain', 
                'R_emaildomain', 'DeviceType', 'DeviceInfo']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('missing').astype(str)
    
    return df_clean

print("Applying missing value handling...")
train_clean = handle_missing(train_df)
if test_df is not None:
    test_clean = handle_missing(test_df)

print(f"✓ Missing values in train: {train_clean.isnull().sum().sum()}")

# ========== 4. FEATURE ENGINEERING ==========
print("\n" + "="*60)
print("4. FEATURE ENGINEERING")
print("="*60)

def create_features(df):
    """Create new features"""
    df_feat = df.copy()
    
    # Log transaction amount (handles skewness)
    df_feat['TransactionAmt_log'] = np.log1p(df_feat['TransactionAmt'])
    
    # Time-based features
    if 'TransactionDT' in df_feat.columns:
        df_feat['TransactionHour'] = (df_feat['TransactionDT'] / 3600) % 24
        df_feat['TransactionDay'] = (df_feat['TransactionDT'] / (24*3600)) % 7
    
    # UID combinations
    if 'card1' in df_feat.columns and 'addr1' in df_feat.columns:
        df_feat['card1_addr1'] = (df_feat['card1'].astype(str) + '_' + 
                                   df_feat['addr1'].astype(str))
    
    # Missing value count per row
    df_feat['missing_count'] = df.isnull().sum(axis=1)
    
    return df_feat

print("Creating new features...")
train_feat = create_features(train_clean)
if test_df is not None:
    test_feat = create_features(test_clean)

print(f"✓ Added features. Train shape: {train_feat.shape}")

# ========== 5. ENCODE CATEGORICAL VARIABLES ==========
print("\n" + "="*60)
print("5. ENCODING CATEGORICAL VARIABLES")
print("="*60)

# Get categorical columns (object type)
cat_cols_train = train_feat.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {len(cat_cols_train)}")

# Encode using train only to avoid leakage
label_encoders = {}
for col in cat_cols_train:
    le = LabelEncoder()
    # Fit on train
    train_feat[col] = le.fit_transform(train_feat[col].astype(str))
    # Transform test if exists
    if test_df is not None and col in test_feat.columns:
        test_feat[col] = test_feat[col].astype(str)
        # Handle unseen categories in test
        for val in test_feat[col].unique():
            if val not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
        test_feat[col] = le.transform(test_feat[col])
    
    label_encoders[col] = le
    print(f"  ✓ {col}: {train_feat[col].nunique()} unique values")

# ========== 6. SELECT FEATURES FOR MODEL ==========
print("\n" + "="*60)
print("6. SELECTING FEATURES FOR MODEL")
print("="*60)

# Columns to drop (not features)
drop_cols = ['TransactionID', 'isFraud']

# All other columns are features
feature_cols = [c for c in train_feat.columns if c not in drop_cols]
print(f"Total features: {len(feature_cols)}")

# Separate features and target for train
X_train = train_feat[feature_cols].copy()
y_train = train_feat['isFraud'].copy()

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ y_train shape: {y_train.shape}")

# For test
if test_df is not None:
    X_test = test_feat[feature_cols].copy()
    test_ids = test_feat['TransactionID'].copy()
    print(f"✓ X_test shape: {X_test.shape}")
else:
    X_test = None
    test_ids = None

# ========== 7. SAVE PROCESSED DATA ==========
print("\n" + "="*60)
print("7. SAVING PROCESSED DATA")
print("="*60)

# Save processed dataframes
train_feat.to_parquet(DATA_PROCESSED / 'train_processed.parquet', index=False)
print(f"✓ Saved: {DATA_PROCESSED / 'train_processed.parquet'}")

# Save feature matrix
X_train.to_parquet(DATA_PROCESSED / 'X_train.parquet', index=False)
y_train.to_frame().to_parquet(DATA_PROCESSED / 'y_train.parquet', index=False)
print(f"✓ Saved: X_train, y_train")

if X_test is not None:
    X_test.to_parquet(DATA_PROCESSED / 'X_test.parquet', index=False)
    print(f"✓ Saved: X_test")
    
    if test_ids is not None:
        test_ids.to_frame().to_parquet(DATA_PROCESSED / 'test_ids.parquet', index=False)
        print(f"✓ Saved: test_ids")

# Save feature list
with open(DATA_PROCESSED / 'features.txt', 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")
print(f"✓ Saved: {DATA_PROCESSED / 'features.txt'}")

# ========== 8. FINAL SUMMARY ==========
print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print(f"\n📊 Final Summary:")
print(f"  - Train samples: {len(X_train):,}")
if X_test is not None:
    print(f"  - Test samples: {len(X_test):,}")
print(f"  - Features: {len(feature_cols)}")
print(f"  - Fraud rate: {y_train.mean():.4f}")
print(f"\n💾 Processed data saved in: {DATA_PROCESSED}")
print("\n✅ Ready for model training!")

# Return data for use in notebook/script
return_dict = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'test_ids': test_ids,
    'feature_cols': feature_cols
}