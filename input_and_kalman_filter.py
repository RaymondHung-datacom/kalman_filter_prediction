import json
import os
import glob
import numpy as np
import pickle
import pandas as pd  # Add to imports at top
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, models
import joblib

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
# Directories to watch
DIR_CURRENT_WEATHER = "current_weather_reports"
DIR_FORECAST_WEATHER = "forecast_weather_reports"
QDRANT_COLLECTION = "retail_news_jp"

# Qdrant Setup
client = QdrantClient("localhost", port=6333)

# ==========================================
# 1. PCA MODEL INITIALIZATION
# ==========================================

PCA_MODEL_FILE = "pca_model.pkl"
PCA_DIM = 3  # Reduce 768 -> 3 dimensions

def load_or_create_pca():
    """
    Load existing PCA model, or create fresh one if missing.
    If created fresh, will be fitted on first news batch.
    """
    if os.path.exists(PCA_MODEL_FILE):
        try:
            pca = joblib.load(PCA_MODEL_FILE)
            print(f"✅ Loaded PCA model from {PCA_MODEL_FILE}")
            return pca
        except Exception as e:
            print(f"⚠️  Failed to load PCA: {e}. Creating fresh model.")
    
    # Create fresh PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM)
    print(f"✅ Created fresh PCA model (will fit on 768 -> {PCA_DIM} dimensions)")
    return pca

def save_pca(pca):
    """Save fitted PCA model for reuse"""
    try:
        joblib.dump(pca, PCA_MODEL_FILE)
        print(f"✅ Saved PCA model to {PCA_MODEL_FILE}")
    except Exception as e:
        print(f"⚠️  Failed to save PCA: {e}")

# Initialize PCA at startup
pca_reducer = load_or_create_pca()
pca_is_fitted = pca_reducer.n_components_ is not None if hasattr(pca_reducer, 'n_components_') else False

# ==========================================
# 2. FILE MANAGEMENT (The "Folder Watcher")
# ==========================================
def get_latest_file(directory, extension="*.json"):
    """
    Scans a directory and returns the file with the most recent name/timestamp.
    Assumes filenames like 'Region_YYYYMMDD_HHMMSS.json' which sort chronologically.
    """
    search_path = os.path.join(directory, extension)
    files = sorted(glob.glob(search_path))
    
    if not files:
        return None
    
    # Return the last file in the sorted list (Latest timestamp)
    return files[-1]

# ==========================================
# 3. DATA LOADERS
# ==========================================

def load_actual_sales_from_csv(filepath="sales_data.csv"):
    """
    Load today's actual sales count from CSV
    
    Expected format:
        date,item_id,item_name,sales_count
        2025-12-03,SKU001,Milk,245
        2025-12-03,SKU002,Bread,189
    
    Returns: Total sales count for today, or None if no data found
    """
    try:
        df = pd.read_csv(filepath)
        
        # Get today's date in the same format as CSV
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Filter for today's sales
        today_sales_df = df[df['date'] == today]
        
        if today_sales_df.empty:
            print(f"⚠️  No sales data found for {today} in {filepath}")
            print(f"   Available dates: {df['date'].unique()}")
            return None
        
        # Sum all items sold today
        total_sales = today_sales_df['sales_count'].sum()
        item_count = len(today_sales_df)
        
        print(f"✅ Loaded sales data for {today}:")
        print(f"   Total Units Sold: {total_sales}")
        print(f"   Items Tracked: {item_count}")
        print(f"   Breakdown: {dict(zip(today_sales_df['item_name'], today_sales_df['sales_count']))}")
        
        return float(total_sales)
        
    except FileNotFoundError:
        print(f"❌ Sales file not found: {filepath}")
        print(f"   Please create '{filepath}' with sales data")
        return None
    except Exception as e:
        print(f"❌ Error loading sales data: {e}")
        return None

def load_observation(filepath):
    """
    Input: Current Weather JSON
    Output: Normalized [Temp, Rain] Vector
    """
    print(f"   -> Reading Current Weather from: {os.path.basename(filepath)}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Normalize (Customize ranges for your region)
    # Temp: -5C to 35C -> 0.0 to 1.0
    temp = (data['temperature'] + 5) / 40.0
    # Rain: 0mm to 50mm -> 0.0 to 1.0
    rain = min(data['precipitation1h'], 50) / 50.0
    
    return np.array([temp, rain]), data['observationTime']

def load_forecast(filepath, current_time_iso):
    """
    Input: Forecast JSON + Current Time
    Output: Normalized [MaxTemp_NextDay, PoP_NextDay]
    Why: We want to know the forecast for TOMORROW, as that drives shopping today.
    """
    print(f"   -> Reading Forecast from: {os.path.basename(filepath)}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Calculate "Tomorrow" relative to the current observation
    current_dt = datetime.fromisoformat(current_time_iso)
    target_date_obj = current_dt + timedelta(days=1)
    target_date_str = target_date_obj.strftime("%Y-%m-%d")
    
    # Fetch forecast for that specific date
    forecast = data['forecasts'].get(target_date_str)
    
    # Fallback: If tomorrow is missing, use today (or handle error)
    if not forecast:
        print(f"      ⚠️ Forecast for {target_date_str} not found. Using available keys.")
        first_key = list(data['forecasts'].keys())[0]
        forecast = data['forecasts'][first_key]

    # 1. Normalize Max Temp
    try:
        max_t = float(forecast['max_temp'])
        norm_max_t = (max_t + 5) / 40.0
    except (ValueError, KeyError):
        norm_max_t = 0.5 # Default neutral

    # 2. Normalize Precipitation Probability (Maximum of the day)
    pops = forecast.get('pops', [])
    max_pop = 0
    valid_count = 0
    for p in pops:
        try:
            # Extract number from string like "00-06時: 10%"
            val = int(p.split(':')[1].replace('%', '').strip())
            max_pop = max(max_pop, val)
            valid_count += 1
        except:
            continue
            
    avg_max_pop = (max_pop / valid_count) if valid_count > 0 else 0
    norm_pop = avg_max_pop / 100.0

    return np.array([norm_max_t, norm_pop])

def get_weekly_news_context(target_date_iso):
    """
    Input: Target Date ISO (e.g., "2025-12-02T16:10:00+09:00")
    Output: 3-Dimensional PCA Vector representing the week's news
    
    On first run: Collects vectors and fits PCA if not already fitted
    On later runs: Uses fitted PCA to project vectors
    """
    global pca_reducer, pca_is_fitted
    
    # 1. Parse the full timestamp from the weather report
    target_dt = datetime.fromisoformat(target_date_iso)
    start_dt = target_dt - timedelta(days=7)
    
    # 2. GET DATE OBJECTS (Do not convert to string!)
    target_date_obj = target_dt.date()
    start_date_obj = start_dt.date()

    print(f"   -> Fetching News Context: {start_date_obj} to {target_date_obj}")

    # 3. Query Qdrant
    response = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="publish_date",
                    range=models.DatetimeRange(
                        gte=start_date_obj,
                        lte=target_date_obj
                    )
                )
            ]
        ),
        limit=50,
        with_vectors=True
    )
    
    points, _ = response
    
    if not points:
        print(f"      ⚠️  No news found for this period. Returning neutral vector.")
        return np.zeros(PCA_DIM)

    # 4. COLLECT ALL VECTORS (for potential PCA fitting)
    all_vectors = []
    weighted_sum = np.zeros(768)
    total_weight = 0.0
    
    for point in points:
        try:
            # Safely extract payload and publish_date
            payload = getattr(point, "payload", None) or {}
            pub_date_str = payload.get("publish_date") if isinstance(payload, dict) else None
            if not pub_date_str:
                continue

            # Support both string and datetime in payload
            if isinstance(pub_date_str, datetime):
                pub_date = pub_date_str.date()
            else:
                pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()

            # Safely get vector
            vec = getattr(point, "vector", None) or payload.get("vector")
            if vec is None:
                continue

            vec = np.asarray(vec, dtype=float)
            if vec.size != 768:
                continue

            # Collect vector for potential PCA fitting
            all_vectors.append(vec)
            
            # Weighted sum for projection
            days_old = (target_date_obj - pub_date).days
            weight = 0.8 ** max(0, days_old)
            weighted_sum += vec * weight
            total_weight += weight
            
        except Exception as e:
            continue
    
    if total_weight == 0:
        print(f"      ⚠️  No valid vectors found. Returning neutral vector.")
        return np.zeros(PCA_DIM)
    
    # 5. FIT PCA IF NOT ALREADY FITTED (First run only)
    if not pca_is_fitted and len(all_vectors) > 10:
        print(f"      📊 Fitting PCA on {len(all_vectors)} news vectors...")
        try:
            all_vectors_array = np.array(all_vectors, dtype=np.float32)
            pca_reducer.fit(all_vectors_array)
            pca_is_fitted = True
            save_pca(pca_reducer)
            print(f"      ✅ PCA fitted successfully (768 -> {PCA_DIM} dimensions)")
        except Exception as e:
            print(f"      ❌ PCA fitting failed: {e}")
            return np.zeros(PCA_DIM)
    
    # 6. PROJECT TO PCA SPACE
    avg_vec = weighted_sum / total_weight
    
    try:
        if pca_is_fitted:
            projected = pca_reducer.transform(avg_vec.reshape(1, -1)).flatten()
            print(f"      ✅ News vector projected to PCA space: {np.round(projected, 3)}")
            return projected
        else:
            print(f"      ⚠️  PCA not fitted yet. Returning neutral vector.")
            return np.zeros(PCA_DIM)
    except Exception as e:
        print(f"      ❌ PCA projection failed: {e}")
        return np.zeros(PCA_DIM)

# ==========================================
# 4. KALMAN FILTER ENGINE
# ==========================================
STATE_FILE = "kalman_filter_state.json"

class RetailKalmanFilter:
    def __init__(self):
        # State: [Sales Level, Momentum]
        self.x = np.array([[100.], [0.]])
        self.P = np.eye(2) * 500
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        
        # B Matrix: 2 states × 7 inputs
        # Inputs: [news_pca_1, news_pca_2, news_pca_3, temp, rain, max_temp_fcst, pop_fcst]
        self.B = np.zeros((2, 7))
        
        # How inputs affect momentum (row 1):
        self.B[1, 0:3] = 0.3   # News PCA components drive trend
        self.B[1, 3]   = -0.2  # Current temp affects momentum
        self.B[1, 4]   = -0.5  # Current rain dampens momentum
        self.B[1, 5]   = 0.1   # Forecast temp gives momentum
        self.B[1, 6]   = -0.3  # Forecast rain dampens momentum
        
        # Measurement noise
        self.R = np.array([[100.]])
        
        # Process noise
        self.Q = np.array([[10., 0.], [0., 5.]])

    def predict(self, u_vector):
        u = u_vector.reshape(-1, 1)
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement):
        """Correct state using actual sales measurement"""
        z = np.array([[measurement]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x

    def save_state(self, filepath="kalman_filter_state.json"):
        import json
        state_dict = {
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, "w") as f:
            json.dump(state_dict, f, indent=2)
        print(f"✅ State saved to {filepath}")

    def load_state(self, filepath="kalman_filter_state.json"):
        import json
        if not os.path.exists(filepath):
            print(f"ℹ️  No previous state found. Initializing fresh.")
            return False
        try:
            with open(filepath, "r") as f:
                state_dict = json.load(f)
            self.x = np.array(state_dict["x"])
            self.P = np.array(state_dict["P"])
            print(f"✅ State loaded from {filepath}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load state: {e}")
            return False

# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Retail Prediction Engine with PCA...\n")
    
    # 1. FIND LATEST FILES
    cur_weather_file = get_latest_file(DIR_CURRENT_WEATHER)
    fcst_weather_file = get_latest_file(DIR_FORECAST_WEATHER)
    
    if not cur_weather_file or not fcst_weather_file:
        print("❌ Error: Missing weather files")
        exit()

    # 2. LOAD & NORMALIZE DATA
    print("\n📥 Loading Input Data:")
    u_obs, obs_time = load_observation(cur_weather_file)
    u_fcst = load_forecast(fcst_weather_file, obs_time)
    u_news = get_weekly_news_context(obs_time)  # ← Returns 3-dim PCA vector
    
    # 3. FUSE INPUTS (3 + 2 + 2 = 7 dimensions)
    u_full = np.hstack([u_news, u_obs, u_fcst])
    
    print("\n📊 FUSED INPUT VECTOR (u) - 7 dimensions:")
    print(f"   [News PCA]     {np.round(u_news, 3)}")
    print(f"   [Current]      Temp: {u_obs[0]:.2f}, Rain: {u_obs[1]:.2f}")
    print(f"   [Forecast]     MaxT: {u_fcst[0]:.2f}, PoP: {u_fcst[1]:.2f}")
    print(f"   [Full vector]  {np.round(u_full, 3)}")
    
    # 4. INITIALIZE & RUN KALMAN FILTER
    print("\n🔧 Initializing Kalman Filter:")
    kf = RetailKalmanFilter()
    kf.load_state()
    
    prediction = kf.predict(u_full)
    
    print("\n" + "="*60)
    print(f"🔮 PREDICTION (before measurement)")
    print("="*60)
    print(f"📈 Predicted Sales: {prediction[0,0]:.2f} units")
    print(f"🚀 Momentum:        {prediction[1,0]:.4f}")
    
    # 5. LOAD & UPDATE WITH ACTUAL SALES
    print("\n📥 Loading Actual Sales:")
    actual_sales = load_actual_sales_from_csv("sales_data.csv")
    
    if actual_sales is not None:
        updated = kf.update(actual_sales)
        print("\n" + "="*60)
        print(f"✅ KALMAN FILTER UPDATE")
        print("="*60)
        print(f"📊 Measurement:    {actual_sales:.0f} units")
        print(f"📈 Corrected Pred:  {updated[0,0]:.2f} units")
        print(f"🚀 Corrected Mom:   {updated[1,0]:.4f}")
    
    # 6. SAVE STATE
    kf.save_state()
    print("\n✅ Complete!")