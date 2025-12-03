import json
import os
import glob
import numpy as np
import pickle
import pandas as pd  # Add to imports at top
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, models

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
# Directories to watch
DIR_CURRENT_WEATHER = "current_weather_reports"
DIR_FORECAST_WEATHER = "forecast_weather_reports"
QDRANT_COLLECTION = "retail_news_jp"

# Qdrant Setup
client = QdrantClient("localhost", port=6333)

# Load PCA Model (Using a dummy if file not found for safety)
try:
    with open("news_pca_model.pkl", "rb") as f:
        pca_reducer = pickle.load(f)
except FileNotFoundError:
    print("⚠️ PCA model not found. Initializing dummy reducer for testing.")
    from sklearn.decomposition import PCA
    pca_reducer = PCA(n_components=3)
    pca_reducer.fit(np.random.rand(100, 768))

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
    """
    # 1. Parse the full timestamp from the weather report
    target_dt = datetime.fromisoformat(target_date_iso)
    start_dt = target_dt - timedelta(days=7)
    
    # 2. GET DATE OBJECTS (Do not convert to string!)
    # The library expects datetime.date objects, not strings.
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
                        # FIX: Pass the 'date' objects directly. 
                        # The library will handle the string conversion.
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
        return np.zeros(3)

    weighted_sum = np.zeros(768)
    total_weight = 0.0
    
    for point in points:
        try:
            # Safely extract payload and publish_date
            payload = getattr(point, "payload", None) or {}
            pub_date_str = payload.get("publish_date") if isinstance(payload, dict) else None
            if not pub_date_str:
                # Missing publish_date -> skip this point
                continue

            # Support both string and datetime in payload
            if isinstance(pub_date_str, datetime):
                pub_date = pub_date_str.date()
            else:
                pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()

            # Safely get vector (point.vector or payload-stored vector)
            vec = getattr(point, "vector", None) or payload.get("vector")
            if vec is None:
                continue

            vec = np.asarray(vec, dtype=float)
            if vec.size != 768:
                # Unexpected embedding size -> skip
                continue

            days_old = (target_date_obj - pub_date).days
            weight = 0.8 ** max(0, days_old)

            weighted_sum += vec * weight
            total_weight += weight
        except Exception:
            # Any parsing error -> skip this point
            continue
            
    if total_weight == 0: return np.zeros(3)
    
    avg_vec = weighted_sum / total_weight
    try:
        return pca_reducer.transform(avg_vec.reshape(1, -1)).flatten()
    except Exception:
        # If PCA fails, return neutral vector
        return np.zeros(3)

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
        
        # B Matrix maps inputs to State Changes
        # Inputs: [News1, News2, News3, CurTemp, CurRain, FutMaxT, FutPoP]
        self.B = np.zeros((2, 7))
        
        # TUNING: Define how inputs affect Momentum (Row 1)
        self.B[1, 0:3] = 0.6   # News sentiment drives trend
        self.B[1, 4]   = -0.5  # Current rain kills momentum
        self.B[1, 6]   = -0.2  # Forecast rain dampens momentum
        
        # Measurement noise (R) – How much we trust actual sales data
        self.R = np.array([[100.]])  # Adjust based on sensor accuracy
        
        # Process noise (Q) – How much the system can change unpredictably
        self.Q = np.array([[10., 0.], [0., 5.]])

    def predict(self, u_vector):
        u = u_vector.reshape(-1, 1)
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement):
        """
        Correct state using actual sales measurement
        measurement: scalar sales value (e.g., today's actual sales)
        """
        z = np.array([[measurement]])
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T / S
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
        return self.x

    def save_state(self, filepath=STATE_FILE):
        """Save filter state to JSON for next iteration"""
        state_dict = {
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, "w") as f:
            json.dump(state_dict, f, indent=2)
        print(f"✅ State saved to {filepath}")

    def load_state(self, filepath=STATE_FILE):
        """Load previous filter state if available"""
        if not os.path.exists(filepath):
            print(f"ℹ️  No previous state found. Initializing fresh.")
            return False
        
        try:
            with open(filepath, "r") as f:
                state_dict = json.load(f)
            self.x = np.array(state_dict["x"])
            self.P = np.array(state_dict["P"])
            print(f"✅ State loaded from {filepath} (saved at {state_dict['timestamp']})")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load state: {e}. Initializing fresh.")
            return False

# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Retail Prediction Engine...\n")
    
    # 1. FIND LATEST FILES
    cur_weather_file = get_latest_file(DIR_CURRENT_WEATHER)
    fcst_weather_file = get_latest_file(DIR_FORECAST_WEATHER)
    
    if not cur_weather_file or not fcst_weather_file:
        print("❌ Error: Missing weather files in subfolders.")
        print(f"Checked: {DIR_CURRENT_WEATHER} and {DIR_FORECAST_WEATHER}")
        exit()

    # 2. LOAD & NORMALIZE DATA
    u_obs, obs_time = load_observation(cur_weather_file)
    u_fcst = load_forecast(fcst_weather_file, obs_time)
    u_news = get_weekly_news_context(obs_time)
    
    # 3. FUSE INPUTS
    u_full = np.hstack([u_news, u_obs, u_fcst])
    
    print("\n📊 FUSED INPUT VECTOR (u):")
    print(f"   [News]     Sentiment Factors: {np.round(u_news, 2)}")
    print(f"   [Current]  Temp: {u_obs[0]:.2f}, Rain: {u_obs[1]:.2f}")
    print(f"   [Forecast] Next Day MaxT: {u_fcst[0]:.2f}, PoP: {u_fcst[1]:.2f}")
    
    # 4. INITIALIZE KALMAN FILTER (Load previous state if exists)
    kf = RetailKalmanFilter()
    kf.load_state()  # ← KEY: Load previous state or initialize fresh
    
    # 5. PREDICT
    prediction = kf.predict(u_full)
    
    print("\n" + "="*50)
    print(f"🔮 PREDICTION RESULT (Based on data at {obs_time})")
    print("="*50)
    print(f"📈 Predicted Sales Volume: {prediction[0,0]:.2f}")
    print(f"🚀 Underlying Momentum:    {prediction[1,0]:.4f}")
    
    # 6. UPDATE with actual sales (if available)
    # Try to fetch from a data source (e.g., POS system, API, file)
    actual_sales = load_actual_sales_from_csv()  # TODO: Replace with real data source
    if actual_sales is not None:
        updated_prediction = kf.update(actual_sales)
    
        print(f"\n📊 Actual Sales Measurement: {actual_sales}")
        print(f"📈 Corrected Sales Prediction: {updated_prediction[0,0]:.2f}")
        print(f"🚀 Corrected Momentum:      {updated_prediction[1,0]:.4f}")
    
    # 7. SAVE STATE FOR NEXT RUN ← KEY: Persist the state
    kf.save_state()
    
    print("="*50)