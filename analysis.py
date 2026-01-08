import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable


# must have train.csv locally to run
# https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data?select=test.zip


# Set aesthetic style
sns.set_style("whitegrid")

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)
def print_split_table(data, title):
    """
    Helper function to print a wide dataframe in two parts using PrettyTable.
    It repeats the index column in the second part for readability.
    """
    # Reset index to make the row labels (like 'mean', 'std' or row numbers) a real column
    data_reset = data.reset_index()
    
    # Convert all columns to string to avoid rounding issues in display
    columns = data_reset.columns.tolist()
    n_cols = len(columns)
    
    # Determine split point
    mid = (n_cols // 2) + 1
    
    # Define column groups
    # Part 1: First half
    cols1 = columns[:mid]
    # Part 2: First column (index) + Second half
    cols2 = [columns[0]] + columns[mid:]
    
    print(f"\n>>> {title} [PART 1 of 2]")
    t1 = PrettyTable()
    t1.field_names = cols1
    for _, row in data_reset[cols1].iterrows():
        t1.add_row(row.tolist())
    print(t1)
    
    print(f"\n>>> {title} [PART 2 of 2]")
    t2 = PrettyTable()
    t2.field_names = cols2
    for _, row in data_reset[cols2].iterrows():
        t2.add_row(row.tolist())
    print(t2)
# ==========================================
# 1. LOAD DATA & INITIAL SETUP
# ==========================================
print_header("INITIALIZING DATASET")
df = pd.read_csv('train.csv', nrows=100000)


print(f"Dataset loaded successfully. Shape: {df.shape}")

# ==========================================
# REQUIREMENT 7: Pre-processing dataset
# ==========================================
print_header("7. PRE-PROCESSING DATASET")

# 1. Handling Missing Values
missing_before = df.isnull().sum().sum()
print(f"Total missing values before cleaning: {missing_before}")
df.dropna(inplace=True)
missing_after = df.isnull().sum().sum()
print(f"Total missing values after cleaning: {missing_after}")

# 2. Basic Conversions
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y': 1, 'N': 0})
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# 3. Feature Engineering: Temporal
df['month'] = df['pickup_datetime'].dt.month
df['day_of_week'] = df['pickup_datetime'].dt.day_name()
df['hour_of_day'] = df['pickup_datetime'].dt.hour
df['day_of_year'] = df['pickup_datetime'].dt.dayofyear

# Sort days for ordered plotting
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=days_order, ordered=True)
print("Added Temporal Features: month, day_of_week, hour_of_day, day_of_year")

# 4. Feature Engineering: Geospatial (Haversine Distance)
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

df['distance_km'] = haversine_np(
    df['pickup_longitude'], df['pickup_latitude'],
    df['dropoff_longitude'], df['dropoff_latitude']
)

# 5. Feature Engineering: Distance from Center (Restored from your code)
# Times Square coordinates
ts_lat, ts_lon = 40.7580, -73.9855

def get_dist_from_center_np(lon1, lat1):
    # Vectorized calculation for distance from Times Square
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(ts_lon), np.radians(ts_lat)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c

df['dist_from_center'] = get_dist_from_center_np(df['pickup_longitude'], df['pickup_latitude'])
print("Added Geospatial Features: distance_km, dist_from_center")

print("\n--- FORMATTED DATA TABLES ---")
# Print Head in 2 parts
print_split_table(df.head(), "First 5 Observations of Cleaned Data")

# Print Statistics in 2 parts
print_split_table(df.describe().round(2), "Statistics of Cleaned Dataset")
# ==========================================
# REQUIREMENT 11: Data Transformation
# ==========================================
print_header("11. DATA TRANSFORMATION")

# Method: Log Transformation
df['log_trip_duration'] = np.log1p(df['trip_duration'])
print("Applied Natural Log Transformation (log1p) to 'trip_duration'.")
print(f"Skewness before: {df['trip_duration'].skew():.2f}")
print(f"Skewness after:  {df['log_trip_duration'].skew():.2f}")

# ==========================================
# REQUIREMENT 8: Outlier detection & removal
# ==========================================
print_header("8. OUTLIER DETECTION & REMOVAL")

# ==========================================
# REQUIREMENT 8: Outlier detection & removal
# ==========================================
print_header("8. OUTLIER DETECTION & REMOVAL")

# 1. IQR for Log Duration
Q1_dur = df['log_trip_duration'].quantile(0.25)
Q3_dur = df['log_trip_duration'].quantile(0.75)
IQR_dur = Q3_dur - Q1_dur
lower_dur = Q1_dur - 1.5 * IQR_dur
upper_dur = Q3_dur + 1.5 * IQR_dur

# 2. IQR for Distance
Q1_dist = df['distance_km'].quantile(0.25)
Q3_dist = df['distance_km'].quantile(0.75)
IQR_dist = Q3_dist - Q1_dist
lower_dist = Q1_dist - 1.5 * IQR_dist
upper_dist = Q3_dist + 1.5 * IQR_dist

# Combine Filters
filtered_entries = (
    (df['log_trip_duration'] >= lower_dur) & (df['log_trip_duration'] <= upper_dur) &
    (df['distance_km'] >= lower_dist) & (df['distance_km'] <= upper_dist)
)

rows_before = df.shape[0]
df_clean = df[filtered_entries].copy()
rows_after = df_clean.shape[0]

print(f"Method Used: Interquartile Range (IQR).")
print(f"Applying filter to BOTH 'log_trip_duration' and 'distance_km'.")
print(f"Duration Bounds (Log): [{lower_dur:.2f}, {upper_dur:.2f}]")
print(f"Distance Bounds (km):  [{lower_dist:.2f}, {upper_dist:.2f}]")
print(f"Rows removed: {rows_before - rows_after}")
print(f"Percentage of data removed: {((rows_before - rows_after)/rows_before)*100:.2f}%")

# 
# ==========================================
# REQUIREMENT 10: Normality test
# ==========================================
print_header("10. NORMALITY TEST")

# Shapiro-Wilk Test is valid for N < 5000, so we sample.
alpha = 0.05
sample_size = 2000

print(f"--- 1. Testing Raw 'trip_duration' ---")
sample_raw = df_clean['trip_duration'].sample(sample_size, random_state=42)
stat_raw, p_raw = stats.shapiro(sample_raw)
print(f"Statistic={stat_raw:.4f}, p-value={p_raw:.4g}")
if p_raw > alpha:
    print("Observation: Raw Sample looks Gaussian (fail to reject H0).")
else:
    print("Observation: Raw Sample does not look Gaussian (reject H0).")

print(f"\n--- 2. Testing Transformed 'log_trip_duration' ---")
sample_log = df_clean['log_trip_duration'].sample(sample_size, random_state=42)
stat_log, p_log = stats.shapiro(sample_log)
print(f"Statistic={stat_log:.4f}, p-value={p_log:.4g}")
if p_log > alpha:
    print("Observation: Log Sample looks Gaussian (fail to reject H0).")
else:
    print("Observation: Log Sample does not look Gaussian (reject H0).")

# Visualization: Separate Plots for Report Clarity

# 1. Histogram Raw
plt.figure(figsize=(8, 6))
sns.histplot(df_clean['trip_duration'], kde=True, bins=50, color='salmon')
plt.title("Distribution of Raw Trip Duration")
plt.xlabel("Trip Duration (seconds)")
plt.show()

# 2. Q-Q Plot Raw
plt.figure(figsize=(8, 6))
stats.probplot(df_clean['trip_duration'], dist="norm", plot=plt)
plt.title("Q-Q Plot (Raw Trip Duration)")
plt.show()

# 3. Histogram Log
plt.figure(figsize=(8, 6))
sns.histplot(df_clean['log_trip_duration'], kde=True, bins=50, color='skyblue')
plt.title("Distribution of Log Trip Duration")
plt.xlabel("Log(Trip Duration)")
plt.show()

# 4. Q-Q Plot Log
plt.figure(figsize=(8, 6))
stats.probplot(df_clean['log_trip_duration'], dist="norm", plot=plt)
plt.title("Q-Q Plot (Log Trip Duration)")
plt.show()

# ==========================================
# REQUIREMENT 9: Principal Component Analysis (PCA)
# ==========================================
print_header("9. PRINCIPAL COMPONENT ANALYSIS (PCA)")

# Included 'dist_from_center' in PCA features
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'distance_km', 'dist_from_center']
x = df_clean[features].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=len(features))
principalComponents = pca.fit_transform(x)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Singular Values:", pca.singular_values_)
condition_number = pca.singular_values_[0] / pca.singular_values_[-1]
print(f"Condition Number: {condition_number:.2f}")

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(features)+1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('PCA Explained Variance (Cumulative)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# ==========================================
# REQUIREMENT 12: Heatmap & Pearson Correlation
# ==========================================
print_header("12. HEATMAP & PEARSON CORRELATION")

corr_cols = [
    'trip_duration', 
    'passenger_count', 
    'distance_km', 
    'dist_from_center', 
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude',
    'store_and_fwd_flag', 
    'month', 
    'day_of_year', 
    'hour_of_day'
]

# Calculate correlation matrix
corr_matrix = df_clean[corr_cols].corr(method='pearson')

print("Pearson Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.2,
    annot_kws={"size": 7}
)

plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=7)  # smaller y labels

plt.tight_layout(pad=0.3)
plt.show()

# Scatter Plot Matrix (Sample)
print("Generating Scatter Plot Matrix (Continuous Numerical Variables)...")

# Select only continuous variables (excluding discrete counts, time units, and flags)
scatter_cols = [
    'trip_duration', 
    'distance_km', 
    'dist_from_center', 
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude'
]

# Sample size 500 as requested
sample_for_plot = df_clean[scatter_cols].sample(500, random_state=42)

# pairplot with some transparency (alpha) to see overlapping points
sns.pairplot(sample_for_plot, diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle("Scatter Plot Matrix (Sample N=500)", y=1.02)
plt.tight_layout()
plt.show()

# ==========================================
# REQUIREMENT 13: Statistics & Multivariate KDE
# ==========================================
print_header("13. STATISTICS & MULTIVARIATE KDE")

# 1. Confidence Interval for Mean Trip Duration
# We calculate the 95% CI for the mean using the t-distribution
sample_mean = df_clean['trip_duration'].mean()
sample_std = df_clean['trip_duration'].std()
n_samples = len(df_clean)
confidence_level = 0.95
degrees_freedom = n_samples - 1

# Standard Error of the Mean
sem = sample_std / np.sqrt(n_samples)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sem)

print(f"\n--- Statistical Tool 1: Confidence Interval ---")
print(f"Mean Trip Duration: {sample_mean:.2f} seconds")
print(f"95% Confidence Interval: {confidence_interval}")
print(f"Observation: We are 95% confident the true population mean lies between {confidence_interval[0]:.2f} and {confidence_interval[1]:.2f} seconds.")

# 2. Independent T-Test (Hypothesis Testing)
# Hypothesis: Is there a significant difference in duration between Vendor 1 and Vendor 2?
print(f"\n--- Statistical Tool 2: Two-Sample T-Test ---")
vendor1 = df_clean[df_clean['vendor_id'] == 1]['trip_duration']
vendor2 = df_clean[df_clean['vendor_id'] == 2]['trip_duration']

# Only run if we have both vendors
if len(vendor1) > 0 and len(vendor2) > 0:
    t_stat, p_val = stats.ttest_ind(vendor1, vendor2, equal_var=False) # Welch's t-test
    print(f"Comparison: Vendor 1 vs Vendor 2 Trip Durations")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4g}")
    if p_val < 0.05:
        print("Result: Reject Null Hypothesis. Significant difference exists between vendors.")
    else:
        print("Result: Fail to Reject Null. No significant difference found.")

# 3. Multivariate KDE Plots
print("\n--- Generating Multivariate KDE Plots ---")

# Plot A: Spatial Density (Geographic)
plt.figure(figsize=(10, 8))
plot_data = df_clean[
    (df_clean.pickup_longitude > -74.05) & (df_clean.pickup_longitude < -73.75) &
    (df_clean.pickup_latitude > 40.60) & (df_clean.pickup_latitude < 40.90)
]
if len(plot_data) > 5000: plot_data = plot_data.sample(5000, random_state=42)

sns.kdeplot(
    x=plot_data['pickup_longitude'], 
    y=plot_data['pickup_latitude'], 
    cmap="Reds", 
    fill=True, 
    bw_adjust=0.5
)
plt.title("Spatial KDE: Pickup Density (Manhattan)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Plot B: Distance vs Duration Density
plt.figure(figsize=(10, 8))
sns.kdeplot(
    x=plot_data['distance_km'], 
    y=plot_data['log_trip_duration'], 
    cmap="Blues", 
    fill=True,
    thresh=0.05,
    levels=15
)
plt.title("Bivariate KDE: Distance vs. Time")
plt.xlabel("Distance (km)")
plt.ylabel("Log(Trip Duration)")
plt.show()

df_clean.to_csv('cleaned_taxi_data.csv', index=False)

print("\nANALYSIS COMPLETE.")