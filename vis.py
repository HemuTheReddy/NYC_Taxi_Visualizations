import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting


# MUST RUN analysis.py FIRST to generate 'cleaned_taxi_data.csv'

# Set aesthetic style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

# ==========================================
# 1. LOAD DATA
# ==========================================
print_header("LOADING CLEANED DATASET")
try:
    df = pd.read_csv('cleaned_taxi_data.csv')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Ensure categorical order for days of week
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=days_order, ordered=True)
    
except FileNotFoundError:
    print("Error: 'cleaned_taxi_data.csv' not found.")
    print("Please run the previous analysis script to generate the cleaned data file.")
    exit()

# Create a sample for computationally expensive plots (Scatter/KDE)
df_sample = df.sample(2000, random_state=42)

# ==========================================
# 2. LINE PLOT
# ==========================================
print_header("GENERATING PLOT 1: Line Plot (Hourly Trend)")
plt.figure()
avg_duration_hour = df.groupby('hour_of_day')['trip_duration'].mean()
plt.plot(avg_duration_hour.index, avg_duration_hour.values, marker='o', color='#1f77b4', linewidth=3)
plt.title('Hourly Trend: Average Trip Duration by Hour of Day', fontsize=16)
plt.xlabel('Hour of Day (0-23)', fontsize=12)
plt.ylabel('Average Duration (seconds)', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ==========================================
# 3. BAR PLOTS (GROUPED & STACKED)
# ==========================================
print_header("GENERATING PLOT 2 & 3: Bar Plots")

# Grouped
plt.figure()
sns.barplot(x='hour_of_day', y='trip_duration', hue='vendor_id', data=df, ci=None, palette='muted')
plt.title('Grouped Bar Plot: Average Trip Duration by Hour & Vendor', fontsize=16)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Avg Duration (sec)', fontsize=12)
plt.legend(title='Vendor ID')
plt.show()

# Stacked
plt.figure()
ct = pd.crosstab(df['day_of_week'], df['vendor_id'])
ct.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], figsize=(12, 7))
plt.title('Stacked Bar Plot: Trip Count by Day of Week & Vendor', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Trip Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Vendor ID')
plt.show()

# ==========================================
# 4. COUNT PLOT
# ==========================================
print_header("GENERATING PLOT 4: Count Plot")
plt.figure()
ax = sns.countplot(x='day_of_week', data=df, palette='coolwarm')
plt.title('Count Plot: Frequency of Trips by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Total Trips', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, color='black')
plt.show()

# ==========================================
# 5. PIE CHART
# ==========================================
print_header("GENERATING PLOT 5: Pie Chart")
plt.figure(figsize=(8, 8))
vendor_counts = df['vendor_id'].value_counts()
plt.pie(vendor_counts, labels=[f'Vendor {v}' for v in vendor_counts.index], 
        autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'], explode=(0.05, 0))
plt.title('Pie Chart: Market Share by Vendor ID', fontsize=16)
plt.show()

# ==========================================
# 6. DIST PLOT
# ==========================================
print_header("GENERATING PLOT 6: Dist Plot (Distance)")
plt.figure()
subset_dist = df[df['distance_km'] < 30]['distance_km']
sns.histplot(subset_dist, kde=True, color='teal', alpha=0.6, line_kws={'linewidth': 2})
plt.title('Dist Plot: Distribution of Trip Distances (Truncated at 30km)', fontsize=16)
plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# ==========================================
# 7. HISTOGRAM WITH KDE
# ==========================================
print_header("GENERATING PLOT 7: Histogram with KDE")
plt.figure()
sns.histplot(df['log_trip_duration'], kde=True, color='purple', alpha=0.6, line_kws={'linewidth': 2})
plt.title('Histogram with KDE: Log-Transformed Trip Duration', fontsize=16)
plt.xlabel('Log(Trip Duration)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# ==========================================
# 8. QQ-PLOT
# ==========================================
print_header("GENERATING PLOT 8: QQ-Plot")
plt.figure()
# Using Raw Trip Duration to demonstrate non-normality
stats.probplot(df['trip_duration'], dist="norm", plot=plt)
plt.title('QQ-Plot: Raw Trip Duration vs Normal Distribution', fontsize=16)
plt.xlabel('Theoretical Quantiles', fontsize=12)
plt.ylabel('Ordered Values (Trip Duration)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ==========================================
# 9. STYLED KDE PLOT
# ==========================================
print_header("GENERATING PLOT 9: Styled KDE Plot")
plt.figure()
# Requirement: fill, alpha=0.6, palette, linewidth
sns.kdeplot(data=df_sample, x='log_trip_duration', hue='vendor_id', 
            fill=True, alpha=0.6, linewidth=3, palette='crest')
plt.title('Styled KDE Plot: Log Duration Density by Vendor', fontsize=16)
plt.xlabel('Log(Trip Duration)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.show()

# ==========================================
# 10. REGRESSION PLOT (SCATTER + LINE)
# ==========================================
print_header("GENERATING PLOT 10: Regression Plot")
plt.figure()
# Using sample to avoid overplotting
sns.regplot(data=df_sample, x='distance_km', y='trip_duration', 
            scatter_kws={'alpha':0.5, 'color':'#1f77b4'}, 
            line_kws={'color':'red', 'linewidth':2})
plt.title('Regression Plot: Trip Duration vs. Distance (Sample N=2000)', fontsize=16)
plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Trip Duration (seconds)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 11. MULTIVARIATE BOX PLOT
# ==========================================
print_header("GENERATING PLOT 11: Multivariate Box Plot")
plt.figure()
# Using log duration to make the box plot readable (hiding extreme outliers)
sns.boxplot(x='day_of_week', y='log_trip_duration', data=df, palette='Set3')
plt.title('Multivariate Box Plot: Log Trip Duration by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Log(Trip Duration)', fontsize=12)
plt.show()

# ==========================================
# 12. MULTIVARIATE BOXEN PLOT
# ==========================================
print_header("GENERATING PLOT 12: Multivariate Boxen Plot")
plt.figure()
sns.boxenplot(x='passenger_count', y='trip_duration', data=df, palette="spring")
plt.title('Multivariate Boxen Plot: Trip Duration by Passenger Count', fontsize=16)
plt.xlabel('Passenger Count', fontsize=12)
plt.ylabel('Trip Duration (seconds)', fontsize=12)
plt.ylim(0, 5000) # Zooming in to relevant range
plt.show()

# ==========================================
# 13. AREA PLOT
# ==========================================
print_header("GENERATING PLOT 13: Area Plot")
plt.figure()
# Sort sample by distance for cumulative area effect
sorted_dist = df_sample.sort_values('distance_km')
plt.fill_between(sorted_dist['distance_km'], sorted_dist['trip_duration'], color="skyblue", alpha=0.4)
plt.plot(sorted_dist['distance_km'], sorted_dist['trip_duration'], color="Slateblue", alpha=0.6)
plt.title('Area Plot: Cumulative Duration Trend by Distance', fontsize=16)
plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Trip Duration', fontsize=12)
plt.show()

# ==========================================
# 14. VIOLIN PLOT
# ==========================================
print_header("GENERATING PLOT 14: Violin Plot")
plt.figure()
sns.violinplot(x='day_of_week', y='log_trip_duration', hue='vendor_id', data=df_sample, split=True, palette='Pastel1')
plt.title('Violin Plot: Duration Distribution by Day and Vendor', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Log(Trip Duration)', fontsize=12)
plt.show()

# ==========================================
# 15. JOINT PLOT (KDE + SCATTER)
# ==========================================
print_header("GENERATING PLOT 15: Joint Plot")
# JointGrid for custom scatter + kde layers
g = sns.JointGrid(data=df_sample, x="distance_km", y="log_trip_duration")
g.plot_joint(sns.scatterplot, s=100, alpha=0.5, color="purple")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.histplot, kde=True, color="purple")
plt.tight_layout()
plt.suptitle('Joint Plot: Distance vs Log Duration (Scatter + KDE)', y=1.02, fontsize=16)
plt.show()

# ==========================================
# 16. RUG PLOT
# ==========================================
print_header("GENERATING PLOT 16: Rug Plot")
plt.figure()
# Combine KDE with Rugplot for detailed density visualization
sns.kdeplot(data=df_sample, x='log_trip_duration', fill=True, color='orange')
sns.rugplot(data=df_sample, x='log_trip_duration', height=0.1, color='black', alpha=0.5)
plt.title('Rug Plot with KDE: Log Trip Duration Density', fontsize=16)
plt.xlabel('Log(Trip Duration)', fontsize=12)
plt.show()

# ==========================================
# 17. 3D PLOT
# ==========================================
print_header("GENERATING PLOT 17: 3D Scatter Plot")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_sample['pickup_longitude'], df_sample['pickup_latitude'], df_sample['trip_duration'], c='r', marker='o', alpha=0.3)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Duration')
ax.set_title('3D Plot: Spatial Origin vs Duration', fontsize=16)
plt.show()

# ==========================================
# 18. CONTOUR PLOT
# ==========================================
print_header("GENERATING PLOT 18: Contour Plot")
plt.figure()
# Focusing on Manhattan coordinates
sns.kdeplot(x='pickup_longitude', y='pickup_latitude', data=df_sample, 
            fill=False, color='blue', thresh=0.05, levels=10)
plt.title('Contour Plot: Pickup Density (Manhattan)', fontsize=16)
plt.xlim(-74.05, -73.75) 
plt.ylim(40.6, 40.9)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.show()

# ==========================================
# 19. CLUSTER MAP
# ==========================================
print_header("GENERATING PLOT 19: Cluster Map")
# Select numerical columns for correlation clustering
cols = ['trip_duration', 'distance_km', 'pickup_latitude', 'pickup_longitude', 'passenger_count']
g = sns.clustermap(
    df_sample[cols].corr(),
    cmap='viridis',
    annot=True,
    figsize=(8, 8)
)

# Add title to the clustermap figure
g.fig.suptitle('Cluster Map: Feature Correlations', y=1.02, fontsize=16)

# Adjust layout so title is not cut off
g.fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()

# ==========================================
# 20. HEXBIN PLOT
# ==========================================
print_header("GENERATING PLOT 20: Hexbin Plot")
plt.figure()
# Limit to < 20km distance to make hexbins readable
hex_data = df[df['distance_km'] < 20]
plt.hexbin(hex_data['distance_km'], hex_data['trip_duration'], gridsize=20, cmap='Blues', mincnt=1)
cb = plt.colorbar(label='Count')
plt.title('Hexbin Plot: Distance vs Duration Density', fontsize=16)
plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Duration (sec)', fontsize=12)
plt.show()

# ==========================================
# 21. STRIP PLOT
# ==========================================
print_header("GENERATING PLOT 21: Strip Plot")
plt.figure()
sns.stripplot(x='vendor_id', y='distance_km', data=df_sample, jitter=True, alpha=0.5, palette='dark')
plt.title('Strip Plot: Trip Distance Distribution by Vendor', fontsize=16)
plt.xlabel('Vendor ID', fontsize=12)
plt.ylabel('Distance (km)', fontsize=12)
plt.show()

# ==========================================
# 22. SWARM PLOT
# ==========================================
print_header("GENERATING PLOT 22: Swarm Plot")
plt.figure()
# Swarm plots do not scale well, using very small sample (N=200)
sns.swarmplot(x='vendor_id', y='distance_km', data=df_sample.iloc[:200], palette='Set2')
plt.title('Swarm Plot: Distance by Vendor (Subset N=200)', fontsize=16)
plt.xlabel('Vendor ID', fontsize=12)
plt.ylabel('Distance (km)', fontsize=12)
plt.show()

print("\nAll 22 Visualizations Complete.")