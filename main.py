# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set styling for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the dataset
df = pd.read_csv(r'C:\Users\Qc\Desktop\Viral Social Media Trends\Viral_Social_Media_Trends.csv')
# Display the first few rows to get a glimpse of the data
print(f"Data shape: {df.shape}")
df.head()
# Check data types and missing values
df.info()

# Display summary statistics
df.describe()

# Check for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage
})
missing_data[missing_data['Missing Values'] > 0]
# Create a function to plot categorical data distribution
def plot_categorical_distribution(data, column, title, palette="viridis"):
    plt.figure(figsize=(12, 6))
    count_plot = sns.countplot(x=column, data=data, palette=palette, order=data[column].value_counts().index)
    plt.title(f'Distribution of {title}', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    
  # Plot distribution of regions (top 15)
top_regions = df['Region'].value_counts().head(15).index
plt.figure(figsize=(14, 8))
sns.countplot(y='Region', data=df[df['Region'].isin(top_regions)], 
              order=df['Region'].value_counts().head(15).index,
              palette="viridis")
plt.title('Top 15 Regions by Post Count', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Region', fontsize=14)
plt.tight_layout()
#plt.show()
# Plot distribution of engagement levels
plot_categorical_distribution(df, 'Engagement_Level', 'Engagement Levels', palette="YlOrRd")
#plt.show()
# Plot distribution of trending hashtags (top 15)
plt.figure(figsize=(14, 8))
df['Hashtag'].value_counts().head(15).plot(kind='bar', colormap='viridis')
plt.title('Top 15 Trending Hashtags', fontsize=16)
plt.xlabel('Hashtag', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
# Create histograms for numerical variables
numerical_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(16, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
#plt.show()
# Create log-transformed histograms for better visualization of skewed data
plt.figure(figsize=(16, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i+1)
    if df[col].min() > 0:  # Ensure we don't try to take log of zero or negative values
        sns.histplot(np.log1p(df[col]), kde=True)
        plt.title(f'Log Distribution of {col}', fontsize=14)
        plt.xlabel(f'Log({col})', fontsize=12)
    else:
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
#plt.show()
# Compute the correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Numerical Variables', fontsize=16)
plt.tight_layout()
plt.show()
# Create scatterplots of correlated variables
plt.figure(figsize=(16, 12))
for i, x in enumerate(numerical_cols[:-1]):
    for j, y in enumerate(numerical_cols[i+1:], i+1):
        plt.subplot(2, 3, i+1)
        plt.scatter(df[x], df[y], alpha=0.5)
        plt.title(f'{x} vs {y}', fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        break  # Just show a few key plots
plt.tight_layout()
plt.show()
# Create a radar chart to compare platforms across all metrics
def radar_chart(data, categories, group_var, value_vars, title):
    # Number of variables
    N = len(value_vars)
    
    # We need to repeat the first value to close the circular graph
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Set the first axis to be on top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], value_vars, fontsize=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(color="grey")
    
    # Set colors for different groups
    cmap = plt.cm.get_cmap("viridis", len(data[group_var].unique()))
    
    # Plot each group
    for i, group in enumerate(data[group_var].unique()):
        group_data = data[data[group_var] == group]
        
        # Scale the values to fit on the same chart
        scaled_values = []
        for val_var in value_vars:
            # Scale between 0 and 1 using min-max scaling
            min_val = data[val_var].min()
            max_val = data[val_var].max()
            if max_val == min_val:
                scaled_values.append(0.5)  # If all values are the same
            else:
                scaled_value = (group_data[val_var].values[0] - min_val) / (max_val - min_val)
                scaled_values.append(scaled_value)
        
        # Add the first value again to close the circular graph
        values = scaled_values + [scaled_values[0]]
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=group, color=cmap(i))
        ax.fill(angles, values, alpha=0.1, color=cmap(i))
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, fontsize=16, y=1.08)
    
    plt.tight_layout()
    plt.show()
# Aggregate metrics by platform
platform_metrics = df.groupby('Platform')[numerical_cols].mean().reset_index()

# Create the radar chart
radar_chart(platform_metrics, 'Platform', 'Platform', numerical_cols, 'Platform Comparison Across Engagement Metrics')
# Number of posts by platform
platform_counts = df['Platform'].value_counts()

# Average engagement metrics by platform
platform_avg = df.groupby('Platform')[numerical_cols].mean()

# Top content types by average views
content_avg_views = df.groupby('Content_Type')['Views'].mean().sort_values(ascending=False).head(5)

# Top hashtags by average views
hashtag_avg_views = df.groupby('Hashtag')['Views'].mean().sort_values(ascending=False).head(5)

# Top regions by average views
region_avg_views = df.groupby('Region')['Views'].mean().sort_values(ascending=False).head(5)

# Correlation between metrics
correlation = df[numerical_cols].corr()
# Print summary
print("=== VIRAL SOCIAL MEDIA TRENDS - KEY INSIGHTS ===\n")

print("1. PLATFORM DISTRIBUTION:")
print(platform_counts)
print("\n2. AVERAGE ENGAGEMENT METRICS BY PLATFORM:")
print(platform_avg)
print("\n3. TOP 5 CONTENT TYPES BY AVERAGE VIEWS:")
print(content_avg_views)
print("\n4. TOP 5 HASHTAGS BY AVERAGE VIEWS:")
print(hashtag_avg_views)
print("\n5. TOP 5 REGIONS BY AVERAGE VIEWS:")
print(region_avg_views)
print("\n6. CORRELATION BETWEEN ENGAGEMENT METRICS:")
print(correlation)