import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def apply_auto_clustering(df):
    """Dynamically picks the best number of clusters using Silhouette Scores."""
    monthly_profile = df.pivot_table(
        index=['Year', 'Month'], columns='Category', values='Amount', aggfunc='sum'
    ).fillna(0)
    
    features = monthly_profile.drop(columns=['Salary', 'Income'], errors='ignore')
    if len(features) < 3: 
        monthly_profile['Cluster_Label'] = 0
        return pd.merge(df, monthly_profile.reset_index()[['Year', 'Month', 'Cluster_Label']], on=['Year', 'Month'], how='left')
        
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    best_k = 2
    best_score = -1
    
    # Mathematically find the optimal cluster size
    for k in range(2, min(6, len(features))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            
    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    monthly_profile['Cluster_Label'] = final_km.fit_predict(scaled)
    
    df = pd.merge(df, monthly_profile.reset_index()[['Year', 'Month', 'Cluster_Label']], on=['Year', 'Month'], how='left')
    return df