"""
Health Region Clustering Analysis

Advanced clustering techniques for identifying health patterns and regions.
Implements multiple clustering algorithms with sophisticated evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, 
    SpectralClustering, GaussianMixture
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, 
    davies_bouldin_score, adjusted_rand_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
from yellowbrick.cluster import (
    KElbowVisualizer, SilhouetteVisualizer, 
    InterclusterDistance
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class HealthClusterAnalyzer:
    """
    Advanced clustering analysis for health regions and indicators.
    
    Provides comprehensive clustering capabilities including:
    - Multiple clustering algorithms
    - Automated parameter tuning
    - Cluster validation and interpretation
    - Interactive visualizations
    """
    
    def __init__(self, data: pd.DataFrame, feature_columns: list = None):
        """
        Initialize the clustering analyzer.
        
        Args:
            data: DataFrame containing health indicators
            feature_columns: List of columns to use for clustering
        """
        self.data = data.copy()
        self.feature_columns = feature_columns or self._select_numeric_features()
        self.scaled_data = None
        self.cluster_results = {}
        self.evaluation_metrics = {}
        
    def _select_numeric_features(self) -> list:
        """Select numeric features for clustering."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID columns and obvious non-feature columns
        exclude_patterns = ['id', 'code', 'year', 'count', 'total']
        return [col for col in numeric_cols 
                if not any(pattern in col.lower() for pattern in exclude_patterns)]
    
    def preprocess_data(self, scaler_type: str = 'standard', 
                       handle_missing: str = 'drop') -> np.ndarray:
        """
        Preprocess data for clustering.
        
        Args:
            scaler_type: 'standard', 'robust', or 'minmax'
            handle_missing: 'drop', 'mean', or 'median'
            
        Returns:
            Scaled feature matrix
        """
        # Handle missing values
        feature_data = self.data[self.feature_columns].copy()
        
        if handle_missing == 'drop':
            feature_data = feature_data.dropna()
        elif handle_missing == 'mean':
            feature_data = feature_data.fillna(feature_data.mean())
        elif handle_missing == 'median':
            feature_data = feature_data.fillna(feature_data.median())
        
        # Scale features
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        self.scaled_data = scaler.fit_transform(feature_data)
        self.scaler = scaler
        self.valid_indices = feature_data.index
        
        return self.scaled_data
    
    def find_optimal_clusters(self, max_clusters: int = 15, 
                            methods: list = None) -> dict:
        """
        Find optimal number of clusters using multiple methods.
        
        Args:
            max_clusters: Maximum number of clusters to test
            methods: List of methods to use ['elbow', 'silhouette', 'gap']
            
        Returns:
            Dictionary with optimal cluster numbers for each method
        """
        if self.scaled_data is None:
            self.preprocess_data()
        
        methods = methods or ['elbow', 'silhouette', 'calinski_harabasz']
        results = {}
        
        # Elbow method
        if 'elbow' in methods:
            inertias = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(self.scaled_data)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            differences = np.diff(inertias)
            second_differences = np.diff(differences)
            elbow_point = np.argmax(second_differences) + 2
            results['elbow'] = elbow_point
        
        # Silhouette method
        if 'silhouette' in methods:
            silhouette_scores = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(self.scaled_data)
                silhouette_avg = silhouette_score(self.scaled_data, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            results['silhouette'] = optimal_k
        
        # Calinski-Harabasz method
        if 'calinski_harabasz' in methods:
            ch_scores = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(self.scaled_data)
                ch_score = calinski_harabasz_score(self.scaled_data, cluster_labels)
                ch_scores.append(ch_score)
            
            optimal_k = k_range[np.argmax(ch_scores)]
            results['calinski_harabasz'] = optimal_k
        
        return results
    
    def perform_clustering(self, algorithms: dict = None) -> dict:
        """
        Perform clustering using multiple algorithms.
        
        Args:
            algorithms: Dictionary of algorithm names and parameters
            
        Returns:
            Dictionary containing clustering results
        """
        if self.scaled_data is None:
            self.preprocess_data()
        
        if algorithms is None:
            # Find optimal number of clusters
            optimal_k = self.find_optimal_clusters()
            k = optimal_k.get('silhouette', 5)
            
            algorithms = {
                'kmeans': {'n_clusters': k, 'random_state': 42, 'n_init': 10},
                'hierarchical': {'n_clusters': k, 'linkage': 'ward'},
                'gaussian_mixture': {'n_components': k, 'random_state': 42},
                'spectral': {'n_clusters': k, 'random_state': 42},
                'dbscan': {'eps': 0.5, 'min_samples': 5},
                'hdbscan': {'min_cluster_size': 10}
            }
        
        results = {}
        
        for name, params in algorithms.items():
            try:
                if name == 'kmeans':
                    model = KMeans(**params)
                    labels = model.fit_predict(self.scaled_data)
                    
                elif name == 'hierarchical':
                    model = AgglomerativeClustering(**params)
                    labels = model.fit_predict(self.scaled_data)
                    
                elif name == 'gaussian_mixture':
                    model = GaussianMixture(**params)
                    labels = model.fit_predict(self.scaled_data)
                    
                elif name == 'spectral':
                    model = SpectralClustering(**params)
                    labels = model.fit_predict(self.scaled_data)
                    
                elif name == 'dbscan':
                    model = DBSCAN(**params)
                    labels = model.fit_predict(self.scaled_data)
                    
                elif name == 'hdbscan':
                    model = hdbscan.HDBSCAN(**params)
                    labels = model.fit_predict(self.scaled_data)
                
                # Store results
                results[name] = {
                    'model': model,
                    'labels': labels,
                    'n_clusters': len(np.unique(labels[labels >= 0])),
                    'n_noise': np.sum(labels == -1) if -1 in labels else 0
                }
                
                # Calculate evaluation metrics
                if len(np.unique(labels)) > 1:
                    valid_labels = labels[labels >= 0]
                    valid_data = self.scaled_data[labels >= 0]
                    
                    if len(valid_labels) > 0 and len(np.unique(valid_labels)) > 1:
                        results[name]['silhouette'] = silhouette_score(
                            valid_data, valid_labels
                        )
                        results[name]['calinski_harabasz'] = calinski_harabasz_score(
                            valid_data, valid_labels
                        )
                        results[name]['davies_bouldin'] = davies_bouldin_score(
                            valid_data, valid_labels
                        )
                
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                continue
        
        self.cluster_results = results
        return results
    
    def create_cluster_profiles(self, algorithm: str = 'kmeans') -> pd.DataFrame:
        """
        Create detailed profiles for each cluster.
        
        Args:
            algorithm: Clustering algorithm to use for profiling
            
        Returns:
            DataFrame with cluster profiles
        """
        if algorithm not in self.cluster_results:
            raise ValueError(f"Algorithm {algorithm} not found in results")
        
        labels = self.cluster_results[algorithm]['labels']
        
        # Create DataFrame with original data and cluster labels
        cluster_data = self.data.loc[self.valid_indices].copy()
        cluster_data['cluster'] = labels
        
        # Calculate cluster profiles
        profiles = []
        
        for cluster_id in sorted(cluster_data['cluster'].unique()):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_subset),
                'percentage': len(cluster_subset) / len(cluster_data) * 100
            }
            
            # Calculate statistics for each feature
            for col in self.feature_columns:
                if col in cluster_subset.columns:
                    profile[f'{col}_mean'] = cluster_subset[col].mean()
                    profile[f'{col}_std'] = cluster_subset[col].std()
                    profile[f'{col}_median'] = cluster_subset[col].median()
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def visualize_clusters(self, algorithm: str = 'kmeans', 
                          method: str = 'pca') -> go.Figure:
        """
        Create interactive cluster visualizations.
        
        Args:
            algorithm: Clustering algorithm to visualize
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            
        Returns:
            Plotly figure
        """
        if algorithm not in self.cluster_results:
            raise ValueError(f"Algorithm {algorithm} not found in results")
        
        labels = self.cluster_results[algorithm]['labels']
        
        # Perform dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.scaled_data)
            title_suffix = f"PCA (explained variance: {sum(reducer.explained_variance_ratio_):.2%})"
            
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedding = reducer.fit_transform(self.scaled_data)
            title_suffix = "t-SNE"
            
        elif method == 'umap':
            reducer = UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.scaled_data)
            title_suffix = "UMAP"
        
        # Create visualization
        fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            color=labels.astype(str),
            title=f"Health Region Clusters - {algorithm.title()} ({title_suffix})",
            labels={'x': f'{method.upper()} Component 1', 
                   'y': f'{method.upper()} Component 2',
                   'color': 'Cluster'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True,
            font=dict(size=12)
        )
        
        return fig
    
    def compare_algorithms(self) -> go.Figure:
        """
        Compare clustering algorithms using evaluation metrics.
        
        Returns:
            Plotly figure comparing algorithms
        """
        if not self.cluster_results:
            raise ValueError("No clustering results available")
        
        # Prepare comparison data
        algorithms = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        n_clusters = []
        
        for name, results in self.cluster_results.items():
            if 'silhouette' in results:
                algorithms.append(name.title())
                silhouette_scores.append(results.get('silhouette', 0))
                calinski_scores.append(results.get('calinski_harabasz', 0))
                davies_bouldin_scores.append(results.get('davies_bouldin', 0))
                n_clusters.append(results.get('n_clusters', 0))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Silhouette Score (Higher is Better)',
                          'Calinski-Harabasz Score (Higher is Better)',
                          'Davies-Bouldin Score (Lower is Better)',
                          'Number of Clusters'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add bar charts
        fig.add_trace(
            go.Bar(x=algorithms, y=silhouette_scores, name='Silhouette',
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=algorithms, y=calinski_scores, name='Calinski-Harabasz',
                  marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=algorithms, y=davies_bouldin_scores, name='Davies-Bouldin',
                  marker_color='lightcoral'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=algorithms, y=n_clusters, name='N Clusters',
                  marker_color='lightyellow'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Clustering Algorithm Comparison",
            showlegend=False,
            height=800,
            width=1000
        )
        
        return fig
    
    def generate_cluster_report(self, algorithm: str = 'kmeans') -> str:
        """
        Generate a comprehensive clustering analysis report.
        
        Args:
            algorithm: Clustering algorithm to report on
            
        Returns:
            Formatted report string
        """
        if algorithm not in self.cluster_results:
            raise ValueError(f"Algorithm {algorithm} not found in results")
        
        results = self.cluster_results[algorithm]
        profiles = self.create_cluster_profiles(algorithm)
        
        report = f"""
# Health Region Clustering Analysis Report
## Algorithm: {algorithm.title()}

### Summary Statistics
- **Number of Clusters:** {results['n_clusters']}
- **Number of Noise Points:** {results.get('n_noise', 0)}
- **Silhouette Score:** {results.get('silhouette', 'N/A'):.3f}
- **Calinski-Harabasz Score:** {results.get('calinski_harabasz', 'N/A'):.3f}
- **Davies-Bouldin Score:** {results.get('davies_bouldin', 'N/A'):.3f}

### Cluster Profiles
"""
        
        for _, profile in profiles.iterrows():
            report += f"""
#### Cluster {int(profile['cluster_id'])}
- **Size:** {int(profile['size'])} regions ({profile['percentage']:.1f}%)
- **Key Characteristics:**
"""
            
            # Find the most distinctive features for this cluster
            feature_scores = []
            for col in self.feature_columns:
                if f'{col}_mean' in profile:
                    mean_val = profile[f'{col}_mean']
                    overall_mean = self.data[col].mean()
                    if not np.isnan(mean_val) and not np.isnan(overall_mean):
                        score = abs(mean_val - overall_mean) / (overall_mean + 1e-8)
                        feature_scores.append((col, score, mean_val))
            
            # Sort by distinctiveness and show top features
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            for col, score, value in feature_scores[:5]:
                report += f"  - {col}: {value:.2f}\n"
        
        report += """
### Interpretation and Recommendations

The clustering analysis reveals distinct health patterns across regions. 
Each cluster represents regions with similar health profiles that may 
benefit from targeted interventions and policy approaches.

### Next Steps
1. Validate clusters with domain experts
2. Develop targeted health interventions for each cluster
3. Monitor cluster stability over time
4. Consider additional variables for refinement
"""
        
        return report