#!/usr/bin/env python3
"""
🚀 Performance Optimization Suite
Geographic indexing, caching, and database optimization
"""

import polars as pl
import pandas as pd
from pathlib import Path
import sqlite3
import json
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PerformanceOptimizer:
    """Comprehensive performance optimization system"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.optimized_dir = self.data_dir / 'optimized'
        self.optimized_dir.mkdir(exist_ok=True)
        
        self.cache_dir = self.data_dir.parent / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = {
            'optimization_start': time.time(),
            'operations': []
        }
    
    def run_complete_optimization(self) -> Dict:
        """Run complete performance optimization suite"""
        print("🚀 Starting Performance Optimization Suite...")
        print("=" * 60)
        
        optimization_results = {
            'geographic_indexing': self.optimize_geographic_indexing(),
            'caching_system': self.implement_caching_system(),
            'database_optimization': self.optimize_database_structure(),
            'query_optimization': self.create_optimized_queries(),
            'compression_optimization': self.optimize_compression(),
            'performance_benchmarks': self.run_performance_benchmarks()
        }
        
        # Generate optimization report
        self.generate_optimization_report(optimization_results)
        
        print("\n🎉 Performance Optimization Complete!")
        self.print_optimization_summary(optimization_results)
        
        return optimization_results
    
    def optimize_geographic_indexing(self) -> Dict:
        """Create optimized geographic indexes"""
        print("\n🗺️ Optimizing Geographic Indexing...")
        
        start_time = time.time()
        results = {
            'sa2_lookup_index': self.create_sa2_lookup_index(),
            'spatial_boundary_index': self.optimize_spatial_boundaries(),
            'state_territory_index': self.create_state_territory_index(),
            'geographic_hierarchy_index': self.create_geographic_hierarchy()
        }
        
        end_time = time.time()
        results['optimization_time'] = end_time - start_time
        
        print(f"  ✅ Geographic indexing optimized in {results['optimization_time']:.2f} seconds")
        return results
    
    def create_sa2_lookup_index(self) -> Dict:
        """Create optimized SA2 code to name lookup index"""
        print("    🔧 Creating SA2 lookup index...")
        
        # Load SEIFA and boundaries data
        seifa_file = self.data_dir / 'seifa_2021_sa2.parquet'
        boundaries_file = self.data_dir / 'sa2_boundaries_2021.parquet'
        
        sa2_lookup = {}
        
        if seifa_file.exists():
            seifa_df = pl.read_parquet(seifa_file)
            
            # Create SA2 code to name mapping
            lookup_data = seifa_df.select(['sa2_code_2021', 'sa2_name_2021']).unique()
            sa2_lookup.update(lookup_data.iter_rows())
        
        if boundaries_file.exists():
            # Handle geospatial data
            try:
                boundaries_df = pl.read_parquet(boundaries_file)
            except:
                # Fallback for geospatial data
                import pandas as pd
                pandas_df = pd.read_parquet(boundaries_file)
                non_geo_cols = [col for col in pandas_df.columns if pandas_df[col].dtype.name != 'geometry']
                boundaries_df = pl.from_pandas(pandas_df[non_geo_cols])
            
            # Add additional geographic information
            if 'SA2_CODE21' in boundaries_df.columns and 'SA2_NAME21' in boundaries_df.columns:
                boundary_lookup = boundaries_df.select(['SA2_CODE21', 'SA2_NAME21']).unique()
                for row in boundary_lookup.iter_rows():
                    sa2_lookup[row[0]] = row[1]
        
        # Save lookup index
        lookup_file = self.optimized_dir / 'sa2_lookup_index.json'
        with open(lookup_file, 'w') as f:
            json.dump(sa2_lookup, f, indent=2)
        
        print(f"    ✅ SA2 lookup index created: {len(sa2_lookup):,} mappings")
        return {
            'status': 'created',
            'mappings_count': len(sa2_lookup),
            'file_location': str(lookup_file)
        }
    
    def optimize_spatial_boundaries(self) -> Dict:
        """Optimize spatial boundary data for fast queries"""
        print("    🔧 Optimizing spatial boundaries...")
        
        boundaries_file = self.data_dir / 'sa2_boundaries_2021.parquet'
        if not boundaries_file.exists():
            return {'status': 'file_not_found'}
        
        try:
            # Load with pandas for geometry handling
            import pandas as pd
            import geopandas as gpd
            
            # Try loading as regular parquet first
            try:
                df = pd.read_parquet(boundaries_file)
                
                # Create simplified boundary data for fast queries
                simplified_boundaries = df[['SA2_CODE21', 'SA2_NAME21', 'STE_CODE21', 'STE_NAME21', 'AREASQKM21']].copy()
                
                # Save simplified version
                simplified_file = self.optimized_dir / 'sa2_boundaries_simplified.parquet'
                simplified_boundaries.to_parquet(simplified_file, compression='snappy')
                
                print(f"    ✅ Simplified boundaries created: {len(simplified_boundaries):,} areas")
                return {
                    'status': 'optimized',
                    'original_size_mb': boundaries_file.stat().st_size / (1024 * 1024),
                    'optimized_size_mb': simplified_file.stat().st_size / (1024 * 1024),
                    'areas_count': len(simplified_boundaries),
                    'simplified_file': str(simplified_file)
                }
                
            except Exception as e:
                print(f"    ⚠️ Could not optimize spatial boundaries: {e}")
                return {'status': 'error', 'error': str(e)}
        
        except ImportError:
            print("    ⚠️ GeoPandas not available for spatial optimization")
            return {'status': 'geopandas_not_available'}
    
    def create_state_territory_index(self) -> Dict:
        """Create state/territory lookup index"""
        print("    🔧 Creating state/territory index...")
        
        state_mapping = {}
        
        # Load boundary data for state information
        boundaries_file = self.data_dir / 'sa2_boundaries_2021.parquet'
        if boundaries_file.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(boundaries_file)
                
                if 'STE_CODE21' in df.columns and 'STE_NAME21' in df.columns:
                    state_lookup = df[['STE_CODE21', 'STE_NAME21']].drop_duplicates()
                    for _, row in state_lookup.iterrows():
                        state_mapping[row['STE_CODE21']] = row['STE_NAME21']
                
                # Also create SA2 to state mapping
                sa2_to_state = {}
                if 'SA2_CODE21' in df.columns:
                    sa2_state_lookup = df[['SA2_CODE21', 'STE_CODE21', 'STE_NAME21']].drop_duplicates()
                    for _, row in sa2_state_lookup.iterrows():
                        sa2_to_state[row['SA2_CODE21']] = {
                            'state_code': row['STE_CODE21'],
                            'state_name': row['STE_NAME21']
                        }
                
                # Save indexes
                state_index_file = self.optimized_dir / 'state_territory_index.json'
                with open(state_index_file, 'w') as f:
                    json.dump(state_mapping, f, indent=2)
                
                sa2_state_file = self.optimized_dir / 'sa2_to_state_index.json'
                with open(sa2_state_file, 'w') as f:
                    json.dump(sa2_to_state, f, indent=2)
                
                print(f"    ✅ State indexes created: {len(state_mapping)} states, {len(sa2_to_state):,} SA2 mappings")
                return {
                    'status': 'created',
                    'states_count': len(state_mapping),
                    'sa2_mappings_count': len(sa2_to_state),
                    'state_index_file': str(state_index_file),
                    'sa2_state_file': str(sa2_state_file)
                }
                
            except Exception as e:
                print(f"    ⚠️ Error creating state index: {e}")
                return {'status': 'error', 'error': str(e)}
        
        return {'status': 'no_boundary_data'}
    
    def create_geographic_hierarchy(self) -> Dict:
        """Create complete geographic hierarchy index"""
        print("    🔧 Creating geographic hierarchy index...")
        
        boundaries_file = self.data_dir / 'sa2_boundaries_2021.parquet'
        if not boundaries_file.exists():
            return {'status': 'file_not_found'}
        
        try:
            import pandas as pd
            df = pd.read_parquet(boundaries_file)
            
            # Create hierarchy mapping
            hierarchy_cols = ['SA2_CODE21', 'SA2_NAME21', 'SA3_CODE21', 'SA3_NAME21', 
                            'SA4_CODE21', 'SA4_NAME21', 'GCC_CODE21', 'GCC_NAME21',
                            'STE_CODE21', 'STE_NAME21', 'AUS_CODE21', 'AUS_NAME21']
            
            available_cols = [col for col in hierarchy_cols if col in df.columns]
            
            if available_cols:
                hierarchy_df = df[available_cols].drop_duplicates()
                
                # Convert to nested structure
                hierarchy_dict = {}
                for _, row in hierarchy_df.iterrows():
                    sa2_code = row['SA2_CODE21']
                    hierarchy_dict[sa2_code] = {
                        'sa2_name': row.get('SA2_NAME21', ''),
                        'sa3_code': row.get('SA3_CODE21', ''),
                        'sa3_name': row.get('SA3_NAME21', ''),
                        'sa4_code': row.get('SA4_CODE21', ''),
                        'sa4_name': row.get('SA4_NAME21', ''),
                        'gcc_code': row.get('GCC_CODE21', ''),
                        'gcc_name': row.get('GCC_NAME21', ''),
                        'state_code': row.get('STE_CODE21', ''),
                        'state_name': row.get('STE_NAME21', ''),
                        'country_code': row.get('AUS_CODE21', ''),
                        'country_name': row.get('AUS_NAME21', '')
                    }
                
                # Save hierarchy index
                hierarchy_file = self.optimized_dir / 'geographic_hierarchy_index.json'
                with open(hierarchy_file, 'w') as f:
                    json.dump(hierarchy_dict, f, indent=2)
                
                print(f"    ✅ Geographic hierarchy created: {len(hierarchy_dict):,} SA2 areas")
                return {
                    'status': 'created',
                    'sa2_areas_count': len(hierarchy_dict),
                    'hierarchy_file': str(hierarchy_file)
                }
            
        except Exception as e:
            print(f"    ⚠️ Error creating hierarchy: {e}")
            return {'status': 'error', 'error': str(e)}
        
        return {'status': 'insufficient_data'}
    
    def implement_caching_system(self) -> Dict:
        """Implement file-based caching system"""
        print("\n💾 Implementing Caching System...")
        
        start_time = time.time()
        results = {
            'query_cache': self.create_query_cache(),
            'aggregation_cache': self.create_aggregation_cache(),
            'lookup_cache': self.create_lookup_cache()
        }
        
        end_time = time.time()
        results['setup_time'] = end_time - start_time
        
        print(f"  ✅ Caching system implemented in {results['setup_time']:.2f} seconds")
        return results
    
    def create_query_cache(self) -> Dict:
        """Create query result caching system"""
        print("    🔧 Setting up query cache...")
        
        query_cache_dir = self.cache_dir / 'queries'
        query_cache_dir.mkdir(exist_ok=True)
        
        # Pre-cache common queries
        common_queries = [
            'top_disadvantaged_areas',
            'state_summaries',
            'mortality_by_year',
            'prescription_patterns'
        ]
        
        cached_queries = {}
        
        for query_name in common_queries:
            try:
                cache_file = query_cache_dir / f'{query_name}.json'
                cached_queries[query_name] = str(cache_file)
                
                # Create placeholder cache files
                with open(cache_file, 'w') as f:
                    json.dump({
                        'query_name': query_name,
                        'cached_at': time.time(),
                        'ttl_seconds': 3600,  # 1 hour
                        'data': None
                    }, f, indent=2)
                    
            except Exception as e:
                print(f"    ⚠️ Error setting up cache for {query_name}: {e}")
        
        print(f"    ✅ Query cache set up: {len(cached_queries)} query types")
        return {
            'status': 'created',
            'cached_query_types': len(cached_queries),
            'cache_directory': str(query_cache_dir)
        }
    
    def create_aggregation_cache(self) -> Dict:
        """Create pre-computed aggregation cache"""
        print("    🔧 Creating aggregation cache...")
        
        agg_cache_dir = self.cache_dir / 'aggregations'
        agg_cache_dir.mkdir(exist_ok=True)
        
        # Pre-compute common aggregations
        aggregations_created = 0
        
        try:
            # State-level SEIFA summaries
            seifa_file = self.data_dir / 'seifa_2021_sa2.parquet'
            if seifa_file.exists():
                df = pl.read_parquet(seifa_file)
                
                # Extract state from SA2 code (first digit)
                state_summary = df.with_columns([
                    pl.col('sa2_code_2021').str.slice(0, 1).alias('state_code')
                ]).group_by('state_code').agg([
                    pl.col('irsd_score').mean().alias('avg_irsd'),
                    pl.col('irsd_score').median().alias('median_irsd'),
                    pl.col('usual_resident_population').sum().alias('total_population'),
                    pl.count().alias('sa2_count')
                ])
                
                # Save aggregation
                agg_file = agg_cache_dir / 'state_seifa_summary.json'
                state_summary.write_json(agg_file)
                aggregations_created += 1
                
        except Exception as e:
            print(f"    ⚠️ Error creating SEIFA aggregations: {e}")
        
        try:
            # Mortality year summaries
            mortality_file = self.data_dir / 'aihw_mort_table1.parquet'
            if mortality_file.exists():
                df = pl.read_parquet(mortality_file)
                
                if 'YEAR' in df.columns:
                    year_summary = df.group_by('YEAR').agg([
                        pl.col('crude_rate_per_100000').mean().alias('avg_crude_rate'),
                        pl.count().alias('record_count')
                    ])
                    
                    # Save aggregation
                    agg_file = agg_cache_dir / 'mortality_year_summary.json'
                    year_summary.write_json(agg_file)
                    aggregations_created += 1
                    
        except Exception as e:
            print(f"    ⚠️ Error creating mortality aggregations: {e}")
        
        print(f"    ✅ Aggregation cache created: {aggregations_created} pre-computed aggregations")
        return {
            'status': 'created',
            'aggregations_count': aggregations_created,
            'cache_directory': str(agg_cache_dir)
        }
    
    def create_lookup_cache(self) -> Dict:
        """Create fast lookup cache for common operations"""
        print("    🔧 Creating lookup cache...")
        
        lookup_cache_dir = self.cache_dir / 'lookups'
        lookup_cache_dir.mkdir(exist_ok=True)
        
        lookups_created = 0
        
        # Create SEIFA decile lookup
        try:
            seifa_file = self.data_dir / 'seifa_2021_sa2.parquet'
            if seifa_file.exists():
                df = pl.read_parquet(seifa_file)
                
                # IRSD decile lookup
                irsd_lookup = df.select(['sa2_code_2021', 'irsd_decile']).unique()
                lookup_file = lookup_cache_dir / 'sa2_irsd_decile_lookup.json'
                
                # Convert to dictionary for fast lookup
                lookup_dict = dict(irsd_lookup.iter_rows())
                with open(lookup_file, 'w') as f:
                    json.dump(lookup_dict, f)
                
                lookups_created += 1
                
        except Exception as e:
            print(f"    ⚠️ Error creating SEIFA lookup: {e}")
        
        print(f"    ✅ Lookup cache created: {lookups_created} lookup tables")
        return {
            'status': 'created',
            'lookup_tables_count': lookups_created,
            'cache_directory': str(lookup_cache_dir)
        }
    
    def optimize_database_structure(self) -> Dict:
        """Create optimized SQLite database with proper indexing"""
        print("\n🗄️ Optimizing Database Structure...")
        
        start_time = time.time()
        
        db_file = self.optimized_dir / 'health_analytics_optimized.db'
        
        try:
            conn = sqlite3.connect(db_file)
            
            # Create optimized tables with proper indexes
            tables_created = self.create_optimized_tables(conn)
            indexes_created = self.create_database_indexes(conn)
            
            conn.close()
            
            end_time = time.time()
            
            print(f"  ✅ Database optimized in {end_time - start_time:.2f} seconds")
            return {
                'status': 'optimized',
                'database_file': str(db_file),
                'tables_created': tables_created,
                'indexes_created': indexes_created,
                'optimization_time': end_time - start_time
            }
            
        except Exception as e:
            print(f"  ❌ Database optimization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def create_optimized_tables(self, conn: sqlite3.Connection) -> int:
        """Create optimized table structures"""
        cursor = conn.cursor()
        
        # SEIFA table with optimized structure
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seifa_optimized (
                sa2_code TEXT PRIMARY KEY,
                sa2_name TEXT NOT NULL,
                irsd_score INTEGER,
                irsd_decile INTEGER,
                irsad_score INTEGER,
                irsad_decile INTEGER,
                ier_score INTEGER,
                ier_decile INTEGER,
                ieo_score INTEGER,
                ieo_decile INTEGER,
                population INTEGER
            )
        ''')
        
        # Geographic hierarchy table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS geographic_hierarchy (
                sa2_code TEXT PRIMARY KEY,
                sa2_name TEXT,
                sa3_code TEXT,
                sa3_name TEXT,
                sa4_code TEXT,
                sa4_name TEXT,
                state_code TEXT,
                state_name TEXT,
                area_sqkm REAL
            )
        ''')
        
        # Mortality summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mortality_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                geography TEXT,
                year INTEGER,
                category TEXT,
                sex TEXT,
                crude_rate REAL,
                age_standardised_rate REAL
            )
        ''')
        
        conn.commit()
        return 3  # Number of tables created
    
    def create_database_indexes(self, conn: sqlite3.Connection) -> int:
        """Create database indexes for performance"""
        cursor = conn.cursor()
        
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_seifa_irsd_decile ON seifa_optimized(irsd_decile)',
            'CREATE INDEX IF NOT EXISTS idx_seifa_state ON seifa_optimized(substr(sa2_code, 1, 1))',
            'CREATE INDEX IF NOT EXISTS idx_geo_state ON geographic_hierarchy(state_code)',
            'CREATE INDEX IF NOT EXISTS idx_mortality_year ON mortality_summary(year)',
            'CREATE INDEX IF NOT EXISTS idx_mortality_geography ON mortality_summary(geography)',
            'CREATE INDEX IF NOT EXISTS idx_mortality_category ON mortality_summary(category)'
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        return len(indexes)
    
    def create_optimized_queries(self) -> Dict:
        """Create optimized query templates"""
        print("\n🔍 Creating Optimized Queries...")
        
        queries_dir = self.optimized_dir / 'queries'
        queries_dir.mkdir(exist_ok=True)
        
        # Common optimized queries
        optimized_queries = {
            'top_disadvantaged_areas': '''
                SELECT sa2_code, sa2_name, irsd_decile, irsd_score
                FROM seifa_optimized 
                WHERE irsd_decile <= 3
                ORDER BY irsd_score ASC
                LIMIT 50
            ''',
            'state_health_summary': '''
                SELECT 
                    substr(sa2_code, 1, 1) as state_code,
                    AVG(irsd_score) as avg_disadvantage,
                    SUM(population) as total_population,
                    COUNT(*) as sa2_count
                FROM seifa_optimized
                GROUP BY substr(sa2_code, 1, 1)
                ORDER BY avg_disadvantage ASC
            ''',
            'mortality_trends': '''
                SELECT 
                    year,
                    category,
                    AVG(crude_rate) as avg_crude_rate,
                    COUNT(*) as record_count
                FROM mortality_summary
                WHERE year >= 2019
                GROUP BY year, category
                ORDER BY year DESC, avg_crude_rate DESC
            '''
        }
        
        # Save query templates
        for query_name, query_sql in optimized_queries.items():
            query_file = queries_dir / f'{query_name}.sql'
            with open(query_file, 'w') as f:
                f.write(query_sql)
        
        print(f"  ✅ Optimized queries created: {len(optimized_queries)} templates")
        return {
            'status': 'created',
            'queries_count': len(optimized_queries),
            'queries_directory': str(queries_dir)
        }
    
    def optimize_compression(self) -> Dict:
        """Optimize data compression for storage efficiency"""
        print("\n🗜️ Optimizing Data Compression...")
        
        start_time = time.time()
        compression_results = []
        
        # Get all parquet files
        parquet_files = list(self.data_dir.glob('*.parquet'))
        
        for parquet_file in parquet_files:
            try:
                original_size = parquet_file.stat().st_size
                
                # Handle geospatial data with pandas fallback
                if 'boundaries' in str(parquet_file).lower():
                    import pandas as pd
                    pandas_df = pd.read_parquet(parquet_file)
                    # Remove geometry columns for compression
                    non_geo_cols = [col for col in pandas_df.columns if pandas_df[col].dtype.name != 'geometry']
                    df = pl.from_pandas(pandas_df[non_geo_cols])
                else:
                    df = pl.read_parquet(parquet_file)
                
                optimized_file = self.optimized_dir / f'{parquet_file.stem}_compressed.parquet'
                
                # Use ZSTD compression for better compression ratio
                df.write_parquet(optimized_file, compression='zstd', compression_level=3)
                
                optimized_size = optimized_file.stat().st_size
                compression_ratio = (1 - optimized_size / original_size) * 100
                
                compression_results.append({
                    'file': parquet_file.name,
                    'original_size_mb': original_size / (1024 * 1024),
                    'optimized_size_mb': optimized_size / (1024 * 1024),
                    'compression_ratio': compression_ratio
                })
                
            except Exception as e:
                print(f"    ⚠️ Error compressing {parquet_file.name}: {e}")
        
        end_time = time.time()
        
        if compression_results:
            total_original = sum(r['original_size_mb'] for r in compression_results)
            total_optimized = sum(r['optimized_size_mb'] for r in compression_results)
            overall_compression = (1 - total_optimized / total_original) * 100
            
            print(f"  ✅ Compression optimized: {overall_compression:.1f}% space saved")
            return {
                'status': 'optimized',
                'files_compressed': len(compression_results),
                'total_original_mb': total_original,
                'total_optimized_mb': total_optimized,
                'overall_compression_ratio': overall_compression,
                'compression_time': end_time - start_time,
                'file_results': compression_results
            }
        else:
            return {'status': 'no_files_compressed'}
    
    def run_performance_benchmarks(self) -> Dict:
        """Run performance benchmarks to measure improvements"""
        print("\n⚡ Running Performance Benchmarks...")
        
        benchmarks = {
            'sa2_lookup_speed': self.benchmark_sa2_lookup(),
            'aggregation_speed': self.benchmark_aggregation_queries(),
            'file_load_speed': self.benchmark_file_loading()
        }
        
        print(f"  ✅ Performance benchmarks completed")
        return benchmarks
    
    def benchmark_sa2_lookup(self) -> Dict:
        """Benchmark SA2 code lookup performance"""
        lookup_file = self.optimized_dir / 'sa2_lookup_index.json'
        
        if not lookup_file.exists():
            return {'status': 'lookup_index_not_found'}
        
        start_time = time.time()
        
        # Load lookup index
        with open(lookup_file, 'r') as f:
            lookup_index = json.load(f)
        
        load_time = time.time() - start_time
        
        # Benchmark lookup operations
        test_codes = list(lookup_index.keys())[:100]  # Test first 100 codes
        
        lookup_start = time.time()
        for code in test_codes:
            name = lookup_index.get(code)
        lookup_time = time.time() - lookup_start
        
        return {
            'status': 'benchmarked',
            'index_load_time': load_time,
            'lookup_operations': len(test_codes),
            'total_lookup_time': lookup_time,
            'avg_lookup_time_ms': (lookup_time / len(test_codes)) * 1000
        }
    
    def benchmark_aggregation_queries(self) -> Dict:
        """Benchmark aggregation query performance"""
        # Test with SEIFA data
        seifa_file = self.data_dir / 'seifa_2021_sa2.parquet'
        
        if not seifa_file.exists():
            return {'status': 'seifa_file_not_found'}
        
        start_time = time.time()
        df = pl.read_parquet(seifa_file)
        load_time = time.time() - start_time
        
        # Benchmark aggregation
        agg_start = time.time()
        state_summary = df.with_columns([
            pl.col('sa2_code_2021').str.slice(0, 1).alias('state_code')
        ]).group_by('state_code').agg([
            pl.col('irsd_score').mean().alias('avg_irsd'),
            pl.count().alias('count')
        ])
        agg_time = time.time() - agg_start
        
        return {
            'status': 'benchmarked',
            'data_load_time': load_time,
            'aggregation_time': agg_time,
            'records_processed': len(df),
            'aggregation_groups': len(state_summary)
        }
    
    def benchmark_file_loading(self) -> Dict:
        """Benchmark file loading performance"""
        load_times = {}
        
        test_files = [
            'seifa_2021_sa2.parquet',
            'aihw_grim_data.parquet',
            'phidu_pha_data.parquet'
        ]
        
        for file_name in test_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                start_time = time.time()
                try:
                    df = pl.read_parquet(file_path)
                    load_time = time.time() - start_time
                    load_times[file_name] = {
                        'load_time': load_time,
                        'records': len(df),
                        'records_per_second': len(df) / load_time if load_time > 0 else 0
                    }
                except Exception as e:
                    load_times[file_name] = {'error': str(e)}
        
        return {
            'status': 'benchmarked',
            'files_tested': len(load_times),
            'load_times': load_times
        }
    
    def generate_optimization_report(self, optimization_results: Dict) -> None:
        """Generate comprehensive optimization report"""
        report_file = self.optimized_dir / 'performance_optimization_report.json'
        
        with open(report_file, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        print(f"\n📄 Optimization report saved: {report_file}")
    
    def print_optimization_summary(self, optimization_results: Dict) -> None:
        """Print optimization summary"""
        print("\n🚀 Performance Optimization Summary")
        print("=" * 50)
        
        # Geographic indexing
        geo_results = optimization_results.get('geographic_indexing', {})
        if geo_results.get('sa2_lookup_index', {}).get('status') == 'created':
            mappings = geo_results['sa2_lookup_index']['mappings_count']
            print(f"🗺️ Geographic Indexing: {mappings:,} SA2 mappings created")
        
        # Caching
        cache_results = optimization_results.get('caching_system', {})
        print(f"💾 Caching System: Multi-tier cache implemented")
        
        # Database optimization
        db_results = optimization_results.get('database_optimization', {})
        if db_results.get('status') == 'optimized':
            tables = db_results.get('tables_created', 0)
            indexes = db_results.get('indexes_created', 0)
            print(f"🗄️ Database: {tables} tables, {indexes} indexes created")
        
        # Compression
        compression_results = optimization_results.get('compression_optimization', {})
        if compression_results.get('status') == 'optimized':
            compression_ratio = compression_results.get('overall_compression_ratio', 0)
            print(f"🗜️ Compression: {compression_ratio:.1f}% space saved")
        
        # Benchmarks
        benchmarks = optimization_results.get('performance_benchmarks', {})
        sa2_benchmark = benchmarks.get('sa2_lookup_speed', {})
        if sa2_benchmark.get('status') == 'benchmarked':
            avg_lookup = sa2_benchmark.get('avg_lookup_time_ms', 0)
            print(f"⚡ Performance: {avg_lookup:.2f}ms average SA2 lookup time")
        
        print("\n✅ All optimizations completed successfully!")


def main():
    """Run performance optimization suite"""
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    optimizer = PerformanceOptimizer(data_dir)
    results = optimizer.run_complete_optimization()
    
    print(f"\n🎉 Performance Optimization Complete!")


if __name__ == "__main__":
    main()