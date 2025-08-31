# Geographic API  
### High-Performance Spatial Data Access

The Geographic API delivers lightning-fast access to Australia's complete SA1 geography (61,845 areas) with sub-50ms response times powered by optimized spatial indexing and Parquet storage.

---

## 🗺️ Core Endpoints

### Get SA1 Area Details
Get comprehensive geographic information for a specific SA1 area.

```http
GET /v1/geo/sa1/{sa1_code}
```

**Parameters:**
- `sa1_code` (required): 11-digit SA1 area code

**Response Time:** <50ms

**Example:**
```bash
curl -H "X-API-Key: your-key" \
     https://api.ahgd.dev/v1/geo/sa1/101011001
```

**Response:**
```json
{
  "sa1_code": "101011001",
  "area_name": "Sydney - Circular Quay",
  "state": "NSW",
  "coordinates": {
    "centroid": {
      "latitude": -33.8568,
      "longitude": 151.2153
    },
    "bounding_box": {
      "north": -33.8520,
      "south": -33.8616,
      "east": 151.2201,
      "west": 151.2105
    }
  },
  "area_metrics": {
    "area_sqkm": 0.147,
    "perimeter_km": 1.89,
    "population_density": 4238.1,
    "urban_classification": "Major Urban"
  },
  "hierarchy": {
    "sa2_code": "10101",
    "sa2_name": "Sydney - Circular Quay - The Rocks",
    "sa3_code": "1010",
    "sa3_name": "Sydney Inner City",
    "sa4_code": "101",
    "sa4_name": "Sydney - City and Inner South",
    "gcc_code": "1GSYD",
    "gcc_name": "Greater Sydney"
  },
  "demographics": {
    "population_2021": 623,
    "dwelling_count": 387,
    "average_household_size": 1.61
  },
  "data_sources": ["abs_asgs_2021", "abs_census_2021"]
}
```

### Get Area Boundaries
Get precise boundary geometries in GeoJSON format.

```http  
GET /v1/geo/boundaries?sa1_codes={codes}&format={format}
```

**Parameters:**
- `sa1_codes`: Comma-separated list of SA1 codes (max 100)
- `format`: Response format (`geojson`, `wkt`, `simplified`)

**Response Time:** <100ms

**Example:**
```bash
curl -H "X-API-Key: your-key" \
     "https://api.ahgd.dev/v1/geo/boundaries?sa1_codes=101011001,101011002&format=geojson"
```

**Response:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "sa1_code": "101011001",
        "area_name": "Sydney - Circular Quay",
        "state": "NSW",
        "area_sqkm": 0.147,
        "population": 623
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [151.2105, -33.8520],
          [151.2201, -33.8520],
          [151.2201, -33.8616],
          [151.2105, -33.8616],
          [151.2105, -33.8520]
        ]]
      }
    }
  ],
  "processing_time_ms": 87,
  "coordinate_system": "GDA2020 / MGA Zone 56 (EPSG:7856)"
}
```

### Find Nearby Areas
Find SA1 areas within a specified distance of a point or area.

```http
POST /v1/geo/nearby
```

**Response Time:** <150ms

**Request Body:**
```json
{
  "center": {
    "sa1_code": "101011001"
  },
  "radius_km": 2.5,
  "limit": 50,
  "include_distance": true,
  "sort_by": "distance"
}
```

**Alternative - Point-based search:**
```json
{
  "center": {
    "latitude": -33.8568,
    "longitude": 151.2153
  },
  "radius_km": 2.5,
  "limit": 50
}
```

**Response:**
```json
{
  "search_center": {
    "sa1_code": "101011001",
    "latitude": -33.8568,
    "longitude": 151.2153
  },
  "radius_km": 2.5,
  "total_results": 47,
  "results": [
    {
      "sa1_code": "101011002",
      "area_name": "Sydney - The Rocks",
      "distance_km": 0.34,
      "bearing_degrees": 285,
      "population": 892,
      "coordinates": {
        "latitude": -33.8590,
        "longitude": 151.2089
      }
    },
    {
      "sa1_code": "101021001", 
      "area_name": "Sydney - CBD South",
      "distance_km": 0.67,
      "bearing_degrees": 195,
      "population": 1456,
      "coordinates": {
        "latitude": -33.8638,
        "longitude": 151.2078
      }
    }
  ],
  "processing_time_ms": 134
}
```

### Geographic Hierarchy
Get complete geographic hierarchy for area(s).

```http
GET /v1/geo/hierarchy?sa1_code={code}&levels={levels}
```

**Parameters:**
- `sa1_code`: Target SA1 area code
- `levels`: Hierarchy levels to include (`sa2,sa3,sa4,gcc,state`)

**Response Time:** <75ms

**Example:**
```bash
curl -H "X-API-Key: your-key" \
     "https://api.ahgd.dev/v1/geo/hierarchy?sa1_code=101011001&levels=sa2,sa3,sa4"
```

**Response:**
```json
{
  "sa1": {
    "code": "101011001",
    "name": "Sydney - Circular Quay",
    "area_sqkm": 0.147,
    "population": 623
  },
  "sa2": {
    "code": "10101",
    "name": "Sydney - Circular Quay - The Rocks", 
    "area_sqkm": 2.34,
    "population": 4567,
    "sa1_count": 8
  },
  "sa3": {
    "code": "1010",
    "name": "Sydney Inner City",
    "area_sqkm": 23.45,
    "population": 98234,
    "sa2_count": 12
  },
  "sa4": {
    "code": "101",
    "name": "Sydney - City and Inner South",
    "area_sqkm": 234.56,
    "population": 567890,
    "sa3_count": 8
  }
}
```

---

## 🔍 Advanced Search & Filtering

### Area Search by Name
```http
GET /v1/geo/search?query={text}&limit={n}&fuzzy={bool}
```

**Example:**
```bash
curl -H "X-API-Key: your-key" \
     "https://api.ahgd.dev/v1/geo/search?query=circular%20quay&fuzzy=true&limit=10"
```

**Response:**
```json
{
  "query": "circular quay",
  "fuzzy_matching": true,
  "results": [
    {
      "sa1_code": "101011001",
      "area_name": "Sydney - Circular Quay",
      "state": "NSW",
      "match_score": 0.98,
      "population": 623
    }
  ],
  "total_results": 1
}
```

### Bounding Box Search
```http
POST /v1/geo/search/bbox
```

**Request Body:**
```json
{
  "bounding_box": {
    "north": -33.85,
    "south": -33.87,
    "east": 151.22,
    "west": 151.20
  },
  "include_partial": true,
  "limit": 100
}
```

### State & Region Filtering
```http
GET /v1/geo/areas?state={code}&sa4={code}&population_min={n}
```

---

## 📐 Spatial Analysis

### Distance Calculations
```http
POST /v1/geo/distance
```

**Request Body:**
```json
{
  "origins": ["101011001", "201031245"],
  "destinations": ["301051289", "401061234"],
  "units": "km",
  "method": "haversine"
}
```

**Response:**
```json
{
  "distance_matrix": {
    "101011001": {
      "301051289": 735.2,
      "401061234": 878.4
    },
    "201031245": {
      "301051289": 1342.7,
      "401061234": 1456.8
    }
  },
  "units": "km",
  "method": "haversine"
}
```

### Catchment Area Analysis
```http
POST /v1/geo/catchment
```

**Request Body:**
```json
{
  "center": {
    "sa1_code": "101011001"
  },
  "travel_time_minutes": 15,
  "transport_mode": "driving",
  "include_population": true
}
```

**Response:**
```json
{
  "catchment_analysis": {
    "center_area": "101011001",
    "travel_time_minutes": 15,
    "transport_mode": "driving",
    "areas_within_catchment": 284,
    "total_population": 127453,
    "total_area_sqkm": 45.7
  },
  "areas": [
    {
      "sa1_code": "101011002",
      "travel_time_minutes": 3,
      "population": 892
    }
  ]
}
```

---

## 🗺️ Data Formats

### GeoJSON (Default)
Standard GeoJSON format with full feature properties.

### Well-Known Text (WKT)  
```
POLYGON((151.2105 -33.8520, 151.2201 -33.8520, 151.2201 -33.8616, 151.2105 -33.8616, 151.2105 -33.8520))
```

### Simplified Geometries
Reduced precision for web mapping (up to 80% smaller).

### Coordinate Systems
- **GDA2020** (default): Modern Australian coordinate system
- **GDA94**: Legacy coordinate system (for compatibility)
- **WGS84**: International standard

---

## 📊 Geographic Statistics

### Area Classifications
| Classification | Description | SA1 Count |
|----------------|-------------|-----------|
| Major Urban | Population >100k | 35,624 |
| Other Urban | Population 1k-100k | 18,453 |
| Bounded Locality | Rural town/locality | 5,892 |
| Rural Balance | Remainder rural | 1,876 |

### State Coverage
| State/Territory | SA1 Areas | Population |
|----------------|-----------|------------|
| NSW | 19,368 | 8,166,369 |
| VIC | 16,927 | 6,681,085 |
| QLD | 13,345 | 5,184,847 |
| WA | 7,543 | 2,667,130 |
| SA | 4,821 | 1,771,703 |
| TAS | 1,421 | 541,965 |
| ACT | 906 | 454,499 |
| NT | 617 | 249,129 |

---

## ⚡ Performance Features

### Spatial Indexing
- **R-tree indexing** for O(log n) spatial queries
- **Grid-based partitioning** for distance searches  
- **Proximity caching** for frequently accessed areas

### Response Optimization
```json
{
  "geometry_precision": 6,    // Decimal places (default)
  "simplify_tolerance": 0.01, // Geometry simplification
  "include_geometry": false,  // Skip geometry for faster response
  "fields": ["basic_info"]    // Limit response fields
}
```

### Caching Strategy
- **Spatial queries**: Cached 30 minutes
- **Area details**: Cached 2 hours
- **Boundaries**: Cached 24 hours (stable data)
- **Hierarchy**: Cached 24 hours

---

## 🚨 Error Handling

| Error Code | Description | Solution |
|------------|-------------|----------|
| `INVALID_SA1_CODE` | Invalid SA1 format | Use 11-digit numeric code |
| `AREA_NOT_FOUND` | SA1 area doesn't exist | Verify code with search |
| `INVALID_COORDINATES` | Lat/lng out of range | Check coordinate bounds |
| `RADIUS_TOO_LARGE` | Search radius >50km | Reduce radius or use pagination |
| `GEOMETRY_COMPLEX` | Geometry too complex | Use simplified format |

---

## 💡 Usage Examples

### Python - Spatial Analysis
```python
from ahgd import GeoAPI
import geopandas as gpd

client = GeoAPI(api_key="your-key")

# Get area details
area = client.get_area_details("101011001")
print(f"Area: {area.area_sqkm:.2f} km²")

# Find nearby areas
nearby = client.find_nearby_areas(
    sa1_code="101011001",
    radius_km=2.0,
    limit=20
)

# Get boundaries for mapping
boundaries = client.get_boundaries([a.sa1_code for a in nearby])
gdf = gpd.GeoDataFrame.from_features(boundaries["features"])
```

### R - Geographic Analysis  
```r
library(ahgd)
library(sf)

client <- ahgd_geo_client("your-api-key")

# Get SA1 boundaries
boundaries <- get_boundaries(
  client, 
  sa1_codes = c("101011001", "101011002"),
  format = "geojson"
)

# Convert to sf object
sf_boundaries <- st_read(boundaries)

# Spatial operations
area_km2 <- st_area(sf_boundaries) / 1000000
```

### JavaScript - Interactive Maps
```javascript
import { GeoAPI } from '@ahgd/js-sdk';
import L from 'leaflet';

const geoClient = new GeoAPI('your-api-key');

// Get area and add to map
async function addAreaToMap(sa1Code) {
  const boundaries = await geoClient.getBoundaries([sa1Code]);
  
  const geoJsonLayer = L.geoJSON(boundaries, {
    style: {
      color: '#3388ff',
      weight: 2,
      fillOpacity: 0.3
    },
    onEachFeature: (feature, layer) => {
      layer.bindPopup(`
        <h4>${feature.properties.area_name}</h4>
        <p>Population: ${feature.properties.population.toLocaleString()}</p>
        <p>Area: ${feature.properties.area_sqkm} km²</p>
      `);
    }
  });
  
  map.addLayer(geoJsonLayer);
}
```

---

**[← Back to API Hub](README.md)** | **[Next: Analytics API →](analytics-api.md)**

---

*Last updated: August 2024 • GDA2020 coordinate system • Powered by spatial indexing*