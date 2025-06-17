/**
 * Service Worker for Australian Health Analytics Platform
 * Optimizes loading performance and enables offline functionality
 */

const CACHE_NAME = 'health-analytics-v1.0.0';
const STATIC_CACHE = 'static-v1';
const DYNAMIC_CACHE = 'dynamic-v1';

// Critical resources to cache immediately
const CORE_ASSETS = [
  '/',
  '/index.html',
  '/assets/css/styles.css',
  '/assets/js/main.js',
  '/data/json/api/v1/overview.json',
  '/data/geojson/centroids/sa2_centroids.geojson',
  '/data/json/performance/platform_performance.json'
];

// External CDN resources
const CDN_ASSETS = [
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css',
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
];

// Install event - cache core assets
self.addEventListener('install', event => {
  console.log('🚀 Service Worker installing...');
  
  event.waitUntil(
    Promise.all([
      // Cache core application assets
      caches.open(STATIC_CACHE).then(cache => {
        console.log('📦 Caching core assets...');
        return cache.addAll(CORE_ASSETS);
      }),
      
      // Cache CDN resources
      caches.open(CDN_ASSETS).then(cache => {
        console.log('🌐 Caching CDN assets...');
        return cache.addAll(CDN_ASSETS.map(url => new Request(url, { mode: 'cors' })));
      })
    ]).then(() => {
      console.log('✅ Service Worker installation complete');
      return self.skipWaiting();
    }).catch(error => {
      console.error('❌ Service Worker installation failed:', error);
    })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('🔄 Service Worker activating...');
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
            console.log('🗑️ Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('✅ Service Worker activation complete');
      return self.clients.claim();
    })
  );
});

// Fetch event - serve cached content with fallback strategies
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') return;
  
  // Strategy: Cache First for static assets
  if (isStaticAsset(request.url)) {
    event.respondWith(cacheFirst(request));
    return;
  }
  
  // Strategy: Network First for API data
  if (isApiRequest(request.url)) {
    event.respondWith(networkFirst(request));
    return;
  }
  
  // Strategy: Stale While Revalidate for other resources
  event.respondWith(staleWhileRevalidate(request));
});

// Cache strategies
async function cacheFirst(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.error('Cache first strategy failed:', error);
    return new Response('Offline - Content unavailable', { status: 503 });
  }
}

async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('Network failed, serving from cache:', request.url);
    const cachedResponse = await caches.match(request);
    return cachedResponse || new Response('Offline - Data unavailable', { 
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

async function staleWhileRevalidate(request) {
  try {
    const cachedResponse = await caches.match(request);
    
    const networkResponsePromise = fetch(request).then(response => {
      if (response.ok) {
        const cache = caches.open(DYNAMIC_CACHE);
        cache.then(c => c.put(request, response.clone()));
      }
      return response;
    }).catch(() => null);
    
    return cachedResponse || await networkResponsePromise;
  } catch (error) {
    console.error('Stale while revalidate failed:', error);
    return new Response('Service temporarily unavailable', { status: 503 });
  }
}

// Helper functions
function isStaticAsset(url) {
  return url.includes('/assets/') || 
         url.includes('.css') || 
         url.includes('.js') || 
         url.includes('/sw.js');
}

function isApiRequest(url) {
  return url.includes('/data/') || 
         url.includes('/api/') ||
         url.includes('.json') ||
         url.includes('.geojson');
}

// Background sync for analytics (optional)
self.addEventListener('sync', event => {
  if (event.tag === 'analytics-sync') {
    event.waitUntil(syncAnalytics());
  }
});

async function syncAnalytics() {
  try {
    // Implement analytics sync if needed
    console.log('📊 Syncing analytics data...');
  } catch (error) {
    console.error('Analytics sync failed:', error);
  }
}

// Performance monitoring
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'PERFORMANCE_LOG') {
    console.log('⚡ Performance metrics:', event.data.metrics);
  }
});