module.exports = {
  ci: {
    collect: {
      url: ['https://massimoraso.github.io/australian-health-analytics/'],
      startServerCommand: null,
      numberOfRuns: 3,
      settings: {
        chromeFlags: '--no-sandbox --headless'
      }
    },
    assert: {
      assertions: {
        'categories:performance': ['warn', { minScore: 0.85 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['warn', { minScore: 0.85 }],
        'categories:seo': ['error', { minScore: 0.9 }],
        'categories:pwa': ['off'],
        
        // Performance budgets
        'first-contentful-paint': ['warn', { maxNumericValue: 2000 }],
        'largest-contentful-paint': ['warn', { maxNumericValue: 2500 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'total-blocking-time': ['warn', { maxNumericValue: 300 }],
        
        // Resource audits
        'unused-css-rules': ['warn', { maxLength: 2 }],
        'unused-javascript': ['warn', { maxLength: 2 }],
        'render-blocking-resources': ['warn', { maxLength: 1 }],
        
        // Image optimization
        'uses-optimized-images': ['warn', { maxLength: 0 }],
        'uses-webp-images': ['warn', { maxLength: 2 }],
        'offscreen-images': ['warn', { maxLength: 1 }],
        
        // Network efficiency
        'uses-text-compression': ['error', { maxLength: 0 }],
        'uses-rel-preconnect': ['warn', { maxLength: 2 }],
        
        // Security
        'is-on-https': ['error', { minScore: 1 }],
        'uses-http2': ['warn', { minScore: 0.5 }]
      }
    },
    upload: {
      target: 'temporary-public-storage'
    },
    server: {
      port: 9001,
      storage: {
        storageMethod: 'sql',
        sqlDialect: 'sqlite',
        sqlDatabasePath: '.lighthouseci/database.sql'
      }
    }
  }
};