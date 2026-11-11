import { defineConfig } from 'vite';

export default defineConfig({
  cacheDir: '../runtime/vite',
  server: {
    host: '127.0.0.1',
    proxy: { '/api': 'http://127.0.0.1:8080' },
  },
});
