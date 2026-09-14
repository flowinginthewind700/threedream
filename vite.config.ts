import { defineConfig } from 'vite';

export default defineConfig({
  root: 'demo',
  resolve: {
    alias: {
      '@threedream': new URL('./src', import.meta.url).pathname,
    },
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'es2022',
  },
  server: {
    port: 5173,
  },
});
