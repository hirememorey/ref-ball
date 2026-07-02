import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://ref-ball.com',
  output: 'static',
  build: { format: 'directory' },
});
