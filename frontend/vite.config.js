import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/fal/token": {
        target: "https://fal.run",
        changeOrigin: true,
        secure: true,
      },
    },
  },
});
