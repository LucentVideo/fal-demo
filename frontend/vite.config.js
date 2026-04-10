import { defineConfig } from "vite";

export default defineConfig({
  envDir: "..",
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
