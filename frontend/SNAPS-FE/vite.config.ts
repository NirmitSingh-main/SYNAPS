import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
  plugins: [
    tsconfigPaths(),
    TanStackRouterVite({ routesDirectory: "src/routes", generatedRouteTree: "src/routeTree.gen.ts" }),
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    open: true,
    proxy: {
      "/signal": "http://127.0.0.1:8000",
      "/analysis": "http://127.0.0.1:8000",
      "/report": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
