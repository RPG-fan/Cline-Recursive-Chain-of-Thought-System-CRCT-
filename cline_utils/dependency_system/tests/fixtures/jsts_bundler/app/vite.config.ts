import path from "path";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

export default defineConfig(({ mode }) => {
  return {
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "src"),
        "~lib": fileURLToPath(new URL("./lib", import.meta.url)),
        "#root": "/rootrel",
        "computed": someHelper("src"),
      },
    },
  };
});
