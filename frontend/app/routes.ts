import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("eda/:datasetId", "routes/eda.tsx"),
] satisfies RouteConfig;
