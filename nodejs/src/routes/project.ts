import { Router } from "express";
import { isAuthenticated } from "~/controllers/authentication.js";
import { addProject, getProjectsByUserId } from "~/controllers/project.js";
import { validateField } from "~/utils/fieldValidate.js";

const projectRoute = Router();
projectRoute.get("/", isAuthenticated, getProjectsByUserId);
projectRoute.post(
  "/",
  isAuthenticated,
  validateField("name", "datasetId"),
  addProject,
);
export default projectRoute;
