import { Router, type Request, type Response } from "express";
import { apiUrl, getData, postData } from "~/api.js";
import { isAuthenticated } from "~/controllers/authentication.js";
import { getPrebuiltdatasets } from "~/controllers/prebuiltdataset.js";
import { findProjectByIdAndUserId } from "~/db/project.js";

const datasetRoute = Router();

datasetRoute.get("/prebuilt", getPrebuiltdatasets);

datasetRoute.get(
  "/*splat",
  isAuthenticated,
  async (req: Request, res: Response) => {
    if (!req.user) {
      throw new Error("Authentication not working");
    }
    const id = req.query.dataset_id?.toString();
    if (!id) {
      throw new Error("No params");
    }
    const filename = (await findProjectByIdAndUserId(id, req.user.id)).filename;
    res.json(await getData((apiUrl + req.originalUrl).replace(id, filename)));
  },
);
datasetRoute.post(
  "/*splat",
  isAuthenticated,
  async (req: Request, res: Response) => {
    if (!req.user) {
      throw new Error("Authentication not working");
    }
    const id = req.query.dataset_id?.toString();
    if (!id) {
      throw new Error("No params");
    }
    const filename = (await findProjectByIdAndUserId(id, req.user.id)).filename;

    res.json(
      postData((apiUrl + req.originalUrl).replace(id, filename), req.body),
    );
  },
);
export default datasetRoute;
