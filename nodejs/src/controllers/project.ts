import { randomUUID } from "crypto";
import type { NextFunction, Request, Response } from "express";
import { apiUrl, postData } from "~/api.js";
import { findDatasetFromDatasetId } from "~/db/dataset.js";
import { findPrebuiltdatasetFromId } from "~/db/prebuiltdataset.js";
import {
  findProjectByIdAndUserId,
  findProjectsFromUserId,
  insertProject,
} from "~/db/project.js";

class ProjectRequest {
  project_filename;
  dataset_filename;
  constructor(project_filename: string, dataset_filename: string) {
    this.dataset_filename = dataset_filename;
    this.project_filename = project_filename;
  }
}
export async function getProjectsByUserId(req: Request, res: Response) {
  if (!req.user) {
    throw new Error("Something broken with user authenticate");
  }
  const projects = await findProjectsFromUserId(req.user.id);
  res.json(projects);
}
export async function addProject(req: Request, res: Response) {
  if (!req.user) {
    throw new Error("Something broken with user authenticate");
  }
  const filename = randomUUID();
  let datasetFilename =
    (await findDatasetFromDatasetId(req.body.datasetId))?.filename ||
    (await findPrebuiltdatasetFromId(req.body.datasetId)).name
      .split(" ")[0]
      .toLowerCase();
  await postData(
    apiUrl + "/project",
    new ProjectRequest(filename, datasetFilename),
  );
  const { id } = await insertProject(req.body.name, req.user.id, filename);
  res.json(id);
}
