import { apiUrl, deleteData, getData, postData } from "~/api";
import type { uploadedDatasetType } from "~/seed";

export type projectResponseType = {
  id: string;
  name: string;
  date: string;
};
export async function getProjects(): Promise<projectResponseType[] | null> {
  const prefix = "/project";
  return await getData<projectResponseType[]>(apiUrl + prefix);
}
export type projectRequestType = {
  name: string;
  dataset_id: string;
};
export async function createProjectName(
  req: projectRequestType,
): Promise<string | null> {
  const prefix = "/project";
  return await postData(apiUrl + prefix, req);
}
export type prebuiltDatasetType = {
  id: string;
  name: string;
  image: string;
  description: string;
};
export async function getPrebuiltDatasets(): Promise<
  prebuiltDatasetType[] | null
> {
  const prefix = "/dataset/prebuilt";
  const url = apiUrl + prefix;
  return await getData<prebuiltDatasetType[]>(url);
}

export type serverStatusType = {
  ram: number;
  storage: string;
};
export async function getServerStatus(): Promise<serverStatusType | null> {
  const prefix = "/server/status";
  return await getData<serverStatusType>(apiUrl + prefix);
}
export async function getUploadedDatasets(
  workspace: string,
): Promise<uploadedDatasetType[] | null> {
  const prefix = `/dataset/uploaded`;
  return await getData<uploadedDatasetType[]>(apiUrl + prefix, workspace);
}
export async function deleteUploadedDatasets(fileId: string) {
  const prefix = `/dataset/uploaded?file_id=${fileId}`;
  return await deleteData(apiUrl + prefix);
}
