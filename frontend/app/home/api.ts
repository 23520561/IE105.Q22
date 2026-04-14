import { apiUrl, getData } from "~/api";

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
