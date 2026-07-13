import { findDatasetsFromUserId } from "~/db/dataset.js";

export async function getDatasetsFromUserId(userId: string) {
  const datasets = await findDatasetsFromUserId(userId);
  return datasets;
}
