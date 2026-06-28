import { apiUrl, postData } from "~/api";

export type RfeResponse = {
  recommended_to_keep: string[];
  feature_ranking: Record<string, number>;
  n_features_kept: number;
  estimator_used: string;
  feature_importances: Record<string, number>;
};
export class RfeRequest {
  constructor(public numberFeature: number = 2) {}
}
export async function getRfe(
  datasetId: string,
  req: RfeRequest,
): Promise<RfeResponse | null> {
  const prefix = `/feature-selection/rfe?dataset_id=${datasetId}`;
  return await postData<RfeResponse>(apiUrl + prefix, req);
}
