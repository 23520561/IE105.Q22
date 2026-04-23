import { apiUrl, postData } from "~/api";

export type TransformationMethodType =
  | "log"
  | "sqrt"
  | "minmax"
  | "standard"
  | "robust"
  | "power"
  | "normalize";
export type EncodingMethodType =
  | "one_hot"
  | "label"
  | "target"
  | "count"
  | "freq"
  | "binary"
  | "ordinal";
export type PipelineStepType = {
  method: EncodingMethodType | TransformationMethodType;

  columns?: string[];
  column?: string;
  target?: string;
  mapping?: Record<string, any>;
};
export async function enconding(datasetId: string, req: PipelineStepType) {
  const prefix = `/features/encoding?dataset_id=${datasetId}`;
  await postData(apiUrl + prefix, req);
}
export async function transform(datasetId: string, req: PipelineStepType) {
  const prefix = `/features/transformation?dataset_id=${datasetId}`;
  await postData(apiUrl + prefix, req);
}
