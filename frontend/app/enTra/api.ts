import { apiUrl, postData } from "~/api";

export type TransformationMethodType =
  "log" | "sqrt" | "minmax" | "standard" | "robust" | "power" | "normalize";
export const EncodingMethod = [
  {
    name: "one_hot",
    description:
      "Creates a new binary column for each unique category. Best for features with a small number of unique values.",
  },
  {
    name: "label",
    description:
      "Assigns each category a unique integer. Suitable for tree-based models, but may introduce an artificial order for other models.",
  },
  {
    name: "target",
    description:
      "Replaces each category with a statistic (typically the mean) of the target variable. Useful for high-cardinality features but should be applied carefully to avoid data leakage.",
  },
  {
    name: "count",
    description:
      "Replaces each category with the number of times it appears in the dataset. Suitable for features with many unique values.",
  },
  {
    name: "freq",
    description:
      "Replaces each category with its relative frequency (proportion) in the dataset. Similar to count encoding but normalized.",
  },
  {
    name: "binary",
    description:
      "Encodes categories as binary numbers across multiple columns. Uses fewer columns than one-hot encoding and works better for medium to high-cardinality features.",
  },
  {
    name: "ordinal",
    description:
      "Maps categories to integers based on a specified order. Only use when the categories have a meaningful ranking (e.g. Low < Medium < High).",
  },
] as const;
export type EncodingMethodType = (typeof EncodingMethod)[number]["name"];
type EncodingData = {
  columns?: string[];
  column?: string;
  target?: string;
  mapping?: Record<string, any>;
};
export class EncodingRequest {
  public column?: string;
  public target?: string;
  public mapping?: Record<string, any>;
  constructor(
    public method: EncodingMethodType,
    public data: EncodingData,
  ) {
    switch (method) {
      case "target":
        if (!data.column || !data.target) {
          throw new Error(
            `They shouldn't be null in EncodingRequest(method: ${method}, target: ${data.target}, column: ${data.column}`,
          );
        }
        this.column = data.column;
        this.target = data.target;
        break;
      case "ordinal":
        if (!data.column || !data.mapping) {
          throw new Error(
            `They shouldn't be null in EncodingRequest(method: ${method}, mapping: ${data.mapping}, column: ${data.column}`,
          );
        }
        this.column = data.column;
        this.mapping = data.mapping;
        break;
      default:
        if (!data.column) {
          throw new Error(
            `They shouldn't be null in EncodingRequest(method: ${method}, column: ${data.column}`,
          );
        }
        this.column = data.column;
    }
  }
}
export class TransformationRequest {
  constructor(
    public method: TransformationMethodType,
    public columns: string[],
  ) {}
}
export async function enconding(datasetId: string, req: EncodingRequest) {
  const prefix = `/features/encoding?dataset_id=${datasetId}`;
  await postData(apiUrl + prefix, req);
}
export async function transform(datasetId: string, req: TransformationRequest) {
  const prefix = `/features/transformation?dataset_id=${datasetId}`;
  await postData(apiUrl + prefix, req);
}
