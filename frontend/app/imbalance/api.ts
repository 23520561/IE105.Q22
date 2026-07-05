import { apiUrl, postData } from "~/api";

type ImbalancedInfoType = {
  name: string;
  description: string;
  method: string;
};
export const ImbalancedInfo: ImbalancedInfoType[] = [
  {
    name: "SMOTE",
    description:
      "Create new examples for classes with fewer samples to balance your data.",
    method: "smote",
  },
  {
    name: "Undersampling",
    description: "Reduce classes with too many samples to balance your data.",
    method: "undersample",
  },
  {
    name: "Oversampling",
    description:
      "Increase classes with fewer samples by duplicating examples to balance your data.",
    method: "oversample",
  },
] as const;
export type ImbalancedMethod = (typeof ImbalancedInfo)[number]["method"];
export class ImbalancedRequest {
  constructor(
    public target: string,
    public method: ImbalancedMethod,
  ) {}
}
export async function imbalanced(datasetId: string, req: ImbalancedRequest) {
  const prefix = `/features/imbalanced/?dataset_id=${datasetId}`;

  return await postData(apiUrl + prefix, req);
}
