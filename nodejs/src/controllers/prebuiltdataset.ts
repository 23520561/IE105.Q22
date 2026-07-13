import type { Request, Response } from "express";
import { findPrebuiltdatasets } from "~/db/prebuiltdataset.js";

export async function getPrebuiltdatasets(_: Request, res: Response) {
  const datasets = await findPrebuiltdatasets();
  res.json(datasets);
}
