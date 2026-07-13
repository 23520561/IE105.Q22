import pool from "./pool.js";

type DatasetSchema = {
  id: string;
  name: string;
  filename: string;
  userId: string;
};
export async function findDatasetFromDatasetId(
  datasetId: string,
): Promise<DatasetSchema> {
  const { rows } = await pool.query("SELECT * FROM dataset WHERE id = $1", [
    datasetId,
  ]);
  return rows[0];
}
export async function findDatasetsFromUserId(
  userId: string,
): Promise<DatasetSchema[]> {
  const { rows } = await pool.query("SELECT * FROM dataset WHERE userId = $1", [
    userId,
  ]);
  return rows;
}
