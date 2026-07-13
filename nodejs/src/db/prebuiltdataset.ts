import pool from "./pool.js";

export async function findPrebuiltdatasetFromId(id: string) {
  const { rows } = await pool.query(
    "SELECT * FROM prebuiltdataset WHERE id = $1",
    [id],
  );
  return rows[0];
}
export async function findPrebuiltdatasets() {
  const { rows } = await pool.query("SELECT * FROM prebuiltdataset");
  return rows;
}
