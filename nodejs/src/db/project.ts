import pool from "./pool.js";

type ProjectSchema = {
  id: string;
  name: string;
  userId: string;
  filename: string;
};
export async function findProjectByIdAndUserId(
  id: string,
  userId: string,
): Promise<ProjectSchema> {
  const { rows } = await pool.query(
    "SELECT * FROM project WHERE id=$1 AND userId = $2",
    [id, userId],
  );
  return rows[0];
}
export async function findProjectsFromUserId(userId: string) {
  const { rows } = await pool.query("SELECT * FROM project WHERE userId = $1", [
    userId,
  ]);
  return rows;
}
export async function insertProject(
  name: string,
  userId: string,
  filename: string,
) {
  const { rows } = await pool.query(
    "INSERT INTO project(name, filename, userId) VALUES ($1, $2, $3) RETURNING *",
    [name, filename, userId],
  );
  return rows[0];
}
