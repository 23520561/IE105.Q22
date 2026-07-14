import pool from "./pool.js";

export type usersSchema = {
  id: string;
  username: string;
  password: string;
};
async function findUserById(id: string): Promise<usersSchema> {
  const { rows } = await pool.query("SELECT * FROM users WHERE id = $1", [id]);
  return rows[0];
}
async function findUserByUsername(username: string) {
  const { rows }: { rows: usersSchema[] } = await pool.query(
    "SELECT * FROM users WHERE username = $1",
    [username],
  );
  return rows[0];
}
async function createUser(username: string, password: string) {
  await pool.query("INSERT INTO users(username, password) VALUES ($1, $2)", [
    username,
    password,
  ]);
}
export { findUserById, findUserByUsername, createUser };
