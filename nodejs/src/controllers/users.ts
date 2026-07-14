import type { Request, Response } from "express";
import { findUserById } from "~/db/users.js";

export async function getUserProfile(req: Request, res: Response) {
  if (!req.user) {
    throw new Error("Authentication not working");
  }
  const user = await findUserById(req.user.id);
  return res.json({ username: user.username });
}
