import bcrypt from "bcryptjs";
import type { NextFunction, Request, Response } from "express";
import passport from "passport";
import { createUser, findUserByUsername } from "~/db/users.js";

export async function isAuthenticated(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  if (!req.isAuthenticated()) {
    res.status(401).json({ message: "User is unauthorized" });
    return;
  }
  next();
}
export async function login(req: Request, res: Response, next: NextFunction) {
  await passport.authenticate("local", {
    failureMessage: true,
  })(req, res, next);
}
export async function signup(req: Request, res: Response, next: NextFunction) {
  try {
    if (!req.body.username || !req.body.password) {
      next(new Error("Username or password hasn't been filled"));
    }
    const username = req.body.username;
    const user = await findUserByUsername(username);
    if (user) {
      res.status(409).json({
        message: "Username already taken",
      });
    }
    const password = req.body.password;
    const hashed = (
      await bcrypt.hash(password, Number(process.env.SALT_NUMBER) || 10)
    ).toString();
    await createUser(username, hashed);
    res.send("ok");
  } catch (err) {
    next(err);
  }
}
