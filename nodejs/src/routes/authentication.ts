import { Router, type Request, type Response } from "express";
import {
  isAuthenticated,
  login,
  logout,
  signup,
} from "~/controllers/authentication.js";

const authenRouter = Router();

authenRouter.get("/login", (req: Request, res: Response) => {
  res.send(`
    <form action="/login" method="POST">
      <input
        type="text"
        name="username"
        placeholder="Username"
        required
      />

      <input
        type="password"
        name="password"
        placeholder="Password"
        required
      />

      <button type="submit">Login</button>
    </form>
  `);
});
authenRouter.post("/login", login, (req, res) => {
  res.json("ok");
});
authenRouter.get("/logout", isAuthenticated, logout);
authenRouter.post("/signup", signup);

export default authenRouter;
