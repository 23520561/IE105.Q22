import express, { type Request, type Response } from "express";
import session from "express-session";
import connectPg from "connect-pg-simple";
import pool from "./db/pool.js";
import cors from "cors";
import authenRouter from "./routes/authentication.js";
import passport from "passport";
import "./config/passport.js";
import projectRoute from "./routes/project.js";
import { apiUrl, deleteData, getData, postData } from "./api.js";
import { findProjectByIdAndUserId } from "./db/project.js";
import { isAuthenticated } from "./controllers/authentication.js";
import datasetRoute from "./routes/prebuiltdataset.js";
import usersRoute from "./routes/users.js";

const pgStore = connectPg(session);

const app = express();
app.use(
  cors({
    origin: process.env.FRONTEND_API,
    credentials: true,
  }),
);
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(
  session({
    secret: process.env.SESSION_SECRET || "random bullshit go",
    resave: false,
    saveUninitialized: false,
    store: new pgStore({
      pool,
      tableName: "session",
      createTableIfMissing: true,
    }),
  }),
);

app.use(passport.session());
app.use("/dataset", datasetRoute);
app.use("/project", projectRoute);
app.use("/user", usersRoute);
app.use(authenRouter);

app.post(
  "/model/decision-tree/split",
  isAuthenticated,
  async (req: Request, res: Response) => {
    res.json(
      await postData(apiUrl + req.originalUrl, req.body)
    );
  },
);

app.delete(
  "/tree",
  isAuthenticated,
  async (req: Request, res: Response) => {
    res.json(await deleteData(apiUrl + req.originalUrl));
  },
);
app.get(
  "/*splat",
  isAuthenticated,
  async (req: Request, res: Response) => {
    if (!req.user) {
      throw new Error("Authentication not working");
    }
    const id = req.query.dataset_id?.toString();
    if (!id) {
      res.json(await getData((apiUrl + req.originalUrl)));
      return;
    }
    const filename = (await findProjectByIdAndUserId(id, req.user.id)).filename;
    res.json(await getData((apiUrl + req.originalUrl).replace(id, filename)));
  },
);
app.delete(
  "/*splat",
  isAuthenticated,
  async (req: Request, res: Response) => {
    if (!req.user) {
      throw new Error("Authentication not working");
    }
    const id = req.query.dataset_id?.toString();
    if (!id) {
      console.log(req.originalUrl);
      throw new Error("No params");
    }
    const filename = (await findProjectByIdAndUserId(id, req.user.id)).filename;
    res.json(await deleteData((apiUrl + req.originalUrl).replace(id, filename)));
  },
);
app.post(
  "/*splat",
  isAuthenticated,
  async (req: Request, res: Response) => {
    if (!req.user) {
      throw new Error("Authentication not working");
    }
    const id = req.query.dataset_id?.toString();
    if (!id) {
      console.log(req.query);
      throw new Error("No params");
    }
    const filename = (await findProjectByIdAndUserId(id, req.user.id)).filename;

    res.json(
      await postData((apiUrl + req.originalUrl).replace(id, filename), req.body),
    );
  },
);

const PORT = process.env.PORT;
app.listen(PORT, (err) => {
  if (err) {
    throw err;
  }
  console.log("App is listening on port:", PORT);
});
