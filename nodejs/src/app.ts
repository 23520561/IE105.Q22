import express, { type Request, type Response } from "express";
import session from "express-session";
import connectPg from "connect-pg-simple";
import pool from "./db/pool.js";
import cors from "cors";
import authenRouter from "./routes/authentication.js";
import passport from "passport";
import "./config/passport.js";
import projectRoute from "./routes/project.js";
import { apiUrl, getData, postData } from "./api.js";
import { findProjectByIdAndUserId } from "./db/project.js";
import { isAuthenticated } from "./controllers/authentication.js";
import datasetRoute from "./routes/prebuiltdataset.js";

const pgStore = connectPg(session);

const app = express();
app.use(
  cors({
    origin: "http://localhost:5173",
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
app.use(authenRouter);
app.get("/*splat", isAuthenticated, async (req: Request, res: Response) => {
  if (!req.user) {
    throw new Error("Authentication not working");
  }
  res.json(await getData(apiUrl + req.url));
});
app.post("/", isAuthenticated, async (req: Request, res: Response) => {
  if (!req.user) {
    throw new Error("Authentication not working");
  }
  const id = Array.isArray(req.params.id) ? req.params.id[0] : req.params.id;
  if (!id) {
    throw new Error("No params");
  }
  const filename = (await findProjectByIdAndUserId(id, req.user.id)).filename;

  res.json(postData((apiUrl + req.url).replace(id, filename), req.body));
});
const PORT = process.env.PORT;
app.listen(PORT, (err) => {
  if (err) {
    throw err;
  }
  console.log("App is listening on port:", PORT);
});
