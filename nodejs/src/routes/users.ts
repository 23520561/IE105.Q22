import { Router } from "express";
import { isAuthenticated } from "~/controllers/authentication.js";
import { getUserProfile } from "~/controllers/users.js";

const usersRoute = Router();
usersRoute.get("/profile", isAuthenticated, getUserProfile);

export default usersRoute;
