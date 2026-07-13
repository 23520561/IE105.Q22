import type { Request } from "express";

export type usersRequest = Request<{}, {}, { id: string }>;
export type usersResponse = {
  id: string;
};
