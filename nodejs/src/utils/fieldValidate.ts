import type { NextFunction, Request, Response } from "express";

export function validateField(...args: string[]) {
  return function (req: Request, res: Response, next: NextFunction) {
    for (const field of args) {
      if (!req.body[field]) {
        res.status(400).json({
          message: `The ${field} field in the request is required`,
        });
      }
    }
    next();
  };
}
