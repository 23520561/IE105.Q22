import { apiUrl, postData } from "~/api";

export type authenticateRequest = {
  username: string;
  password: string;
};
export async function login(req: authenticateRequest) {
  const prefix = "/login";
  return await postData(apiUrl + prefix, req);
}
export async function signup(req: authenticateRequest) {
  const prefix = "/signup";
  return await postData(apiUrl + prefix, req);
}
