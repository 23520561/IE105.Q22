import { useState } from "react";
import { login, signup } from "./api";

const LoginSignupModal = function ({
  closeHandler,
}: {
  closeHandler: VoidFunction;
}) {
  const [isLoginPage, setIsLoginPage] = useState(true);
  const [hidePassword, setHidePassword] = useState(true);
  const [password, setPassword] = useState<string | undefined>(undefined);
  const [username, setUsername] = useState<string | undefined>(undefined);
  const [error, setError] = useState<string | undefined>(undefined);
  async function submitHandler(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!username || !password) {
      setError("Please fill both username and password");
      return;
    }
    try {
      if (isLoginPage) {
        await login({ username: username, password: password });
        closeHandler();
      } else {
        await signup({ username: username, password: password });
      }
    } catch (err) {
      setError("Incorrect username or password");
    }
  }
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-surface-container rounded-lg border border-outline-variant/10 shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden transition-all duration-500">
        <div className="pt-10 pb-6 px-4 text-center relative">
          <button
            className="text-on-surface-variant hover:text-on-surface transition-colors absolute right-5 top-5"
            onClick={() => closeHandler()}
          >
            <span className="material-symbols-outlined">close</span>
          </button>

          <div className="flex flex-col items-center gap-3">
            <div className="relative w-12 h-12 flex items-center justify-center">
              <img
                src="/android-chrome-192x192.png"
                alt="The Observational Engine logo"
                className="h-10 w-10"
              />
            </div>
            <div>
              <h1 className="font-headline font-extrabold text-2xl text-on-surface">
                The Observational Engine
              </h1>
            </div>
          </div>
        </div>

        <div className="px-10">
          <div className="flex bg-surface-container-low rounded p-1 mb-8">
            <button
              className={`flex-1 py-2 text-sm font-semibold rounded transition-all duration-200 ${isLoginPage ? "bg-surface-container-high text-primary shadow-sm" : "text-on-surface-variant hover:text-on-surface"}`}
              id="loginTab"
              onClick={() => {
                setIsLoginPage(true);
              }}
            >
              Login
            </button>
            <button
              className={`flex-1 py-2 text-sm font-semibold rounded transition-all duration-200 ${isLoginPage ? "text-on-surface-variant hover:text-on-surface" : "bg-surface-container-high text-primary shadow-sm"}`}
              id="signupTab"
              onClick={() => {
                setIsLoginPage(false);
                setError(undefined);
              }}
            >
              Signup
            </button>
          </div>
        </div>

        <div className="px-10 pb-10">
          <form className="space-y-5" id="authForm" onSubmit={submitHandler}>
            <div className="space-y-2">
              <label
                className="block text-xs font-bold uppercase tracking-widest text-on-surface-variant ml-1"
                htmlFor="email"
              >
                User Name <span className="text-error">*</span>
              </label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-outline text-lg">
                  label
                </span>
                <input
                  className="w-full bg-surface-container-lowest border border-outline-variant/30 rounded px-10 py-3 text-on-surface placeholder:text-outline focus:ring-1 focus:ring-primary focus:border-primary outline-none transition-all duration-200"
                  id="email"
                  placeholder="user123"
                  type="text"
                  value={username}
                  onChange={(e) => {
                    setUsername(e.target.value);
                  }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label
                  className="block text-xs font-bold uppercase tracking-widest text-on-surface-variant ml-1"
                  htmlFor="password"
                >
                  Password <span className="text-error">*</span>
                </label>
                <a
                  className={`${!isLoginPage && "invisible"} text-xs text-primary hover:text-primary-fixed-dim transition-colors`}
                  href="#"
                  id="forgotLink"
                >
                  Forgot password?
                </a>
              </div>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-outline text-lg">
                  lock
                </span>
                <input
                  className="w-full bg-surface-container-lowest border border-outline-variant/30 rounded px-10 py-3 text-on-surface placeholder:text-outline focus:ring-1 focus:ring-primary focus:border-primary outline-none transition-all duration-200"
                  id="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                  }}
                  type={hidePassword ? "password" : "text"}
                />
                <button
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-outline hover:text-on-surface transition-colors"
                  type="button"
                  onClick={() => setHidePassword(!hidePassword)}
                >
                  <span className="material-symbols-outlined text-lg">
                    visibility
                  </span>
                </button>
              </div>
            </div>
            <p className={`${error ?? "invisible"} text-xs text-error `}>
              {error ?? "Error placeholder"}
            </p>

            <button
              className="w-full py-3.5  bg-primary text-on-primary font-bold rounded shadow-lg shadow-primary/10 hover:scale-95 active:scale-[0.98] transition-all duration-200"
              id="primaryCta"
            >
              Sign In
            </button>

            <div className="relative flex items-center py-4">
              <div className="grow border-t border-outline-variant/10"></div>
              <span className="shrink mx-4 text-[10px] font-bold text-outline uppercase tracking-widest">
                Or continue with
              </span>
              <div className="grow border-t border-outline-variant/10"></div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <button className="flex items-center justify-center gap-2 py-2.5 bg-surface-container-high border border-outline-variant/20 rounded hover:bg-surface-bright transition-colors text-sm font-semibold">
                <svg className="w-4 h-4" viewBox="0 0 24 24">
                  <path
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    fill="currentColor"
                  ></path>
                  <path
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    fill="currentColor"
                  ></path>
                  <path
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.26s.81 1.37 1.81 2.42z"
                    fill="currentColor"
                  ></path>
                  <path
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.66l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    fill="currentColor"
                  ></path>
                </svg>
                Google
              </button>
              <button className="flex items-center justify-center gap-2 py-2.5 bg-surface-container-high border border-outline-variant/20 rounded hover:bg-surface-bright transition-colors text-sm font-semibold">
                <svg
                  className="w-4 h-4"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.43.372.823 1.102.823 2.222 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"></path>
                </svg>
                GitHub
              </button>
            </div>

            <p
              className={`${isLoginPage && "invisible"} text-center text-[11px] leading-relaxed text-on-surface-variant px-2 max-w-[50ch]`}
              id="disclaimer"
            >
              By signing up, you agree to our{" "}
              <a className="underline hover:text-primary" href="#">
                Terms of Service
              </a>{" "}
              and{" "}
              <a className="underline hover:text-primary" href="#">
                Privacy Policy
              </a>
              . Synthetix Lab uses cookies for essential authentication only.
            </p>
          </form>
        </div>
      </div>
    </div>
  );
};
export default LoginSignupModal;
