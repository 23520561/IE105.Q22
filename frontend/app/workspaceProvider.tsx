import { createContext, useContext, useState, type ReactNode } from "react";
type SessionContextType = {
  workspace: string;
  setWorkspace: React.Dispatch<React.SetStateAction<string>>;
};

const SessionContext = createContext<SessionContextType | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [workspace, setWorkspace] = useState("");

  return (
    <SessionContext.Provider value={{ workspace, setWorkspace }}>
      {children}
    </SessionContext.Provider>
  );
}
export function useSession() {
  const context = useContext(SessionContext);

  if (!context) {
    throw new Error("useSession must be used inside SessionProvider");
  }

  return context;
}
