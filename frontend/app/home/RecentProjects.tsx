import { useEffect, useState } from "react";
import { getProjects, type projectResponseType } from "./api";
import { useNavigate } from "react-router";
const RecentProjects = function () {
  const [projects, setProjects] = useState<projectResponseType[]>();
  const navigate = useNavigate();
  const openProjectHandler = function (id: string) {
    navigate(`/feature-selection/${id}`);
  };
  useEffect(() => {
    const fetchData = async () => {
      const data = await getProjects();
      if (!data) {
        return;
      }
      setProjects(data);
    };
    fetchData();
  }, []);
  return (
    <>
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-headline font-bold text-white tracking-tight">
          Recent Projects
        </h2>
        <button className="text-xs text-secondary hover:text-white transition-colors">
          View Workspace
        </button>
      </div>
      <div className="space-y-4">
        {projects &&
          projects.map((project) => {
            return (
              <div
                className="bg-surface-container rounded-lg p-5 flex items-center gap-6 hover:bg-surface-bright/40 transition-colors group"
                key={project.id}
                onClick={() => {
                  openProjectHandler(project.id);
                }}
              >
                <div className="w-14 h-14 bg-surface-container-highest rounded-lg flex items-center justify-center border border-white/5">
                  <span className="material-symbols-outlined text-slate-400 group-hover:text-primary transition-colors">
                    hub
                  </span>
                </div>
                <div className="grow">
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="text-on-surface font-bold text-sm">
                      {project.name}
                    </h4>
                    <span className="text-[10px] text-on-surface-variant font-medium tabular-nums">
                      {`Edited ${project.date}`}
                    </span>
                  </div>
                </div>
                <span className="material-symbols-outlined text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer">
                  more_vert
                </span>
              </div>
            );
          })}
      </div>
    </>
  );
};
export default RecentProjects;
