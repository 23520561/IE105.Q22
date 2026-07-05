import { useState } from "react";
import { createProjectName, type projectRequestType } from "./api";
import { useNavigate } from "react-router";

const CreateProjectModal = function ({
  datasetId,
  closeHandler,
}: {
  datasetId: string;
  closeHandler: VoidFunction;
}) {
  const [name, setName] = useState("");
  const navigate = useNavigate();
  async function createHandler() {
    const req: projectRequestType = { name: name, dataset_id: datasetId };
    const id = await createProjectName(req);
    navigate(`/feature-selection/${id}`);
  }
  return (
    <div className="fixed inset-0 bg-surface-container-lowest/80 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-surface-container w-full max-w-md border border-outline-variant shadow-2xl overflow-hidden rounded">
        <div className="px-6 py-4 border-b border-outline-variant flex items-center justify-between bg-surface-container-high">
          <h2 className="text-xl font-headline font-bold text-on-surface">
            Create New Project
          </h2>
          <button
            className="text-on-surface-variant hover:text-on-surface transition-colors"
            onClick={() => closeHandler()}
          >
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        <div className="p-6 space-y-6">
          <div className="space-y-2">
            <label
              className="block text-sm font-medium text-on-surface-variant"
              htmlFor="project-name"
            >
              Project Name
            </label>
            <input
              className="w-full bg-surface-container-low border border-outline-variant text-on-surface rounded px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all placeholder:text-outline"
              id="project-name"
              placeholder="Enter project name..."
              value={name}
              onChange={(e) => {
                setName(e.target.value);
              }}
              type="text"
            />
          </div>
        </div>

        <div className="px-6 py-4 bg-surface-container-high border-t border-outline-variant flex items-center justify-end gap-3">
          <button
            className="px-5 py-2 text-sm font-bold text-on-surface-variant hover:text-on-surface transition-colors rounded"
            onClick={() => closeHandler()}
          >
            Cancel
          </button>
          <button
            className="px-5 py-2 text-sm font-bold bg-primary text-on-primary-fixed hover:bg-primary/90 transition-all shadow-lg rounded"
            onClick={() => createHandler()}
          >
            Create Project
          </button>
        </div>
      </div>
    </div>
  );
};
export default CreateProjectModal;
