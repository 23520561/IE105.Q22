import { deleteStepPipeline, type PipelineResponseType } from "~/api";

export function PipelineList({
  pipeline,
  datasetId,
  refresh,
}: {
  pipeline: PipelineResponseType;
  datasetId: string;
  refresh: () => void; // re-fetch pipeline after delete
}) {
  const handleDelete = async (index: number) => {
    const res = await deleteStepPipeline(datasetId, index);
    if (res) {
      refresh();
    }
  };

  return (
    <div className="bg-surface-container rounded-xl p-6 border border-white/5 h-full overflow-scroll">
      <h4 className="text-xs font-bold text-white mb-4 flex items-center gap-2">
        <span className="material-symbols-outlined text-sm">account_tree</span>
        Pipeline
      </h4>

      <div className="space-y-2">
        {pipeline.length === 0 && (
          <div className="p-2 text-xs text-slate-400">No steps yet</div>
        )}

        {pipeline.map((step, i) => {
          const label =
            step.column ?? (step.columns ? step.columns.join(", ") : "unknown");

          return (
            <div
              key={i}
              className="flex items-center gap-3 p-2 rounded hover:bg-surface-variant/40 cursor-pointer group transition-colors"
            >
              {/* left dot */}
              <div className="w-3 h-3 rounded-full border border-outline shrink-0"></div>

              {/* step label */}
              <span
                className="text-xs text-slate-300 group-hover:text-white truncate"
                title={label}
              >
                {label}
              </span>

              {/* method badge */}
              <span className="ml-auto text-[10px] text-slate-500 font-mono">
                {step.method.toUpperCase()}
              </span>

              {/* delete button */}
              <button
                onClick={() => handleDelete(i)}
                className="ml-2 opacity-0 group-hover:opacity-100 transition text-red-400 hover:text-red-300"
              >
                <span className="material-symbols-outlined text-sm">
                  delete
                </span>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
