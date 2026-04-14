import { useCallback, useEffect, useRef, useState } from "react";
import { useRowsHook } from "./useRowsHook";

const RawDataInspector = function ({
  datasetId,
  columns,
  expand,
  expandHandler,
}: {
  datasetId: string;
  columns: string[];
  expand: boolean;
  expandHandler: () => boolean;
}) {
  const [newPage, setNewPage] = useState(0);
  const { rows, loading } = useRowsHook(expand, datasetId, newPage);
  const observer = useRef<IntersectionObserver>(null);
  const lastRowRef = useCallback(
    (e: HTMLTableRowElement) => {
      if (!expand || loading) {
        return;
      }
      if (observer.current) {
        observer.current.disconnect();
      }
      observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && rows.offset != rows.count) {
          setNewPage(newPage + 1);
        }
      });
      if (e) observer.current.observe(e);
    },
    [expand, loading],
  );
  return (
    <div className="bg-surface-container-low rounded-xl border border-white/5 overflow-hidden">
      <div className="px-6 py-4 flex justify-between items-center bg-surface-container-high/50">
        <h4 className="text-sm font-bold text-white">Raw Data Inspector</h4>
        <button>
          <span
            className="material-symbols-outlined text-xs hover:text-primary"
            onClick={() => {
              expandHandler();
            }}
          >
            {expand ? "expand_circle_down" : "expand_circle_up"}
          </span>
        </button>
        <div className="flex gap-4">
          <span className="text-[10px] text-on-surface-variant font-mono">
            {rows.rows.length} of {rows.count}
          </span>
          <div className="flex gap-1">
            <button className="w-6 h-6 flex items-center justify-center rounded bg-surface-variant/40 hover:bg-surface-variant">
              <span className="material-symbols-outlined text-xs">
                chevron_left
              </span>
            </button>
            <button className="w-6 h-6 flex items-center justify-center rounded bg-surface-variant/40 hover:bg-surface-variant">
              <span className="material-symbols-outlined text-xs">
                chevron_right
              </span>
            </button>
          </div>
        </div>
      </div>
      <div className="max-h-150 overflow-auto custom-scrollbar">
        <table className="w-full text-left border-collapse">
          <thead className="sticky top-0 bg-surface-container-low border-b border-white/5">
            <tr>
              {columns.map((e) => (
                <th
                  key={e}
                  className="sticky top-0 px-6 py-3 text-[10px] uppercase font-bold tracking-widest text-on-surface-variant"
                >
                  {e}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {rows.rows.map((row, i) =>
              i === rows.rows.length - 1 ? (
                <tr
                  ref={lastRowRef}
                  className="hover:bg-surface-bright/40 transition-colors"
                >
                  {row.map((data) =>
                    typeof data === "string" ? (
                      <td className="px-6 py-4 font-mono text-[11px] text-primary">
                        {data}
                      </td>
                    ) : (
                      <td className="px-6 py-4 tabular-nums text-[13px] text-white">
                        {data}
                      </td>
                    ),
                  )}
                </tr>
              ) : (
                <tr className="hover:bg-surface-bright/40 transition-colors">
                  {row.map((data) =>
                    typeof data === "string" ? (
                      <td className="px-6 py-4 font-mono text-[11px] text-primary">
                        {data}
                      </td>
                    ) : (
                      <td className="px-6 py-4 tabular-nums text-[13px] text-white">
                        {data}
                      </td>
                    ),
                  )}
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
export default RawDataInspector;
