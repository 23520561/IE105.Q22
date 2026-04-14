import { uploadedDataset } from "~/seed";
import SearchBox from "~/components/SearchBox";
import { useState } from "react";
import UploadedModal from "./UploadedModal";
const UploadedDatasets = function () {
  const [modalOpened, setModalOpened] = useState(false);
  function onClose() {
    console.log("click");
    setModalOpened(false);
  }
  return (
    <>
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-headline font-bold text-white tracking-tight">
          Uploaded Datasets
        </h2>
        <span className="material-symbols-outlined text-primary cursor-pointer hover:scale-110 transition-transform">
          add_box
        </span>
      </div>
      <div className="bg-surface-container-low rounded-xl overflow-hidden border border-white/5">
        <div className="divide-y divide-white/5">
          <SearchBox placeholder="Search library..."></SearchBox>
          {uploadedDataset.map((dataset, i) => {
            const color = ["sky", "green", "slate"][i % 3];
            return (
              <div
                key={i}
                className="px-4 py-3 flex items-center justify-between hover:bg-surface-container-high transition-colors cursor-pointer group"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`material-symbols-outlined text-${color}-400 text-lg`}
                  >
                    {dataset.name.includes(".csv")
                      ? "table_chart"
                      : "description"}
                  </span>
                  <div>
                    <p className="text-xs font-medium text-on-surface">
                      {dataset.name}
                    </p>
                    <p className="text-[9px] text-on-surface-variant tabular-nums">
                      {`${dataset.size} • Modified ${dataset.dateModdified}`}
                    </p>
                  </div>
                </div>
                <span className="material-symbols-outlined text-slate-500 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                  download
                </span>
              </div>
            );
          })}
        </div>
        <div
          className="p-4 bg-surface-container-low"
          onClick={() => {
            setModalOpened(true);
          }}
        >
          <div className="border-2 border-dashed border-outline-variant/20 rounded-lg p-6 flex flex-col items-center justify-center gap-2 hover:border-primary/40 transition-colors cursor-pointer group">
            <span className="material-symbols-outlined text-slate-500 group-hover:text-primary transition-colors">
              cloud_upload
            </span>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest group-hover:text-white">
              Upload New Data
            </span>
          </div>
        </div>
        {modalOpened && <UploadedModal onClose={onClose}></UploadedModal>}{" "}
      </div>
    </>
  );
};
export default UploadedDatasets;
