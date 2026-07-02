import { EncodingMethod } from "./api";
import type { PipelineStepType } from "../pipeline/PipelineStepType";
import type { EncodingMethodType } from "./api";
import MoreInfo from "~/components/MoreInfo";
const Encoding = function ({
  encodingHandler,
  pipeline,
  selectedFeature,
  uniqueCount,
}: {
  encodingHandler: (method: EncodingMethodType, i: number) => void;
  pipeline: PipelineStepType[];
  selectedFeature: string;
  uniqueCount: number;
}) {
  return (
    <div className="p-6 bg-surface-container rounded-xl border-t-2 border-primary/20">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
          <span className="material-symbols-outlined text-primary text-xl">
            dataset
          </span>
        </div>
        <div>
          <h3 className="text-sm font-bold text-white">Categorical Encoding</h3>
          <p className="text-[11px] text-on-surface-variant">
            Convert strings to machine-readable vectors
          </p>
        </div>
      </div>
      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">
            Encoding Method
          </label>
          <div className="grid grid-cols-3 gap-2">
            {EncodingMethod.map((method) => {
              const index = pipeline.findIndex(
                (step) =>
                  step.type === "encoding" &&
                  step.data.column === selectedFeature &&
                  step.data.method === method.name.toLowerCase(),
              );
              const disabled = method.name === "one_hot" && uniqueCount > 5;
              return (
                <>
                  <button
                    className={`px-3 py-2 text-[10px] font-bold rounded border border-outline-variant/10 transition-colors ${index !== -1 ? "bg-primary text-on-primary border-primary" : `${disabled ? "bg-error text-on-error" : "bg-surface-container-lowest text-on-surface-variant"}  hover:bg-surface-variant/40`} relative group`}
                    onClick={() => encodingHandler(method.name, index)}
                    disabled={disabled}
                  >
                    <MoreInfo message={method.description}></MoreInfo>{" "}
                    {method.name}
                  </button>
                </>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
export default Encoding;
