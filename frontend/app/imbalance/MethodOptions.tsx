import { ImbalancedInfo, type ImbalancedMethod } from "./api";

const MethodOptions = function({
  method,
  methodChangeHandler,
}: {
  method: ImbalancedMethod;
  methodChangeHandler: (method: ImbalancedMethod) => void;
}) {
  return (
    <>
      <label className="text-xs font-bold uppercase tracking-widest text-on-surface-variant">
        Sampling Method
      </label>
      <div className="grid gap-4">
        {ImbalancedInfo.map((imbalance) => (
          <label key={imbalance.method} className="group cursor-pointer">
            <input
              onChange={() => methodChangeHandler(imbalance.method)}
              className="hidden peer"
              name="method"
              type="radio"
              checked={imbalance.method == method}
            />
            <div className="flex flex-col p-4 rounded-xl border border-outline-variant bg-surface-container-low peer-checked:bg-surface-container-highest peer-checked:border-primary transition-all group-hover:bg-surface-bright/20">
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-white font-headline">
                  {imbalance.name}
                </span>
              </div>
              <p className="text-xs text-on-surface-variant leading-relaxed">
                {imbalance.description}
              </p>
            </div>
          </label>
        ))}
      </div>
    </>
  );
};
export default MethodOptions;
