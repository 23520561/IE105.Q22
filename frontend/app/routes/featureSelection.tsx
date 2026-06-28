import { useState } from "react";
import { useParams } from "react-router";
import HeaderPreprocessing from "~/components/HeaderPreprocessing";
import { useDataset } from "~/customHooks/useDataset";
import { getRfe, RfeRequest, type RfeResponse } from "~/featureSelection/api";
import RankChart from "~/featureSelection/RankingChart";

const FeatureEngineer = function () {
  const datasetId = useParams()?.datasetId ?? "";
  const { info } = useDataset(datasetId);
  const [req, setReq] = useState<RfeRequest>(new RfeRequest(info.shape[1] - 1));
  const [rfe, setRfe] = useState<RfeResponse | null>(null);
  async function executeHandler() {
    if (req) {
      const data = await getRfe(datasetId, req);
      if (data) {
        setRfe(data);
      }
    }
  }

  return (
    <main className="flex-1 relative overflow-hidden flex flex-col bg-surface-dim p-8">
      <HeaderPreprocessing
        title="Feature Selection Engine"
        desc="Optimize model performance by identifying high-variance dimensional significance."
        stepNumber={0}
        nextStep={`/encode&transform/${datasetId}`}
      ></HeaderPreprocessing>

      <div className="flex-1 overflow-y-auto px-8 pb-12">
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <section className="bg-surface-container-low rounded-xl p-6 border border-outline-variant/5">
              <h3 className="text-xs font-bold uppercase tracking-widest text-primary mb-4">
                Method Selection
              </h3>
              <div className="grid gap-3">
                <label className="cursor-pointer">
                  <input
                    checked={false}
                    className="hidden peer"
                    name="method"
                    type="radio"
                  />
                  <div className="p-3 border border-outline-variant/20 rounded-lg peer-checked:border-primary peer-checked:bg-primary-container/20 transition-all text-center">
                    <p className="text-sm font-bold">RFE</p>
                    <p className="text-[10px] text-on-surface-variant">
                      Recursive Elimination
                    </p>
                  </div>
                </label>
              </div>
            </section>

            <section className="bg-surface-container-low rounded-xl p-6 border border-outline-variant/5">
              <h3 className="text-xs font-bold uppercase tracking-widest text-primary mb-4">
                Algorithm Parameters
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-on-surface-variant mb-2">
                    Target Number of Features
                  </label>
                  <input
                    className="w-full bg-surface-container-lowest border border-outline-variant/30 rounded-lg px-4 py-2 text-sm focus:ring-1 focus:ring-primary focus:border-primary outline-none transition-all"
                    type="number"
                    min={0}
                    max={info.shape[1] - 1}
                    value={req.numberFeature}
                    onChange={(e) =>
                      setReq(new RfeRequest(Number(e.target.value)))
                    }
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-on-surface-variant mb-2">
                    Step Size
                  </label>
                  <input
                    className="w-full bg-surface-container-lowest border border-outline-variant/30 rounded-lg px-4 py-2 text-sm focus:ring-1 focus:ring-primary focus:border-primary outline-none transition-all"
                    type="number"
                    value="1"
                  />
                </div>
                <div className="pt-4 border-t border-outline-variant/10">
                  <label className="block text-xs font-bold text-primary mb-2">
                    Estimator Base
                  </label>
                  <div className="flex items-center gap-3 p-3 bg-surface-container rounded-lg border border-outline-variant/10">
                    <span className="material-symbols-outlined text-secondary">
                      functions
                    </span>
                    <span className="text-sm font-mono">
                      DecisionTreeclassNameifier(random_state=42)
                    </span>
                  </div>
                </div>
                <button
                  className="px-5 py-2 rounded-lg btn-gradient text-on-primary bg-primary font-bold shadow-lg shadow-primary/20 flex items-center gap-2 transition-transform active:scale-95 hover:scale-95"
                  onClick={() => executeHandler()}
                >
                  <span className="material-symbols-outlined text-lg">
                    rocket_launch
                  </span>
                  Run Feature Selection
                </button>
              </div>
            </section>
          </div>

          <div className="col-span-12 lg:col-span-8 space-y-6">
            <section className="bg-surface-container-low rounded-xl p-6 border border-outline-variant/5">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xs font-bold uppercase tracking-widest text-primary">
                  Feature Importance Ranking
                </h3>
              </div>
              {rfe && (
                <RankChart
                  data={Object.entries(rfe.feature_importances)}
                  max={Math.max(...Object.values(rfe.feature_importances))}
                ></RankChart>
              )}
            </section>

            <section className="bg-surface-container-low rounded-xl border border-outline-variant/5 overflow-hidden">
              <div className="px-6 py-4 border-b border-outline-variant/10 flex justify-between items-center">
                <h3 className="text-xs font-bold uppercase tracking-widest text-primary">
                  Dataset Columns Analysis
                </h3>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-sm text-on-surface-variant">
                    search
                  </span>
                  <input
                    className="bg-surface-container-lowest border-none rounded-full pl-9 pr-4 py-1.5 text-xs text-on-surface w-48 focus:ring-1 focus:ring-primary outline-none"
                    placeholder="Filter features..."
                    type="text"
                  />
                </div>
              </div>
              <div className="overflow-x-auto">
                {rfe ? (
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-surface-container-highest/30">
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider">
                          Feature Name
                        </th>
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider text-center">
                          RFE Ranking
                        </th>
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider text-right">
                          Importance
                        </th>
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider text-right">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-outline-variant/5">
                      {Object.keys(info.columns).map(
                        (val, i) =>
                          i < info.shape[1] - 1 && (
                            <tr className="hover:bg-surface-bright/20 transition-colors group">
                              <td className="px-6 py-4">
                                <div className="flex items-center gap-3">
                                  <span className="material-symbols-outlined text-primary text-sm">
                                    key
                                  </span>
                                  <span className="text-sm font-mono font-medium">
                                    {val}
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4">
                                <span className="px-2 py-1 rounded bg-primary/10 text-primary text-[10px] font-bold uppercase tracking-tighter">
                                  {rfe.recommended_to_keep.some(
                                    (f) => f === val,
                                  )
                                    ? "Recommended"
                                    : "Not Recommended"}
                                </span>
                              </td>
                              <td className="px-6 py-4 text-center font-mono text-sm text-on-surface">
                                {rfe.feature_ranking[val] ?? "No Data"}
                              </td>
                              <td className="px-6 py-4 text-right font-mono text-sm text-[#7bd0ff]">
                                {rfe.feature_importances[val] ?? "No Data"}
                              </td>
                              <td className="px-6 py-4 text-right">
                                <button className="material-symbols-outlined text-on-surface-variant hover:text-white transition-colors">
                                  more_vert
                                </button>
                              </td>
                            </tr>
                          ),
                      )}
                    </tbody>
                  </table>
                ) : (
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-surface-container-highest/30">
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider">
                          Feature Name
                        </th>
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider text-center">
                          RFE Ranking
                        </th>
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider text-right">
                          Importance
                        </th>
                        <th className="px-6 py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-wider text-right">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-outline-variant/5">
                      <tr className="hover:bg-surface-bright/20 transition-colors group">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <span className="material-symbols-outlined text-primary text-sm">
                              key
                            </span>
                            <span className="text-sm font-mono font-medium">
                              user_id_hash
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-2 py-1 rounded bg-primary/10 text-primary text-[10px] font-bold uppercase tracking-tighter">
                            Recommended
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center font-mono text-sm text-on-surface">
                          1
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-sm text-[#7bd0ff]">
                          0.982
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button className="material-symbols-outlined text-on-surface-variant hover:text-white transition-colors">
                            more_vert
                          </button>
                        </td>
                      </tr>
                      <tr className="hover:bg-surface-bright/20 transition-colors group">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <span className="material-symbols-outlined text-primary text-sm">
                              show_chart
                            </span>
                            <span className="text-sm font-mono font-medium">
                              daily_active_mins
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-2 py-1 rounded bg-primary/10 text-primary text-[10px] font-bold uppercase tracking-tighter">
                            Recommended
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center font-mono text-sm text-on-surface">
                          1
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-sm text-[#7bd0ff]">
                          0.845
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button className="material-symbols-outlined text-on-surface-variant hover:text-white transition-colors">
                            more_vert
                          </button>
                        </td>
                      </tr>
                      <tr className="hover:bg-surface-bright/20 transition-colors group">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <span className="material-symbols-outlined text-primary text-sm">
                              database
                            </span>
                            <span className="text-sm font-mono font-medium">
                              session_depth_idx
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-2 py-1 rounded bg-primary/10 text-primary text-[10px] font-bold uppercase tracking-tighter">
                            Recommended
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center font-mono text-sm text-on-surface">
                          2
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-sm text-[#7bd0ff]">
                          0.712
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button className="material-symbols-outlined text-on-surface-variant hover:text-white transition-colors">
                            more_vert
                          </button>
                        </td>
                      </tr>
                      <tr className="hover:bg-surface-bright/20 transition-colors group opacity-60">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <span className="material-symbols-outlined text-outline-variant text-sm">
                              block
                            </span>
                            <span className="text-sm font-mono font-medium">
                              last_login_timestamp
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-2 py-1 rounded bg-surface-container-highest text-on-surface-variant text-[10px] font-bold uppercase tracking-tighter">
                            Exclude
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center font-mono text-sm text-on-surface-variant">
                          14
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-sm text-on-surface-variant">
                          0.021
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button className="material-symbols-outlined text-on-surface-variant hover:text-white transition-colors">
                            more_vert
                          </button>
                        </td>
                      </tr>
                      <tr className="hover:bg-surface-bright/20 transition-colors group opacity-60">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <span className="material-symbols-outlined text-outline-variant text-sm">
                              block
                            </span>
                            <span className="text-sm font-mono font-medium">
                              device_os_version
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-2 py-1 rounded bg-surface-container-highest text-on-surface-variant text-[10px] font-bold uppercase tracking-tighter">
                            Exclude
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center font-mono text-sm text-on-surface-variant">
                          18
                        </td>
                        <td className="px-6 py-4 text-right font-mono text-sm text-on-surface-variant">
                          0.004
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button className="material-symbols-outlined text-on-surface-variant hover:text-white transition-colors">
                            more_vert
                          </button>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                )}
              </div>
            </section>

            <div className="flex justify-between items-center bg-surface-container-low rounded-xl p-6 border border-outline-variant/10">
              <div className="flex items-center gap-4">
                <div className="bg-primary/10 px-3 py-1.5 rounded-lg border border-primary/20">
                  <p className="text-[10px] text-primary uppercase font-bold tracking-widest">
                    Configuration Summary
                  </p>
                  <p className="text-xs text-on-surface">
                    RFE • DecisionTreeclassNameifier • {req.numberFeature}{" "}
                    Features Target
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
};
export default FeatureEngineer;
