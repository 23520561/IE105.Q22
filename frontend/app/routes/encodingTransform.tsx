import { useParams } from "react-router";
import type { Route } from "../+types/root";
import { useDataset } from "~/customHooks/useDataset";
import FeatureSelection from "~/eda/FeatureSelection";
import { createTypeList } from "~/eda/charts/helper";
import Encoding from "~/enTra/Encoding";
import {
  EncodingRequest,
  enconding,
  type EncodingMethodType,
} from "~/enTra/api";
import { type PipelineStepType } from "~/pipeline/PipelineStepType";
import EdaCarousel from "~/eda/EdaCarousel";
import { PipelineList } from "~/pipeline/PipelineList";
import Transforming from "~/enTra/Transforming";
import { usePipeline } from "~/customHooks/usePipeline";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Encoding and Transform" }];
}
const EncodingTransform = function () {
  const datasetId = useParams()?.datasetId ?? "";
  const { info, chooseFeatureHandler } = useDataset(datasetId);
  const { pipeline, refreshHandler, deleteHandler } = usePipeline(datasetId);
  const typeList = createTypeList(info.columns);
  const selectedColumn =
    Object.keys(info.columns).find(
      (key) => info?.columns[key].selected === true,
    ) ?? "";
  async function encodingHandler(method: EncodingMethodType, i: number) {
    const req = new EncodingRequest(method, {
      columns: Object.keys(info.columns),
      column: selectedColumn,
    });
    if (i !== -1) {
      deleteHandler(i);
    } else {
      try {
        await enconding(datasetId, req);
        refreshHandler();
      } catch {
        console.log("Error");
      }
    }
  }

  return (
    <>
      <main className="flex-1 relative overflow-hidden flex flex-col bg-surface-dim">
        <header className="h-16 flex items-center justify-between px-8 bg-surface-container-low shrink-0">
          <div className="flex items-center gap-4">
            <button className="p-2 hover:bg-surface-variant/40 rounded-full transition-colors">
              <span className="material-symbols-outlined text-on-surface-variant">
                arrow_back
              </span>
            </button>
            <div>
              <h1 className="font-headline text-lg font-bold text-white leading-tight">
                Encoding &amp; Transformation
              </h1>
              <p className="text-xs text-on-surface-variant font-medium tracking-tight">
                Step 04 • Preprocessing Pipeline
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <button className="px-4 py-1.5 rounded-md border border-outline-variant/20 text-on-surface-variant text-xs font-semibold hover:bg-surface-container-high transition-colors">
              Discard Changes
            </button>
            <button className="px-6 py-1.5 rounded-md bg-primary text-on-primary text-xs font-bold hover:bg-surface-tint transition-colors">
              Save Config
            </button>
          </div>
        </header>

        <div className="flex-1 p-8 grid grid-cols-12 gap-6 overflow-y-auto custom-scrollbar">
          <section className="col-span-12 lg:col-span-3 flex flex-col gap-4">
            <FeatureSelection
              typeLists={typeList}
              selectedFeature={selectedColumn}
              chooseHandler={chooseFeatureHandler}
            ></FeatureSelection>
            <PipelineList
              pipeline={pipeline}
              deleteHandler={deleteHandler}
            ></PipelineList>
          </section>

          <div className="col-span-12 lg:col-span-9 flex flex-col gap-6">
            <div className="grid grid-rows-1 md:grid-rows-2 gap-6">
              <Encoding
                encodingHandler={encodingHandler}
                selectedFeature={selectedColumn}
                pipeline={pipeline}
              ></Encoding>

              <Transforming
                datasetId={datasetId}
                selectedColumns={Object.keys(info.columns)}
                refresh={refreshHandler}
              ></Transforming>
            </div>
          </div>
          <section className="col-span-12">
            <EdaCarousel datasetId={datasetId}></EdaCarousel>{" "}
          </section>
        </div>
      </main>
    </>
  );
};
export default EncodingTransform;
