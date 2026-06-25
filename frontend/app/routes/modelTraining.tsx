import { useEffect, useState } from "react";
import { useParams } from "react-router";
import { useDataset } from "~/customHooks/useDataset";
import {
  addNode,
  getTree,
  type DecisionTreeNodeVisualization,
} from "~/modelTraining/api";
import NodeInsights from "~/modelTraining/NodeInsights";
import PlayGround from "~/modelTraining/playground";

const modelTraining = function () {
  const datasetId = useParams()?.datasetId ?? "";
  const { info, chooseFeatureHandler } = useDataset(datasetId);
  const [visualNodes, setVisualNodes] = useState<
    DecisionTreeNodeVisualization[]
  >([]);
  const selectedNode = visualNodes.find((n) => n.selected) || null;
  const NodeWidth = 180,
    NodeHeight = 100;
  function setSelectedNode(nodeId: string | null) {
    setVisualNodes(
      visualNodes.map((node) => {
        return {
          ...node,
          selected: node.id === nodeId,
        };
      }),
    );
  }
  async function addSplitNode(id: string) {
    const newNodes = await addNode(id);
    if (!newNodes) {
      return;
    }
    const newSplit: DecisionTreeNodeVisualization = newNodes[0];
    const one: DecisionTreeNodeVisualization = newNodes[1];
    const two: DecisionTreeNodeVisualization = newNodes[2];
    const oldDatasetIndex: number = visualNodes.findIndex((n) => n.id === id);
    const parent = visualNodes[oldDatasetIndex];
    if (!parent) {
      throw new Error(`Id does not exist ${id}`);
    }
    const pI = visualNodes.findIndex(
      (n) => n.y > parent.y && n.x >= parent.x + NodeWidth / 2,
    );
    visualNodes[oldDatasetIndex] = newSplit;
    if (pI == -1) {
      visualNodes.push(one, two);
    } else {
      visualNodes.splice(pI, 0, one, two);
    }
    setVisualNodes([...visualNodes]);
  }

  useEffect(() => {
    const fetchData = async () => {
      const nodes = await getTree(datasetId);
      console.log(nodes);
      if (!nodes) {
        return;
      }
      setVisualNodes(
        nodes.map((n) => ({
          ...n,
          x: 0,
          y: 0,
          selected: false,
        })),
      );
    };
    fetchData();
  }, []);
  return (
    <div className="flex-1 relative overflow-hidden canvas-bg bg-surface-dim">
      <div className="absolute top-6 left-6 z-10 flex items-center gap-4">
        <div className="bg-surface-container-high px-4 py-2 rounded-xl flex items-center gap-3 border border-outline-variant/10 shadow-lg">
          <h1 className="font-headline font-bold text-lg text-on-surface">
            Decision Tree Explorer
          </h1>
          <div className="h-4 w-px bg-outline-variant/30"></div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-on-surface-variant font-medium">
              Depth: 4
            </span>
            <span className="text-xs text-on-surface-variant font-medium">
              Nodes: 15
            </span>
          </div>
        </div>
      </div>
      <div className="flex h-full">
        <div className="flex-1 min-w-0">
          {visualNodes.length > 0 && (
            <PlayGround
              selectedNode={selectedNode}
              visualNodes={visualNodes}
              setSelectedNode={setSelectedNode}
              nodeWidth={NodeWidth}
              nodeHeight={NodeHeight}
            />
          )}
        </div>

        {selectedNode && (
          <NodeInsights
            totalSample={info.shape[0]}
            node={selectedNode}
            setSelectedNode={setSelectedNode}
            addSplitNode={addSplitNode}
          />
        )}
      </div>{" "}
    </div>
  );
};
export default modelTraining;
