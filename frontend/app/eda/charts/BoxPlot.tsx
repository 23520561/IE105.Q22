import * as echarts from "echarts";
import { useEffect, useRef, useState } from "react";
import { formatBoxPlot, getCssVar } from "./helper";
import { initChartStyle } from "./styles";
import { getBoxPlot } from "./api";
type boxPlotVisual = {
  columnNames: string[];
  echartBoxes: number[][];
  echartOutliners: number[][];
};
const BoxPlot = function ({
  subset,
  datasetId,
}: {
  subset: string[] | undefined;
  datasetId: string | undefined;
}) {
  const chartRef = useRef(null);
  const [boxPlot, setBoxPlot] = useState<boxPlotVisual | null>(null);
  useEffect(() => {
    async function fetchData() {
      if (!subset || !datasetId) {
        return;
      }
      const data = await getBoxPlot(subset, datasetId);
      if (data) {
        setBoxPlot(formatBoxPlot(data));
      }
    }
    fetchData();
  }, [datasetId, subset]);
  useEffect(() => {
    const chartStyle = initChartStyle();
    let option = {
      tooltip: {
        position: "top",
      },
      grid: {
        height: "70%",
        top: 0,
        left: 0,
        right: 0,
      },
      xAxis: {
        type: "category",
        data: boxPlot?.columnNames,
        axisLabel: {
          color: chartStyle.fontColor,
          fontFamily: chartStyle.fontFamily,
          fontSize: chartStyle.fontSize,
        },
      },
      yAxis: {
        type: "value",
        inverse: true,
        axisLabel: {
          color: chartStyle.fontColor,
          fontFamily: chartStyle.fontFamily,
          fontSize: chartStyle.fontSize,
        },
      },
      // visualMap: {
      //   min: 0,
      //   max: 100,
      //   calculable: true,
      //   orient: "horizontal",
      //   left: "center",
      //   bottom: "10%",
      //   inRange: {
      //     color: chartStyle.itemColor,
      //   },
      //   textStyle: {
      //     color: chartStyle.fontColor,
      //     fontFamily: chartStyle.fontFamily,
      //     fontSize: chartStyle.fontSize,
      //   },
      // },
      series: [
        {
          name: "boxplot",
          type: "boxplot",
          data: boxPlot?.echartBoxes,
          label: {
            show: true,
            textBorderColor: getCssVar("--color-surface-container-low"),
            textBorderWidth: 2,
            fontFamily: chartStyle.fontFamily,
            fontSize: chartStyle.fontSize,
          },
          itemStyle: {
            color: getCssVar("--color-surface-container-low"),
            borderColor: chartStyle.itemColor[0],
            borderWidth: 5,
            borderRadius: 5,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
        {
          name: "outliner",
          type: "scatter",
          data: boxPlot?.echartOutliners,
          itemStyle: {
            color: chartStyle.itemColor[1],
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
      ],
    };
    const chart = echarts.init(chartRef.current);
    chart.setOption(option);
    return () => {
      chart.dispose();
    };
  }, [boxPlot]);
  return <div ref={chartRef} style={{ height: "400px", width: "100%" }}></div>;
};
export default BoxPlot;
