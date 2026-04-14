import { useEffect, useRef, useState } from "react";
import { getPcaChart, type pcaChartType } from "./api";
import { initChartStyle } from "./styles";
import * as echarts from "echarts";
import { getCssVar } from "./helper";

const PcaChart = function ({ datasetId }: { datasetId: string }) {
  const chartRef = useRef(null);
  const [pcaChart, setPcaChart] = useState<pcaChartType>({
    explained_variance: [0],
    points: [[0, 0, 0]],
    total_variance: 0,
  });
  useEffect(() => {
    async function fetchData() {
      if (datasetId === "") {
        return;
      }
      const data = await getPcaChart(datasetId);
      if (data) {
        setPcaChart(data);
      }
    }
    fetchData();
  }, [datasetId]);
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
        axisLabel: {
          color: chartStyle.fontColor,
          fontFamily: chartStyle.fontFamily,
          fontSize: chartStyle.fontSize,
        },
        axisLine: {
          lineStyle: {
            color: getCssVar("--color-error"),
          },
        },
      },
      yAxis: {
        axisLabel: {
          color: chartStyle.fontColor,
          fontFamily: chartStyle.fontFamily,
          fontSize: chartStyle.fontSize,
        },
        axisLine: {
          lineStyle: {
            color: getCssVar("--color-error"),
          },
        },
      },
      visualMap: {
        dimension: 2,
        calculable: true,
        orient: "horizontal",
        left: "center",
        max: Math.max(...pcaChart.points.map((p) => p[2])),
        min: Math.min(...pcaChart.points.map((p) => p[2])),
        inRange: {
          color: [getCssVar("--color-primary"), getCssVar("--color-error")],
        },
        textStyle: {
          color: chartStyle.fontColor,
          fontFamily: chartStyle.fontFamily,
          fontSize: chartStyle.fontSize,
        },
      },
      series: [
        {
          name: "punch card",
          type: "scatter",
          data: pcaChart.points,
          label: {
            color: chartStyle.fontColor,
            fontfamily: chartStyle.fontFamily,
            fontsize: chartStyle.fontSize,
          },
          itemstyle: {
            color: getCssVar("--color-surface-container-low"),
            bordercolor: getCssVar("--color-surface-container-low"),
            borderwidth: 5,
            borderradius: 5,
          },
          emphasis: {
            itemstyle: {
              shadowblur: 10,
              symbolSize: 30,
              shadowcolor: "rgba(0, 0, 0, 0.5)",
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
  }, []);
  return <div ref={chartRef} style={{ height: "400px", width: "100%" }}></div>;
};
export default PcaChart;
