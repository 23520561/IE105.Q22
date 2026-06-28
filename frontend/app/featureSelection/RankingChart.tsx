import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import { initChartStyle } from "~/eda/charts/styles";

const RankChart = function ({
  data,
  max,
}: {
  data: [string, number][];
  max: number;
}) {
  const chartRef = useRef(null);

  useEffect(() => {
    const chartStyle = initChartStyle();
    const chart = echarts.init(chartRef.current);
    const option = {
      tooltip: {},
      grid: [{ top: 20, left: 0, right: 0 }],
      xAxis: [
        {
          type: "category",
          scale: true,
          axisLabel: {
            interval: 0,
            color: chartStyle.fontColor,
            fontFamily: chartStyle.fontFamily,
            fontSize: chartStyle.fontSize,
          },
        },
      ],
      yAxis: [
        {
          type: "value",
          axisLabel: {
            show: false,
          },
          axisLine: { show: false },
          splitLine: { show: false },
        },
      ],
      visualMap: {
        min: 0,
        max: max / 3,
        inRange: {
          color: chartStyle.itemColor,
        },
        show: false,
      },
      series: [
        {
          name: "histogram",
          type: "bar",
          label: {
            show: true,
            position: "top",
            color: chartStyle.fontColor,
            fontFamily: chartStyle.fontFamily,
            fontSize: chartStyle.fontSize,
          },
          itemStyle: {
            borderRadius: [10, 10, 0, 0],
          },
          data: data,
        },
      ],
    };
    chart.setOption(option);
    return () => {
      chart.dispose();
    };
  }, []);

  return <div ref={chartRef} style={{ height: "400px", width: "100%" }}></div>;
};
export default RankChart;
