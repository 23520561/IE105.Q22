import type { histogramChartType } from "./api";

export function formatMatrix(
  matrix: number[][] | null,
): [number, number, number][] {
  if (!matrix) {
    return [[0, 0, 0]];
  }
  const result: [number, number, number][] = [];
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) {
      result.push([i, j, Math.round(matrix[i][j] * 100) / 100]);
    }
  }
  return result;
}
export function formatHistogramChart(
  chart: histogramChartType | null,
): [string, number][] {
  if (!chart) {
    return [["-", 0]];
  }
  return chart.histogram.map((row) => [
    ((row.bin_end + row.bin_start) / 2).toFixed(2),
    row.count,
  ]);
}
export function getCssVar(name: string) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}
