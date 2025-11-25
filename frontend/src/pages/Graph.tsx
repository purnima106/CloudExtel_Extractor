import { useState, useEffect } from "react";
import { BarChart3, Loader2, AlertCircle } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { getGraphData } from "../api";

interface GraphDataPoint {
  x: number;
  e: number | null;
  g: number | null;
}

interface GraphDataResponse {
  x_column: string;
  e_column: string;
  g_column: string;
  columns: string[];
  data: GraphDataPoint[];
  x_label: string;
  y_label: string;
}

export default function Graph() {
  const [data, setData] = useState<GraphDataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const response = await getGraphData();
        setData(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load graph data");
        console.error("Error fetching graph data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Format data for Recharts (convert nulls to undefined for better handling)
  const rawChartData = data?.data
    .filter((point) => point.x !== null && point.x !== undefined && !isNaN(Number(point.x)))
    .map((point) => {
      const xVal = Number(point.x);
      const eVal = point.e !== null && point.e !== undefined && !isNaN(Number(point.e)) 
        ? Number(point.e) 
        : undefined;
      const gVal = point.g !== null && point.g !== undefined && !isNaN(Number(point.g)) 
        ? Number(point.g) 
        : undefined;
      
      return {
        x: xVal,
        "Budget Line": eVal,
        "Actual Line": gVal,
      };
    })
    .sort((a, b) => a.x - b.x) || []; // Sort by X value for proper line rendering

  // For better readability, we'll hide dots by default and only show them on hover
  // This keeps the lines clean while still allowing detailed inspection
  const chartData = rawChartData;
  
  // Optionally, we can sample data points if there are too many for smoother rendering
  // But keep all points for accurate line drawing
  const shouldSampleData = rawChartData.length > 200;
  const sampledChartData = shouldSampleData 
    ? rawChartData.filter((_, index) => index % Math.ceil(rawChartData.length / 200) === 0 || index === 0 || index === rawChartData.length - 1)
    : rawChartData;
  
  // Use sampled data for rendering if we have too many points, but keep all for tooltips
  const displayData = shouldSampleData ? sampledChartData : chartData;
  
  // Store sampling info for display
  const samplingInfo = shouldSampleData ? `(displaying ${displayData.length} for clarity)` : '';

  // Debug: Log data to console
  useEffect(() => {
    if (data) {
      console.log("Graph data received:", data);
      console.log("Chart data points:", chartData);
      console.log("Data points count:", chartData.length);
      if (chartData.length > 0) {
        console.log("First 3 data points:", chartData.slice(0, 3));
        console.log("Last 3 data points:", chartData.slice(-3));
      }
    }
  }, [data, chartData]);

  return (
    <div className="graph-page fade-in">
      <header className="page-header">
        <div className="page-title">
          <BarChart3 size={28} style={{ verticalAlign: "middle", marginRight: "0.75rem" }} />
          <h1>Budget vs Actuals Projection</h1>
        </div>
        <p className="page-subtitle">
          Budget Line vs Actual Line S-Curve
        </p>
      </header>

      {loading && (
        <div className="graph-container">
          <div className="graph-placeholder">
            <Loader2 size={48} style={{ opacity: 0.5, marginBottom: "1rem", animation: "spin 0.6s linear infinite" }} />
            <p style={{ color: "var(--muted)", fontSize: "var(--text-lg)" }}>
              Loading graph data...
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="graph-container">
          <div className="error" style={{ padding: "var(--space-6)", textAlign: "center" }}>
            <AlertCircle size={32} style={{ marginBottom: "var(--space-3)" }} />
            <p style={{ color: "var(--danger)", fontSize: "var(--text-lg)", margin: 0 }}>
              {error}
            </p>
            <p style={{ color: "var(--muted)", fontSize: "var(--text-sm)", marginTop: "var(--space-2)" }}>
              Make sure Book6.xlsx exists in the root directory and the backend is running.
            </p>
          </div>
        </div>
      )}

      {!loading && !error && data && (
        <div className="graph-container">
          <div className="graph-info">
            <div className="info-item">
              <span className="info-label">X-Axis:</span>
              <span className="info-value">{data.x_label}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Y-Axis:</span>
              <span className="info-value">{data.y_label}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Data Points:</span>
              <span className="info-value">{chartData.length} {samplingInfo}</span>
            </div>
            {chartData.length > 0 && (
              <>
                <div className="info-item">
                  <span className="info-label">X Range:</span>
                  <span className="info-value">
                    {Math.min(...chartData.map(d => d.x)).toFixed(1)} - {Math.max(...chartData.map(d => d.x)).toFixed(1)} km
                  </span>
                </div>
                <div className="info-item">
                  <span className="info-label">Budget Values:</span>
                  <span className="info-value">
                    {chartData.filter(d => d["Budget Line"] !== undefined).length} points
                  </span>
                </div>
                <div className="info-item">
                  <span className="info-label">Actual Values:</span>
                  <span className="info-value">
                    {chartData.filter(d => d["Actual Line"] !== undefined).length} points
                  </span>
                </div>
              </>
            )}
            <div className="info-item">
              <span className="info-label">X Column:</span>
              <span className="info-value">{data.x_column}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Budget Column:</span>
              <span className="info-value">{data.e_column || "Not found"}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Actual Column:</span>
              <span className="info-value">{data.g_column || "Not found"}</span>
            </div>
          </div>

          {chartData.length === 0 ? (
            <div className="graph-placeholder">
              <AlertCircle size={48} style={{ opacity: 0.5, marginBottom: "1rem" }} />
              <p style={{ color: "var(--warning)", fontSize: "var(--text-lg)", margin: 0 }}>
                No data points found to plot.
              </p>
              <p style={{ color: "var(--muted)", fontSize: "var(--text-sm)", marginTop: "var(--space-2)" }}>
                Please check that Book6.xlsx contains valid numeric data in the X column and columns E and G.
              </p>
            </div>
          ) : (
          <div className="chart-wrapper" style={{ background: "var(--card)" }}>
            <ResponsiveContainer width="100%" height={600}>
              <LineChart
                data={displayData}
                margin={{ top: 20, right: 30, left: 80, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" opacity={0.2} />
                <XAxis
                  dataKey="x"
                  type="number"
                  scale="linear"
                  domain={[0, 60]}
                  ticks={[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]}
                  allowDecimals={false}
                  label={{ 
                    value: data.x_label, 
                    position: "insideBottom", 
                    offset: -10,
                    style: { fill: "var(--text)", fontSize: "16px", fontWeight: "600" }
                  }}
                  stroke="var(--text)"
                  tick={{ fill: "var(--text)", fontSize: "14px", fontWeight: "600" }}
                  tickFormatter={(value) => `${value}`}
                />
                <YAxis
                  type="number"
                  domain={[0, 60]}
                  ticks={[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]}
                  label={{ 
                    value: data.y_label, 
                    angle: -90, 
                    position: "insideLeft",
                    style: { fill: "var(--text)", fontSize: "16px", fontWeight: "600" }
                  }}
                  stroke="var(--text)"
                  tick={{ fill: "var(--text)", fontSize: "14px", fontWeight: "600" }}
                  tickFormatter={(value) => `${value}`}
                  reversed={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--card)",
                    border: "1px solid var(--border-default)",
                    borderRadius: "var(--radius-md)",
                    color: "var(--text)",
                    boxShadow: "var(--shadow-lg)",
                  }}
                  labelStyle={{ color: "var(--brand)", fontWeight: "600", marginBottom: "4px" }}
                  formatter={(value: any, name: string) => {
                    if (value === undefined || value === null) return ["N/A", name];
                    return [`${value} Cr`, name];
                  }}
                  labelFormatter={(label) => `X: ${label} km`}
                />
                <Legend
                  wrapperStyle={{ paddingTop: "var(--space-4)" }}
                  iconType="line"
                  iconSize={16}
                  formatter={(value) => <span style={{ color: "var(--text)" }}>{value}</span>}
                />
                <Line
                  type="monotone"
                  dataKey="Budget Line"
                  stroke="#fbbf24"
                  strokeWidth={3.5}
                  dot={false}
                  activeDot={{ r: 7, fill: "#fbbf24", strokeWidth: 2.5, stroke: "#fff", strokeOpacity: 0.8 }}
                  name="Budget Line"
                  connectNulls={false}
                  isAnimationActive={true}
                  strokeOpacity={1}
                />
                <Line
                  type="monotone"
                  dataKey="Actual Line"
                  stroke="#10b981"
                  strokeWidth={3.5}
                  dot={false}
                  activeDot={{ r: 7, fill: "#10b981", strokeWidth: 2.5, stroke: "#fff", strokeOpacity: 0.8 }}
                  name="Actual Line"
                  connectNulls={false}
                  isAnimationActive={true}
                  strokeOpacity={1}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          )}
        </div>
      )}
    </div>
  );
}

