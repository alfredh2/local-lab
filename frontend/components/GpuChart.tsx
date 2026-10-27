"use client";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export type Series = { key: string; label: string; color: string; axis?: "left" | "right" };

export default function LineChartCard({ data, series, xKey = "t", xLabel, height = 220 }: {
  data: Record<string, any>[]; series: Series[]; xKey?: string; xLabel?: (v: any) => string; height?: number;
}) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
          <XAxis dataKey={xKey} stroke="#71717a" fontSize={11} tickFormatter={xLabel} minTickGap={40} />
          <YAxis yAxisId="left" stroke="#71717a" fontSize={11} width={44} />
          {series.some((s) => s.axis === "right") && <YAxis yAxisId="right" orientation="right" stroke="#71717a" fontSize={11} width={44} />}
          <Tooltip contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }}
            labelFormatter={xLabel} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {series.map((s) => (
            <Line key={s.key} yAxisId={s.axis ?? "left"} type="monotone" dataKey={s.key} name={s.label}
              stroke={s.color} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
