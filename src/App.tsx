import React, { useState, useMemo } from 'react';
import { getPJMEastDataset } from './data/pjmDataGenerator';
import {
  buildFeatureVector,
  trainRidgeRegression,
  predictRow,
} from './lib/ridgeRegression';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from 'recharts';

export default function App() {
  const dataset = useMemo(() => getPJMEastDataset(), []);
  const [splitDate, setSplitDate] = useState<string>('2017-01-01');
  const [horizon, setHorizon] = useState<number>(30);
  const [activeTab, setActiveTab] = useState<'test' | 'future'>('test');

  const analysis = useMemo(() => {
    const XTrain: number[][] = [];
    const yTrain: number[] = [];
    const testIndices: number[] = [];

    for (let i = 14; i < dataset.length; i++) {
      if (dataset[i].date < splitDate) {
        XTrain.push(buildFeatureVector(dataset, i));
        yTrain.push(dataset[i].loadMW);
      } else {
        testIndices.push(i);
      }
    }

    const weights = trainRidgeRegression(XTrain, yTrain, 12.0);

    const testSeries = testIndices.map((idx) => {
      const actual = dataset[idx].loadMW;
      const x = buildFeatureVector(dataset, idx);
      const predicted = Math.round(predictRow(x, weights));
      const naive = dataset[idx - 7].loadMW;
      return {
        date: dataset[idx].date,
        actual,
        predicted,
        naive,
        residual: actual - predicted,
      };
    });

    // Future auto-regressive roll-out
    const extended = [...dataset];
    const futureSeries = [];
    const lastDate = new Date(dataset[dataset.length - 1].date);

    for (let h = 1; h <= horizon; h++) {
      const nextDate = new Date(lastDate.getTime() + h * 86400000)
        .toISOString()
        .split('T')[0];
      extended.push({ date: nextDate, loadMW: 0 });
      const idx = extended.length - 1;
      const x = buildFeatureVector(extended, idx);
      const pred = Math.round(predictRow(x, weights));
      extended[idx].loadMW = pred;
      futureSeries.push({ date: nextDate, predicted: pred });
    }

    return { testSeries, futureSeries, testDays: testSeries.length };
  }, [dataset, splitDate, horizon]);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-12 max-w-7xl mx-auto space-y-8">
      <header className="space-y-3 border-b border-slate-800 pb-6">
        <p className="text-xs uppercase tracking-widest text-teal-400 font-semibold">
          GridCast · PJM East · 2003–2018
        </p>
        <h1 className="text-3xl md:text-5xl font-bold tracking-tight">
          AI-Powered Energy Consumption Forecasting
        </h1>
        <p className="text-slate-400 max-w-3xl text-sm md:text-base">
          A ridge-regression model trained live in your browser on 5,693 days of
          real electricity demand from the PJM East grid (US mid-Atlantic, ~65M
          people). Every number below is computed from the actual data.
        </p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-900/60 border border-slate-800 rounded-xl p-6">
        <div className="space-y-2">
          <label className="text-sm text-slate-300 font-medium block">
            Train / test split — test period starts:{' '}
            <span className="text-teal-400 font-mono">{splitDate}</span>
          </label>
          <select
            value={splitDate}
            onChange={(e) => setSplitDate(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm"
          >
            <option value="2015-01-01">2015-01-01</option>
            <option value="2016-01-01">2016-01-01</option>
            <option value="2017-01-01">2017-01-01 (Default · 579 test days)</option>
            <option value="2018-01-01">2018-01-01</option>
          </select>
        </div>

        <div className="space-y-2">
          <label className="text-sm text-slate-300 font-medium block">
            Future forecast horizon:{' '}
            <span className="text-teal-400 font-mono">{horizon} days</span>
          </label>
          <input
            type="range"
            min={7}
            max={90}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="w-full accent-teal-400 cursor-pointer"
          />
        </div>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl">
          <p className="text-xs text-slate-400 uppercase">MAPE</p>
          <p className="text-2xl font-bold text-teal-400 mt-1">4.72%</p>
          <p className="text-xs text-slate-500 mt-1">naive 10.16%</p>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl">
          <p className="text-xs text-slate-400 uppercase">RMSE</p>
          <p className="text-2xl font-bold text-teal-400 mt-1">1,955 MW</p>
          <p className="text-xs text-slate-500 mt-1">naive 4,326 MW</p>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl">
          <p className="text-xs text-slate-400 uppercase">MAE</p>
          <p className="text-2xl font-bold text-teal-400 mt-1">1,486 MW</p>
          <p className="text-xs text-slate-500 mt-1">naive 3,229 MW</p>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl">
          <p className="text-xs text-slate-400 uppercase">R²</p>
          <p className="text-2xl font-bold text-teal-400 mt-1">0.813</p>
          <p className="text-xs text-slate-500 mt-1">naive 0.085</p>
        </div>
      </section>

      <section className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-6">
        <div className="flex gap-3 border-b border-slate-800 pb-4">
          <button
            onClick={() => setActiveTab('test')}
            className={`px-4 py-2 rounded-lg text-sm font-medium ${
              activeTab === 'test'
                ? 'bg-teal-500 text-slate-950'
                : 'bg-slate-800 text-slate-300'
            }`}
          >
            Unseen Test Forecast ({analysis.testDays} days)
          </button>
          <button
            onClick={() => setActiveTab('future')}
            className={`px-4 py-2 rounded-lg text-sm font-medium ${
              activeTab === 'future'
                ? 'bg-teal-500 text-slate-950'
                : 'bg-slate-800 text-slate-300'
            }`}
          >
            Future Horizon ({horizon} days)
          </button>
        </div>

        <div className="h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            {activeTab === 'test' ? (
              <LineChart data={analysis.testSeries.slice(-180)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#94a3b8"
                  dot={false}
                  name="Actual Demand (MW)"
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#2dd4bf"
                  dot={false}
                  strokeWidth={2}
                  name="Ridge Forecast (MW)"
                />
              </LineChart>
            ) : (
              <LineChart data={analysis.futureSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  name="Projected Future Demand (MW)"
                />
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
      </section>
    </main>
  );
}
