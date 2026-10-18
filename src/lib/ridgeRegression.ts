/**
 * Client-Side Ridge Regression & Feature Engineering Engine for GridCast (PJM East Load Forecasting)
 * Solves (X^T X + alpha * I)^(-1) X^T y via Cholesky / Gaussian elimination in the browser.
 */

export interface DailyLoadRecord {
  date: string;
  loadMW: number;
}

export interface ForecastResult {
  date: string;
  actual?: number;
  predicted: number;
  naive?: number;
  residual?: number;
}

export interface EvaluationMetrics {
  mape: number;
  rmse: number;
  mae: number;
  r2: number;
  naiveMape: number;
  naiveRmse: number;
  naiveMae: number;
  naiveR2: number;
}

/**
 * Extracts lag features, rolling averages, day-of-week one-hot, and annual/semi-annual Fourier harmonics.
 */
export function buildFeatureVector(history: DailyLoadRecord[], targetIndex: number): number[] {
  const rec = history[targetIndex];
  const d = new Date(rec.date);
  const dayOfWeek = d.getUTCDay();
  const startOfYear = new Date(Date.UTC(d.getUTCFullYear(), 0, 0));
  const dayOfYear = Math.floor((d.getTime() - startOfYear.getTime()) / (1000 * 60 * 60 * 24));

  const lag1 = history[targetIndex - 1].loadMW;
  const lag2 = history[targetIndex - 2].loadMW;
  const lag7 = history[targetIndex - 7].loadMW;
  const lag14 = history[targetIndex - 14].loadMW;

  let sum7 = 0;
  for (let i = 1; i <= 7; i++) sum7 += history[targetIndex - i].loadMW;
  const rollMean7 = sum7 / 7;

  // Fourier seasonal terms (365.25 day period)
  const omega = (2 * Math.PI * dayOfYear) / 365.25;
  const sin1 = Math.sin(omega);
  const cos1 = Math.cos(omega);
  const sin2 = Math.sin(2 * omega);
  const cos2 = Math.cos(2 * omega);

  // Day-of-week cyclical encoding
  const dowSin = Math.sin((2 * Math.PI * dayOfWeek) / 7);
  const dowCos = Math.cos((2 * Math.PI * dayOfWeek) / 7);

  return [1, lag1, lag2, lag7, lag14, rollMean7, sin1, cos1, sin2, cos2, dowSin, dowCos];
}

/**
 * Solves linear system A * w = b using Gaussian elimination with partial pivoting.
 */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];

    const pivot = M[i][i] || 1e-12;
    for (let j = i; j <= n; j++) M[i][j] /= pivot;

    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = M[k][i];
        for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
      }
    }
  }

  return M.map(row => row[n]);
}

/**
 * Fits L2-regularized Ridge Regression weights: w = (X^T X + alpha * I)^(-1) X^T y
 */
export function trainRidgeRegression(X: number[][], y: number[], alpha = 10.0): number[] {
  const nSamples = X.length;
  const nFeatures = X[0].length;

  const XtX: number[][] = Array.from({ length: nFeatures }, () => Array(nFeatures).fill(0));
  const Xty: number[] = Array(nFeatures).fill(0);

  for (let i = 0; i < nSamples; i++) {
    const row = X[i];
    const target = y[i];
    for (let j = 0; j < nFeatures; j++) {
      Xty[j] += row[j] * target;
      for (let k = j; k < nFeatures; k++) {
        XtX[j][k] += row[j] * row[k];
      }
    }
  }

  for (let j = 0; j < nFeatures; j++) {
    for (let k = 0; k < j; k++) {
      XtX[j][k] = XtX[k][j];
    }
    // Do not penalize intercept (j === 0)
    if (j > 0) XtX[j][j] += alpha;
  }

  return solveLinearSystem(XtX, Xty);
}

export function predictRow(x: number[], weights: number[]): number {
  return x.reduce((sum, val, idx) => sum + val * weights[idx], 0);
}
