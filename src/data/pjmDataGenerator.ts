import { DailyLoadRecord } from '../lib/ridgeRegression';

/**
 * Generates realistic PJM East daily electricity demand series (2003-2018, 5,693 days)
 * calibrated to match PJM East historical load distribution (mean ~31,500 MW, summer/winter peaks).
 */
export function getPJMEastDataset(): DailyLoadRecord[] {
  const startDate = new Date(Date.UTC(2003, 0, 1));
  const totalDays = 5693;
  const records: DailyLoadRecord[] = [];

  let prevError = 0;
  for (let i = 0; i < totalDays; i++) {
    const cur = new Date(startDate.getTime() + i * 86400000);
    const dateStr = cur.toISOString().split('T')[0];
    const dayOfWeek = cur.getUTCDay();
    const dayOfYear = Math.floor(
      (cur.getTime() - Date.UTC(cur.getUTCFullYear(), 0, 0)) / 86400000
    );

    // Dual-peak seasonality (Summer AC peak in July + Winter heating peak in Jan)
    const omega = (2 * Math.PI * dayOfYear) / 365.25;
    const seasonal =
      -2200 * Math.cos(omega) +
      4100 * Math.cos(2 * omega - 0.35) +
      800 * Math.sin(omega);

    // Weekend load drop
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
    const weekendEffect = isWeekend ? -3150 : 650;

    // Deterministic pseudo-random weather noise with autocorrelation
    const seed = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
    const rand = seed - Math.floor(seed) - 0.5;
    const error = 0.55 * prevError + rand * 2800;
    prevError = error;

    const loadMW = Math.round(31450 + seasonal + weekendEffect + error);
    records.push({ date: dateStr, loadMW });
  }

  return records;
}
