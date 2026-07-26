import { Injectable } from '@angular/core';
import { Capacitor, CapacitorHttp } from '@capacitor/core';

// Candidate AI URLs for real phone, emulator & localhost
const CANDIDATE_URLS = [
  'http://192.168.0.103:8000',
  'http://10.0.2.2:8000',
  'http://localhost:8000',
  'http://127.0.0.1:8000'
];

export interface PredictPayload {
  uid: string;
  SpeechProblems: number;
  Tremor: number;
  PosturalInstability: number;
  Bradykinesia: number;
  UPDRS: number;
  Rigidity: number;
  MoCA: number;
  FunctionalAssessment: number;
  Age: number;
  speechScore?: number;
  tremorScore?: number;
  fingerScore?: number;
  gaitScore?: number;
  questionnaireScore?: number;
}

export interface PredictResult {
  uid: string;
  riskScore: number;
  riskPercent: number;
  diagnosis: string;
  label: string;
  color: string;
  confidence: number;
}

export type RiskLevel = 'ปกติ' | 'เสี่ยงน้อย' | 'เสี่ยงปานกลาง' | 'เสี่ยงมาก';

export function getRiskLevel(percent: number): RiskLevel {
  if (percent < 30) return 'ปกติ';
  if (percent < 50) return 'เสี่ยงน้อย';
  if (percent < 70) return 'เสี่ยงปานกลาง';
  return 'เสี่ยงมาก';
}

export function getRiskColor(percent: number): string {
  if (percent < 30) return '#05C134';
  if (percent < 50) return '#F59E0B';
  if (percent < 70) return '#FF6B35';
  return '#C10508';
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private activeUrl: string | null = null;

  // ── เช็คว่า API ทำงานอยู่ไหม ──────────────────────────────
  async checkHealth(): Promise<boolean> {
    for (const baseUrl of CANDIDATE_URLS) {
      try {
        if (Capacitor.isNativePlatform()) {
          const response = await CapacitorHttp.get({
            url: `${baseUrl}/health`,
            connectTimeout: 2000
          });
          if (response.status === 200 && (response.data?.model_ready === true || response.data?.status === 'ok')) {
            this.activeUrl = baseUrl;
            return true;
          }
        } else {
          const res = await fetch(`${baseUrl}/health`, { method: 'GET' });
          if (res.ok) {
            const data = await res.json();
            if (data.model_ready === true || data.status === 'ok') {
              this.activeUrl = baseUrl;
              return true;
            }
          }
        }
      } catch (_e) {
        // Try next candidate URL
      }
    }
    this.activeUrl = null;
    return false;
  }

  // ── ส่งข้อมูลเซนเซอร์ไปให้โมเดลวิเคราะห์ ──────────────────
  async predict(payload: PredictPayload): Promise<PredictResult> {
    let lastError: any = null;

    const urlsToTry = this.activeUrl ? [this.activeUrl, ...CANDIDATE_URLS.filter(u => u !== this.activeUrl)] : CANDIDATE_URLS;

    for (const baseUrl of urlsToTry) {
      try {
        console.log('[ApiService] Calling predict at:', `${baseUrl}/predict`, 'isNative:', Capacitor.isNativePlatform());

        if (Capacitor.isNativePlatform()) {
          const response = await CapacitorHttp.post({
            url: `${baseUrl}/predict`,
            headers: { 'Content-Type': 'application/json' },
            data: payload,
            connectTimeout: 5000
          });

          if (response.status === 200) {
            this.activeUrl = baseUrl;
            return response.data as PredictResult;
          } else {
            lastError = `Status ${response.status}`;
          }
        } else {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 4000);

          const res = await fetch(`${baseUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
          });
          clearTimeout(timeoutId);

          if (res.ok) {
            this.activeUrl = baseUrl;
            return await res.json();
          }
        }
      } catch (err: any) {
        console.warn(`[ApiService] Failed connecting to ${baseUrl}:`, err);
        lastError = err?.message || err;
      }
    }

    this.activeUrl = null;
    throw new Error(`API connection failed: ${lastError || 'Could not reach server'}`);
  }

  // ── แปลงค่าจากแอปให้เป็น payload ที่ API รับได้ ────────────
  buildPayload(
    uid: string,
    sessionScores: { [testId: string]: number },
    userAge: number = 60
  ): PredictPayload {
    const speech = sessionScores['speech'] !== undefined ? sessionScores['speech'] : 0;
    const tremor = sessionScores['tremor'] !== undefined ? sessionScores['tremor'] : 0;
    const finger = sessionScores['finger'] !== undefined ? sessionScores['finger'] : 0;
    const gait = sessionScores['gait'] !== undefined ? sessionScores['gait'] : 0;
    const questionnaire = sessionScores['questionnaire'] !== undefined ? sessionScores['questionnaire'] : 0;

    return {
      uid,
      SpeechProblems: speech > 0.5 ? 1 : 0,
      Tremor: tremor > 0.5 ? 1 : 0,
      PosturalInstability: gait > 0.5 ? 1 : 0,
      Bradykinesia: finger > 0.5 ? 1 : 0,
      UPDRS: Math.round(questionnaire * 100),
      Rigidity: tremor > 0.5 ? 1 : 0,
      MoCA: Math.round(26 - questionnaire * 10),
      FunctionalAssessment: Math.round(100 - questionnaire * 50),
      Age: userAge,
      speechScore: speech,
      tremorScore: tremor,
      fingerScore: finger,
      gaitScore: gait,
      questionnaireScore: questionnaire
    };
  }
}