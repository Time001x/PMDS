import { Injectable } from '@angular/core';
import { Capacitor, CapacitorHttp } from '@capacitor/core';

// Candidate AI URLs for real phone, emulator & localhost (10.0.2.2 is Android emulator host loopback)
const CANDIDATE_URLS = [
  'http://10.0.2.2:8000',
  'http://192.168.0.102:8000',
  'http://localhost:8000',
  'http://127.0.0.1:8000',
  'http://192.168.0.100:8000',
  'http://192.168.0.101:8000',
  'http://192.168.0.103:8000',
  'http://192.168.0.104:8000'
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
  level: number; // 0, 1, 2, 3, 4 MDS-UPDRS Scale
  riskScore?: number;
  riskPercent?: number;
  diagnosis: string;
  label: string;
  color: string;
  description: string;
  confidence: number;
}

export type RiskLevel = 'ไม่มีอาการ' | 'เล็กน้อย' | 'เสี่ยงปานกลาง' | 'เสี่ยงมาก' | 'อาการรุนแรง';

export function getRiskLevelInfo(level: number) {
  const lvl = Math.max(0, Math.min(4, Math.round(level)));
  switch (lvl) {
    case 0: return { level: 0, label: 'ไม่มีอาการ', color: '#05C134', desc: 'ปกติ สมบูรณ์ ไม่มีอาการแสดงของโรคพาร์กินสัน' };
    case 1: return { level: 1, label: 'เล็กน้อย', color: '#10B981', desc: 'มีความผิดปกติเพียงเล็กน้อย ไม่กระทบต่อการดำเนินชีวิต' };
    case 2: return { level: 2, label: 'เสี่ยงปานกลาง', color: '#F59E0B', desc: 'มีอาการชัดเจนขึ้น เริ่มส่งผลกระทบต่อบางกิจกรรม' };
    case 3: return { level: 3, label: 'เสี่ยงมาก', color: '#F97316', desc: 'มีอาการชัดเจน รบกวนการทำกิจกรรมประจำวันอย่างมาก' };
    case 4: return { level: 4, label: 'อาการรุนแรง', color: '#C10508', desc: 'มีอาการรุนแรงมาก ไม่สามารถพึ่งพาตนเองได้' };
    default: return { level: 0, label: 'ไม่มีอาการ', color: '#05C134', desc: 'ปกติ สมบูรณ์ ไม่มีอาการแสดงของโรคพาร์กินสัน' };
  }
}

export function getItemLevel(score: number): number {
  if (score <= 0) return 0;
  if (score > 4) {
    // If percentage (0 - 100)
    return score >= 80 ? 4 : score >= 60 ? 3 : score >= 40 ? 2 : score >= 20 ? 1 : 0;
  }
  if (score > 1.0) {
    // If integer 2, 3, 4
    return Math.min(4, Math.round(score));
  }
  // If float ratio (0.0 to 1.0) e.g. questionnaire or sensor risk ratio
  return score >= 0.80 ? 4 : score >= 0.60 ? 3 : score >= 0.40 ? 2 : score >= 0.20 ? 1 : 0;
}

export function getRiskLevel(scoreOrPercent: number): string {
  const lvl = getItemLevel(scoreOrPercent);
  return getRiskLevelInfo(lvl).label;
}

export function getRiskColor(scoreOrPercent: number): string {
  const lvl = getItemLevel(scoreOrPercent);
  return getRiskLevelInfo(lvl).color;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private activeUrl: string | null = null;

  constructor() {}

  // ── เช็คความพร้อมของ Backend ──────────────────────────────
  async checkHealth(): Promise<boolean> {
    if (this.activeUrl) {
      try {
        if (Capacitor.isNativePlatform()) {
          const res = await CapacitorHttp.get({ url: `${this.activeUrl}/health`, connectTimeout: 1500 });
          if (res.status === 200) return true;
        } else {
          const res = await fetch(`${this.activeUrl}/health`);
          if (res.ok) return true;
        }
      } catch (_e) {
        this.activeUrl = null;
      }
    }

    for (const baseUrl of CANDIDATE_URLS) {
      try {
        if (Capacitor.isNativePlatform()) {
          const response = await CapacitorHttp.get({
            url: `${baseUrl}/health`,
            connectTimeout: 1500
          });
          if (response.status === 200) {
            this.activeUrl = baseUrl;
            console.log('[ApiService] Found active backend server at:', baseUrl);
            return true;
          }
        } else {
          const controller = new AbortController();
          const tid = setTimeout(() => controller.abort(), 1500);
          const res = await fetch(`${baseUrl}/health`, { signal: controller.signal });
          clearTimeout(tid);
          if (res.ok) {
            this.activeUrl = baseUrl;
            console.log('[ApiService] Found active backend server at:', baseUrl);
            return true;
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

    // Fast discover active URL if not set
    if (!this.activeUrl) {
      await this.checkHealth();
    }

    const urlsToTry = this.activeUrl
      ? [this.activeUrl, ...CANDIDATE_URLS.filter(u => u !== this.activeUrl)]
      : CANDIDATE_URLS;

    for (const baseUrl of urlsToTry) {
      try {
        console.log('[ApiService] Trying predict at:', `${baseUrl}/predict`);

        // 1. Try standard fetch first (works reliably on both web and Android WebView)
        try {
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
            const data = await res.json();
            this.activeUrl = baseUrl;
            return data as PredictResult;
          }
        } catch (fetchErr) {
          console.warn(`[ApiService] fetch failed for ${baseUrl}, trying CapacitorHttp...`, fetchErr);
        }

        // 2. Try CapacitorHttp native plugin if fetch failed on Native Platform
        if (Capacitor.isNativePlatform()) {
          const response = await CapacitorHttp.post({
            url: `${baseUrl}/predict`,
            headers: { 'Content-Type': 'application/json' },
            data: payload,
            connectTimeout: 4000
          });

          if (response.status === 200) {
            this.activeUrl = baseUrl;
            const parsedData = typeof response.data === 'string' ? JSON.parse(response.data) : response.data;
            return parsedData as PredictResult;
          } else {
            lastError = `Status ${response.status}`;
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

  // ── สมัครสมาชิกไปยัง Backend ──────────────────────────────
  async registerUserOnBackend(username: string, fullName: string, email: string, password_raw: string): Promise<any> {
    if (!this.activeUrl) {
      await this.checkHealth();
    }
    const urlsToTry = this.activeUrl ? [this.activeUrl, ...CANDIDATE_URLS.filter(u => u !== this.activeUrl)] : CANDIDATE_URLS;
    const body = { username: username || email, password: password_raw, fullName: fullName, email: email };

    for (const baseUrl of urlsToTry) {
      try {
        console.log('[ApiService] Registering user at:', `${baseUrl}/auth/register`);
        try {
          const controller = new AbortController();
          const tid = setTimeout(() => controller.abort(), 3000);
          const res = await fetch(`${baseUrl}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal
          });
          clearTimeout(tid);
          if (res.ok || res.status === 400) {
            this.activeUrl = baseUrl;
            return await res.json();
          }
        } catch (_fetchErr) {}

        if (Capacitor.isNativePlatform()) {
          const res = await CapacitorHttp.post({
            url: `${baseUrl}/auth/register`,
            headers: { 'Content-Type': 'application/json' },
            data: body,
            connectTimeout: 3000
          });
          if (res.status === 200 || res.status === 400) {
            this.activeUrl = baseUrl;
            return typeof res.data === 'string' ? JSON.parse(res.data) : res.data;
          }
        }
      } catch (err) {
        console.warn(`[ApiService] Register failed for ${baseUrl}:`, err);
      }
    }
    return null;
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