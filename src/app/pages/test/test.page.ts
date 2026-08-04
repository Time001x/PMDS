import { Component, OnInit, OnDestroy, NgZone } from '@angular/core';
import { Router, ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Subject, of, EMPTY } from 'rxjs';
import { concatMap, catchError, takeUntil } from 'rxjs/operators';
import { DbService } from '../../services/db.service';
import { TestConfigService } from '../../services/test-config.service';
import { ApiService, getRiskLevel, getRiskColor } from '../../services/api.service';
import { SpeechTestComponent } from './components/speech-test/speech-test.component';
import { TremorTestComponent } from './components/tremor-test/tremor-test.component';
import { FingerTapTestComponent } from './components/finger-tap-test/finger-tap-test.component';
import { GaitTestComponent } from './components/gait-test/gait-test.component';
import { UpdrsQuestionnaireComponent } from './components/updrs-questionnaire/updrs-questionnaire.component';

@Component({
  selector: 'app-test',
  templateUrl: 'test.page.html',
  styleUrls: ['test.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    SpeechTestComponent,
    TremorTestComponent,
    FingerTapTestComponent,
    GaitTestComponent,
    UpdrsQuestionnaireComponent
  ]
})
export class TestPage implements OnInit, OnDestroy {
  Math = Math;
  currentUser: any = null;
  currentView: 'menu' | 'test' | 'history' | 'result' = 'menu';
  currentTestId: string | null = null;
  backStack: (() => void)[] = [];

  testMenuData: any[] = [];
  allTestsDone = false;

  historyData: any[] = [];
  historySummary: any = { count: 0, latest: 0, avg: 0 };

  resultRingPercent = 0;
  resultRiskLevel = 'ปกติ';
  resultColor = '#888';
  resultLabel = '';
  resultScores: any[] = [];

  modalVisible = false;
  modalHtml = '';

  toastMessage = '';
  toastVisible = false;
  private toastTimeout: any;

  // ── สถานะการเรียก API ──────────────────────────────────────
  isAnalyzing = false;
  analyzeError = false;

  private saveScore$ = new Subject<{ testId: string; score: number }>();
  private destroy$ = new Subject<void>();

  getRiskLevel = getRiskLevel;
  getRiskColor = getRiskColor;

  constructor(
    private db: DbService,
    private testConfig: TestConfigService,
    private router: Router,
    private route: ActivatedRoute,
    private apiService: ApiService,
    private ngZone: NgZone
  ) {}

  ngOnInit() {
    const session = this.db.getSession();
    if (!session) {
      this.router.navigateByUrl('/login');
      return;
    }
    this.currentUser = session;

    this.saveScore$.pipe(
      concatMap(({ testId, score }) => {
        const sessionScores = this.db.getSessionScores(this.currentUser.uid);
        sessionScores[testId] = score;
        this.db.setSessionScores(this.currentUser.uid, sessionScores);

        const tests = this.testConfig.getTestList();
        const done = tests.filter(t => sessionScores[t.id] !== undefined);
        if (done.length >= 5) {
          // ── ครบ 5 test → ส่งไป AI วิเคราะห์ ───────────────
          setTimeout(() => this.analyzeWithAI(sessionScores), 300);
        } else {
          this.renderTestMenu();
        }
        return of(undefined);
      }),
      catchError((err) => {
        console.error('[TestPage] saveScore$ error:', err);
        return EMPTY;
      }),
      takeUntil(this.destroy$)
    ).subscribe();

    this.route.queryParams.subscribe(params => {
      const testType = params['type'];
      const view = params['view'];
      if (view === 'history') {
        this.renderHistory();
      } else if (testType) {
        this.startTest(testType);
      } else {
        this.renderTestMenu();
      }
    });
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }

  // ── ส่งข้อมูลไป AI วิเคราะห์ ────────────────────────────────
  async analyzeWithAI(sessionScores: { [testId: string]: number }) {
    this.isAnalyzing = true;
    this.analyzeError = false;
    this.currentView = 'result';
    this.currentTestId = null;
    this.backStack = [];

    try {
      // สร้าง payload ส่งไป API
      const payload = this.apiService.buildPayload(
        this.currentUser?.uid || 'demo_user',
        sessionScores,
        this.currentUser?.age ? parseInt(this.currentUser.age) : 60
      );

      console.log('[PMDS] ส่งข้อมูลไป AI:', payload);

      // เรียก API
      const result = await this.apiService.predict(payload);

      console.log('[PMDS] ผลจาก AI:', result);

      // ใช้ผลจาก AI แทนการคำนวณเอง
      const totalRisk = result.riskPercent;
      const riskLevel = getRiskLevel(totalRisk);
      const riskColor = getRiskColor(totalRisk);

      const record = {
        riskPercent: totalRisk,
        riskLevel: riskLevel,
        scores: { ...sessionScores },
        level: riskLevel
      };
      this.db.addHistory(this.currentUser.uid, record);
      this.db.clearSessionScores(this.currentUser.uid);

      this.resultRingPercent = totalRisk;
      this.resultRiskLevel = riskLevel;
      this.resultColor = riskColor;
      this.resultLabel = riskLevel;

      const tests = this.testConfig.getTestList();
      this.resultScores = tests.map(t => ({
        ...t,
        score: Math.round((sessionScores[t.id] || 0) * 100),
        color: (sessionScores[t.id] || 0) > 0.6
          ? 'var(--accent-danger)'
          : (sessionScores[t.id] || 0) > 0.35
            ? 'var(--accent-warn)'
            : 'var(--accent-good)'
      }));

      this.showToast(`🤖 AI วิเคราะห์เสร็จ: ${riskLevel} (${result.confidence * 100 | 0}% confidence)`, 'teal');

    } catch (err: any) {
      console.error('[PMDS] API error:', err);
      this.analyzeError = true;
      const errMsg = err?.message || 'ไม่สามารถเชื่อมต่อ AI ได้';
      this.showToast(`⚠️ เชื่อม AI ไม่สำเร็จ: ${errMsg} (ใช้เกณฑ์ประเมินมาตรฐาน)`, 'warn');

      // Fallback: คำนวณแบบเดิมถ้า API ไม่ตอบ
      this.showFinalResultFallback(sessionScores);
    } finally {
      this.isAnalyzing = false;
    }
  }

  // ── Fallback ถ้า API ไม่ตอบ ──────────────────────────────────
  showFinalResultFallback(sessionScores: { [testId: string]: number }) {
    const tests = this.testConfig.getTestList();
    const totalRisk = this.testConfig.calculateTotalRisk(sessionScores);
    const riskLevel = getRiskLevel(totalRisk);
    const riskColor = getRiskColor(totalRisk);

    const record = { riskPercent: totalRisk, riskLevel: riskLevel, scores: { ...sessionScores }, level: riskLevel };
    this.db.addHistory(this.currentUser.uid, record);
    this.db.clearSessionScores(this.currentUser.uid);

    this.resultRingPercent = totalRisk;
    this.resultRiskLevel = riskLevel;
    this.resultColor = riskColor;
    this.resultLabel = riskLevel;
    this.resultScores = tests.map(t => ({
      ...t,
      score: Math.round((sessionScores[t.id] || 0) * 100),
      color: (sessionScores[t.id] || 0) > 0.6
        ? 'var(--accent-danger)'
        : (sessionScores[t.id] || 0) > 0.35
          ? 'var(--accent-warn)'
          : 'var(--accent-good)'
    }));
  }

  // ── Navigation ───────────────────────────────────────────────
  navTo(page: string) {
    this.router.navigateByUrl('/' + page);
  }

  handleBack() {
    if (this.backStack.length > 0) {
      const fn = this.backStack.pop()!;
      fn();
      return;
    }
    if (this.currentView === 'history' || this.currentView === 'result') {
      this.renderTestMenu();
      return;
    }
    this.router.navigateByUrl('/home');
  }

  // ── Toast ─────────────────────────────────────────────────────
  showToast(msg: string, type: string = 'teal') {
    this.toastMessage = msg;
    this.toastVisible = true;
    clearTimeout(this.toastTimeout);
    this.toastTimeout = setTimeout(() => this.toastVisible = false, 3000);
  }

  // ── CSV File Upload & Dataset Analysis Mode ───────────────────
  selectedFileName = '';
  isProcessingFile = false;
  datasetSummary: any = null;

  triggerFileInput(fileInput: HTMLInputElement) {
    fileInput.click();
  }

  async onFileSelected(event: any) {
    const file = event.target?.files?.[0];
    if (!file) return;

    this.ngZone.run(() => {
      this.selectedFileName = file.name;
      this.isProcessingFile = true;
      this.showToast(`📁 กำลังอ่านและส่งไฟล์วิเคราะห์ AI: ${file.name}...`, 'teal');
    });

    const reader = new FileReader();
    reader.onload = (e: any) => {
      this.ngZone.run(() => {
        try {
          const text = e.target.result || '';
          const lines = text.split(/\r?\n/).map((l: string) => l.trim()).filter((l: string) => l.length > 0);

          if (lines.length < 2) {
            this.showToast('⚠️ ไฟล์ไม่มีข้อมูลหรือมีขนาดเล็กเกินไป', 'warn');
            this.isProcessingFile = false;
            return;
          }

          const delimiter = lines[0].includes(',') ? ',' : (lines[0].includes('\t') ? '\t' : /\s+/);
          const headers = lines[0].split(delimiter).map((h: string) => h.trim().replace(/^"|"$/g, ''));
          const rows = lines.slice(1);

          let validCount = 0;
          let sumSpeech = 0;
          let sumTremor = 0;
          let sumFinger = 0;
          let sumGait = 0;
          let sumQ = 0;

          const speechColIdx = headers.findIndex((h: string) => 
            ['speechscore', 'speechproblems', 'mdvp:jitter(%)'].includes(h.toLowerCase())
          );
          const tremorColIdx = headers.findIndex((h: string) => 
            ['tremorscore', 'tremor', 'doctor_diagnosis_0_5'].includes(h.toLowerCase())
          );
          const fingerColIdx = headers.findIndex((h: string) => 
            ['fingerscore', 'bradykinesia'].includes(h.toLowerCase())
          );
          const gaitColIdx = headers.findIndex((h: string) => 
            ['gaitscore', 'posturalinstability'].includes(h.toLowerCase())
          );
          const qColIdx = headers.findIndex((h: string) => 
            ['questionnairescore', 'updrs'].includes(h.toLowerCase())
          );

          for (const line of rows) {
            const vals = line.split(delimiter).map((v: string) => v.trim().replace(/^"|"$/g, ''));
            if (vals.length < Math.max(1, headers.length * 0.3)) continue;

            validCount++;

            const speech = speechColIdx >= 0 ? parseFloat(vals[speechColIdx]) || 0 : 0;
            let tremor = tremorColIdx >= 0 ? parseFloat(vals[tremorColIdx]) || 0 : 0;
            const finger = fingerColIdx >= 0 ? parseFloat(vals[fingerColIdx]) || 0 : 0;
            const gait = gaitColIdx >= 0 ? parseFloat(vals[gaitColIdx]) || 0 : 0;
            const q = qColIdx >= 0 ? parseFloat(vals[qColIdx]) || 0 : 0;

            let normSpeech = speech;
            if (speechColIdx >= 0) {
              const colName = headers[speechColIdx].toLowerCase();
              if (colName.includes('jitter') && speech < 0.1) {
                normSpeech = speech * 40.0;
              }
            }

            if (tremor > 1.0) {
              tremor = tremor / 4.0; // Scale 0-4 diagnosis to 0-1
            }

            sumSpeech += normSpeech;
            sumTremor += tremor;
            sumFinger += finger;
            sumGait += gait;
            sumQ += q;
          }

          if (validCount === 0) {
            this.showToast('⚠️ ไม่พบแถวข้อมูลที่สมบูรณ์ในไฟล์', 'warn');
            this.isProcessingFile = false;
            return;
          }

          const datasetSessionScores: { [testId: string]: number } = {
            speech: parseFloat((sumSpeech / validCount).toFixed(4)),
            tremor: parseFloat((sumTremor / validCount).toFixed(4)),
            finger: parseFloat((sumFinger / validCount).toFixed(4)),
            gait: parseFloat((sumGait / validCount).toFixed(4)),
            questionnaire: parseFloat((sumQ / validCount).toFixed(4))
          };

          this.showToast(`🤖 สกัดข้อมูลจากไฟล์ ${file.name} ส่งวิเคราะห์ AI...`, 'teal');
          this.analyzeWithAI(datasetSessionScores);

        } catch (err: any) {
          console.error('[PMDS] File processing error:', err);
          this.showToast('⚠️ เกิดข้อผิดพลาดในการวิเคราะห์ไฟล์', 'warn');
        } finally {
          this.isProcessingFile = false;
        }
      });
    };

    reader.readAsText(file);
  }

  // ── Test Menu ─────────────────────────────────────────────────
  renderTestMenu() {
    this.currentView = 'menu';
    this.currentTestId = null;
    this.backStack = [];

    const tests = this.testConfig.getTestList();
    const sessionScores = this.db.getSessionScores(this.currentUser.uid);
    this.testMenuData = tests.map(t => ({
      ...t,
      done: sessionScores[t.id] !== undefined
    }));
    this.allTestsDone = Object.keys(sessionScores).length >= 5;
  }

  startTest(id: string) {
    this.backStack.push(() => this.renderTestMenu());
    this.currentView = 'test';
    this.currentTestId = id;
  }

  // ── Child Component Events ────────────────────────────────────
  onChildTestCompleted(testId: string, score: number) {
    this.saveTestScore(testId, score);
  }

  saveTestScore(testId: string, score: number) {
    this.saveScore$.next({ testId, score });
  }

  // ── showFinalResult (ใช้โดย history detail) ──────────────────
  showFinalResult() {
    const sessionScores = this.db.getSessionScores(this.currentUser.uid);
    this.analyzeWithAI(sessionScores);
  }

  restartTests() {
    this.db.clearSessionScores(this.currentUser.uid);
    this.renderTestMenu();
  }

  // ── History ───────────────────────────────────────────────────
  renderHistory() {
    this.currentView = 'history';
    this.currentTestId = null;
    this.backStack = [];

    const hist = this.db.getHistory(this.currentUser.uid);
    const sorted = [...hist].reverse();
    this.historyData = sorted;
    this.historySummary = {
      count: hist.length,
      latest: hist.length > 0 ? hist[0].riskPercent : 0,
      avg: hist.length > 0
        ? Math.round(hist.reduce((s: number, h: any) => s + h.riskPercent, 0) / hist.length)
        : 0
    };
  }

  getTestListForHistory(): any[] {
    return [
      { id: 'speech', name: 'เสียงพูด', icon: '🎤' },
      { id: 'tremor', name: 'อาการสั่น', icon: '📳' },
      { id: 'finger', name: 'แตะนิ้ว', icon: '👆' },
      { id: 'gait', name: 'การเดิน', icon: '🚶' },
      { id: 'questionnaire', name: 'แบบสอบถาม', icon: '📝' }
    ];
  }

  // ── HTML เรียกใช้ตัวนี้ (alias ของ getTestListForHistory) ───
  getTestList(): any[] {
    return this.getTestListForHistory();
  }

  // ── ปิด modal เมื่อคลิกพื้นหลัง ───────────────────────────────
  onModalSheetClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      this.closeModal();
    }
  }

  viewHistoryDetail(h: any) {
    const tests = this.getTestListForHistory();
    const riskLevel = getRiskLevel(h.riskPercent);
    const riskColor = getRiskColor(h.riskPercent);
    const scoresHtml = tests.map(t => {
      const sc = h.scores?.[t.id];
      const pct = sc !== undefined ? Math.round(sc * 100) : null;
      const color = pct === null ? '#ccc' : pct > 60 ? '#C10508' : pct > 35 ? '#F59E0B' : '#05C134';
      return `<div class="hdetail-score-row">
        <span class="hdsr-icon">${t.icon}</span>
        <span class="hdsr-name">${t.name}</span>
        <div class="hdsr-bar"><div class="hdsr-bar-fill" style="width:${pct||0}%;background:${color}"></div></div>
        <span class="hdsr-val" style="color:${color}">${pct !== null ? pct + '%' : '-'}</span>
      </div>`;
    }).join('');

    this.showModal(`
      <div class="hdetail-wrap">
        <h3 style="margin-bottom:4px">🔍 รายละเอียดผลการทดสอบ</h3>
        <p style="font-size:13px;color:#666;margin-bottom:16px">📅 ${h.date}</p>
        <div class="hdetail-total" style="border-color:${riskColor}">
          <div class="hdetail-total-val" style="color:${riskColor}">${riskLevel}</div>
          <div class="hdetail-total-label">${h.riskPercent}%</div>
        </div>
        <div class="hdetail-scores">${scoresHtml}</div>
        <button class="btn btn-primary btn-full mt-16" data-action="close-modal">ปิด</button>
      </div>
    `);
  }

  // ── Modal ─────────────────────────────────────────────────────
  showModal(html: string) {
    this.modalHtml = html;
    this.modalVisible = true;
  }

  closeModal() {
    this.modalVisible = false;
  }

  // ── Template Helpers ──────────────────────────────────────────
  getRingDashArray(): string {
    const r = 60;
    const circ = 2 * Math.PI * r;
    const dash = circ * (this.resultRingPercent / 100);
    return dash + ' ' + circ;
  }
}