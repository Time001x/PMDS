import { Component, OnInit, OnDestroy, NgZone } from '@angular/core';
import { Router, ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Subject, of, EMPTY } from 'rxjs';
import { concatMap, catchError, takeUntil } from 'rxjs/operators';
import { DbService } from '../../services/db.service';
import { TestConfigService } from '../../services/test-config.service';
import { ApiService, getRiskLevel, getRiskColor, getRiskLevelInfo, getItemLevel } from '../../services/api.service';
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
  resultLevelNumber = 0;
  resultRiskLevel = 'ไม่เสี่ยง';
  resultColor = '#05C134';
  resultLabel = '';
  resultDescription = '';
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
  getItemLevel = getItemLevel;

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
      const lvlInfo = getRiskLevelInfo(result.level !== undefined ? result.level : 0);
      const totalLevel = lvlInfo.level;
      const riskLevel = lvlInfo.label;
      const riskColor = lvlInfo.color;
      const riskDesc = lvlInfo.desc;

      const record = {
        levelNumber: totalLevel,
        riskPercent: totalLevel * 100,
        riskLevel: riskLevel,
        scores: { ...sessionScores },
        level: riskLevel
      };
      this.db.addHistory(this.currentUser.uid, record);
      this.db.clearSessionScores(this.currentUser.uid);

      this.resultLevelNumber = totalLevel;
      this.resultRingPercent = totalLevel * 100;
      this.resultRiskLevel = riskLevel;
      this.resultColor = riskColor;
      this.resultLabel = riskLevel;
      this.resultDescription = riskDesc;

      const tests = this.testConfig.getTestList();
      this.resultScores = tests.map(t => {
        const sc = sessionScores[t.id] || 0;
        const itemLvl = getItemLevel(sc);
        const info = getRiskLevelInfo(itemLvl);
        return {
          ...t,
          level: info.level,
          label: info.label,
          color: info.color
        };
      });

      this.showToast(`🤖 AI ประเมิน: ระดับ ${totalLevel} (${riskLevel})`, 'teal');

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
    const itemLvl = getItemLevel(totalRisk);
    const lvlInfo = getRiskLevelInfo(itemLvl);

    const record = { levelNumber: lvlInfo.level, riskPercent: lvlInfo.level * 100, riskLevel: lvlInfo.label, scores: { ...sessionScores }, level: lvlInfo.label };
    this.db.addHistory(this.currentUser.uid, record);
    this.db.clearSessionScores(this.currentUser.uid);

    this.resultLevelNumber = lvlInfo.level;
    this.resultRingPercent = lvlInfo.level * 100;
    this.resultRiskLevel = lvlInfo.label;
    this.resultColor = lvlInfo.color;
    this.resultLabel = lvlInfo.label;
    this.resultDescription = lvlInfo.desc;
    this.resultScores = tests.map(t => {
      const sc = sessionScores[t.id] || 0;
      const subLvl = getItemLevel(sc);
      const info = getRiskLevelInfo(subLvl);
      return {
        ...t,
        level: info.level,
        label: info.label,
        color: info.color
      };
    });
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

    const getRecordLevel = (h: any) => {
      if (h.levelNumber !== undefined) return h.levelNumber;
      return getItemLevel(h.riskPercent || 0);
    };

    const levels = hist.map(h => getRecordLevel(h));
    const latestLvl = levels.length > 0 ? levels[0] : 0;
    const avgLvl = levels.length > 0 ? Math.round(levels.reduce((a, b) => a + b, 0) / levels.length) : 0;

    this.historySummary = {
      count: hist.length,
      latest: latestLvl,
      avg: avgLvl
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
    const riskLevel = getRiskLevel(h.riskPercent || h.levelNumber || 0);
    const riskColor = getRiskColor(h.riskPercent || h.levelNumber || 0);
    const lvlNum = getItemLevel(h.riskPercent || h.levelNumber || 0);
    const scoresHtml = tests.map(t => {
      const sc = h.scores?.[t.id];
      const itemLvl = sc !== undefined ? getItemLevel(sc) : null;
      const color = itemLvl === null ? '#ccc' : itemLvl >= 1 ? '#C10508' : '#05C134';
      const label = itemLvl !== undefined && itemLvl !== null ? getRiskLevel(itemLvl) : '-';
      return `<div class="hdetail-score-row">
        <span class="hdsr-icon">${t.icon}</span>
        <span class="hdsr-name">${t.name}</span>
        <div class="hdsr-bar"><div class="hdsr-bar-fill" style="width:${itemLvl !== null ? itemLvl * 100 : 0}%;background:${color}"></div></div>
        <span class="hdsr-val" style="color:${color}">${itemLvl !== null ? 'ระดับ ' + itemLvl + ' (' + label + ')' : '-'}</span>
      </div>`;
    }).join('');

    this.showModal(`
      <div class="hdetail-wrap">
        <h3 style="margin-bottom:4px">🔍 รายละเอียดผลการทดสอบ</h3>
        <p style="font-size:13px;color:#666;margin-bottom:16px">📅 ${h.date}</p>
        <div class="hdetail-total" style="border-color:${riskColor}">
          <div class="hdetail-total-val" style="color:${riskColor}">${riskLevel}</div>
          <div class="hdetail-total-label">ระดับ ${lvlNum} (MDS-UPDRS)</div>
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