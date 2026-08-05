import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { DbService } from '../../services/db.service';
import { TestConfigService } from '../../services/test-config.service';
import { getRiskLevel, getRiskColor, getItemLevel } from '../../services/api.service';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
  standalone: true,
  imports: [CommonModule]
})
export class HomePage implements OnInit {
  currentUser: any = null;
  riskLevel = 'ยังไม่มีข้อมูล';
  riskColor = '#888';
  riskPercent: number | string = '--';
  totalTests = 0;
  lastResult: any = null;
  testBars: any[] = [];
  recentHistory: any[] = [];

  getRiskLevel = getRiskLevel;
  getRiskColor = getRiskColor;
  getItemLevel = getItemLevel;

  constructor(
    private db: DbService,
    private testConfig: TestConfigService,
    private router: Router
  ) {}

  ngOnInit() {
    const session = this.db.getSession();
    if (!session) {
      this.router.navigateByUrl('/login');
      return;
    }
    this.currentUser = session;
    this.loadData();
  }

  // ── แก้บั๊ก: โหลดข้อมูลใหม่ทุกครั้งที่เข้าหน้านี้ ──────────
  ionViewWillEnter() {
    const session = this.db.getSession();
    if (!session) {
      this.router.navigateByUrl('/login');
      return;
    }
    // อัปเดต currentUser ทุกครั้ง เพื่อให้ได้ชื่อล่าสุด
    this.currentUser = session;
    this.loadData();
  }

  loadData() {
    if (!this.currentUser) return;
    const hist = this.db.getHistory(this.currentUser.uid);
    this.lastResult = hist[0] || null;
    this.totalTests = hist.length;

    if (this.lastResult) {
      this.riskPercent = this.lastResult.riskPercent;
      const riskValue = typeof this.riskPercent === 'number' ? this.riskPercent : 0;
      this.riskLevel = getRiskLevel(riskValue);
      this.riskColor = getRiskColor(riskValue);
    } else {
      this.riskPercent = '--';
      this.riskColor = '#888';
      this.riskLevel = 'ยังไม่มีข้อมูล';
    }

    const tests = this.testConfig.getTestList();
    this.testBars = tests.map(t => {
      const sc = this.lastResult?.scores?.[t.id];
      const pct = sc !== undefined ? Math.round(sc * 100) : 0;
      const color = pct > 60 ? '#C10508' : pct > 35 ? '#F59E0B' : '#05C134';
      const riskLabel = getRiskLevel(pct);
      return { ...t, pct, riskLabel, color };
    });

    this.recentHistory = hist.slice(0, 3).map((h: any) => {
      const isRisk = h.riskPercent >= 50;
      return {
        date: h.date,
        riskPercent: h.riskPercent,
        riskLevel: getRiskLevel(h.riskPercent),
        isRisk,
        badgeClass: isRisk ? 'badge-risk' : 'badge-normal',
        badgeText: isRisk ? 'เสี่ยง' : 'ปกติ'
      };
    });
  }

  navTo(page: string) {
    this.router.navigateByUrl('/' + page);
  }

  refreshHome() {
    const session = this.db.getSession();
    if (session) this.currentUser = session;
    this.loadData();
  }

  getRiskValue(): number {
    return typeof this.riskPercent === 'number' ? this.riskPercent : 0;
  }
}