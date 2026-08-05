import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DbService } from '../../services/db.service';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-register',
  templateUrl: 'register.page.html',
  styleUrls: ['register.page.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class RegisterPage {
  name = '';
  age = '';
  gender = '';
  email = '';
  password = '';
  history = 'none';
  toastMessage = '';
  toastType = '';
  toastVisible = false;
  private toastTimeout: any;

  constructor(
    private db: DbService,
    private router: Router,
    private apiService: ApiService
  ) {}

  async doRegister() {
    if (!this.name.trim() || !this.email.trim() || !this.password) {
      this.showToast('กรุณากรอกข้อมูลให้ครบ', 'warn');
      return;
    }

    const emailTrimmed = this.email.trim();
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(emailTrimmed)) {
      this.showToast('รูปแบบอีเมลไม่ถูกต้อง (ต้องมี @ และชื่อโดเมน เช่น user@domain.com)', 'warn');
      return;
    }

    if (this.password.length < 6) {
      this.showToast('รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร', 'warn');
      return;
    }

    const users = this.db.getUsers();
    const existing = Object.values(users).find((u: any) => u.email === emailTrimmed);
    if (existing) {
      this.showToast('อีเมลนี้มีผู้ใช้งานแล้ว', 'warn');
      return;
    }

    const uid = 'u' + Date.now();
    const user = {
      uid,
      name: this.name.trim(),
      age: this.age,
      gender: this.gender,
      email: this.email.trim(),
      password: btoa(this.password),
      history: this.history,
      createdAt: new Date().toISOString()
    };

    users[uid] = user;
    this.db.saveUsers(users);
    this.db.setSession(user);

    // Sync encrypted user registration to backend database synchronously
    try {
      await this.apiService.registerUserOnBackend(
        this.name.trim() || emailTrimmed.split('@')[0],
        this.name.trim(),
        emailTrimmed,
        this.password
      );
    } catch (err) {
      console.warn('[PMDS Backend Reg Sync Note]', err);
    }

    this.showToast('สมัครสมาชิกสำเร็จ 🎉', 'good');
    setTimeout(() => {
      this.router.navigateByUrl('/home');
    }, 600);
  }

  goLogin() {
    this.router.navigateByUrl('/login');
  }

  private showToast(msg: string, type: string) {
    this.toastMessage = msg;
    this.toastType = type;
    this.toastVisible = true;
    clearTimeout(this.toastTimeout);
    this.toastTimeout = setTimeout(() => this.toastVisible = false, 2500);
  }
}