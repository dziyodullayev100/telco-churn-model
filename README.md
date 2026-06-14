# 📉 Telco Customer Churn Prediction API

Ushbu loyiha telekommunikatsiya kompaniyasi mijozlarining xizmatdan voz kechish (churn) ehtimolini oldindan bashorat qiluvchi End-to-End Machine Learning tizimidir. Loyiha ma'lumotlarni qayta ishlashdan tortib, uni xizmat sifatida taqdim etuvchi (API) servergacha bo'lgan to'liq jarayonni o'z ichiga oladi.

## 🚀 Loyiha Haqida

Mijozlarni saqlab qolish har qanday biznes uchun eng muhim ko'rsatkichlardan biridir. Ushbu tizim mijozning demografik ma'lumotlari, xizmat turlari va to'lov usullariga asoslanib, uning kompaniyadan ketish xavfini tahlil qiladi. Model o'qitish jarayonida `Cross-Validation` orqali bir nechta algoritmlar sinovdan o'tkazilgan va eng yuqori hamda barqaror aniqlikni ko'rsatgan model tanlab olingan.

### 🛠️ Asosiy Texnologiyalar:
* **Machine Learning:** `scikit-learn`, `pandas`
* **Data Preprocessing:** `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`
* **Backend API:** `FastAPI`, `Pydantic`, `Uvicorn`
* **Model Serialization:** `joblib`

## 📁 Loyiha Strukturasi

Loyiha modulli arxitektura asosida qurilgan bo'lib, quyidagi qismlardan iborat:
* `main.py` - Datasetni o'qitish, turli modellarni musobaqalashtirish va eng yaxshi natija ko'rsatganini avtomatik saqlash (`.pkl` formatida) uchun mas'ul fayl.
* `app.py` - Saqlangan modelni ishga tushiruvchi va foydalanuvchi/mijozlardan so'rovlarni (POST request) qabul qilib, bashorat natijalarini qaytaruvchi FastAPI serveri.
* `model/data/` - Loyihaning ma'lumotlar bazasi (`dataset.csv`) va tayyor o'qitilgan model (`.pkl`) saqlanadigan xavfsiz papka (Xavfsizlik yuzasidan Git'ga yuklanmagan).

## 💻 Qanday Ishga Tushiriladi?

Agar ushbu loyihani o'z kompyuteringizda ishga tushirmoqchi bo'lsangiz, quyidagi qadamlarni bajaring:

1. Repozitoriyni kompyuteringizga ko'chirib oling:
   ```bash
   git clone [https://github.com/dziyodullayev100/telco-churn-model.git](https://github.com/dziyodullayev100/telco-churn-model.git)
