# Martingale Optimizer UI

Bu modül, Martingale Optimizer uygulamasının kullanıcı arayüzünü içerir.

## Yapı

```
ui/
├── main.py                 # Ana uygulama giriş noktası
├── components/             # Yeniden kullanılabilir UI bileşenleri
│   ├── parameter_inputs.py    # Parametre giriş bileşenleri
│   ├── system_performance.py  # Sistem performans monitörü
│   ├── progress_section.py    # İlerleme takibi
│   ├── results_section.py     # Sonuç gösterimi
│   └── sidebar.py             # Kenar çubuğu navigasyonu
├── pages/                  # Sayfa bileşenleri
│   ├── calculation_page.py    # Hesaplama sayfası
│   └── results_page.py        # Sonuçlar sayfası
├── utils/                  # Yardımcı fonksiyonlar
│   └── config.py             # Konfigürasyon ve stil ayarları
└── requirements.txt        # Bağımlılıklar
```

## Kurulum

```bash
cd ui
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run main.py
```

## Özellikler

- **Modüler Yapı**: Her bileşen ayrı dosyalarda tanımlanmıştır
- **Responsive Tasarım**: Farklı ekran boyutlarına uyumlu
- **Gerçek Zamanlı Sistem Monitörü**: CPU ve RAM kullanımını takip eder
- **İlerleme Takibi**: Optimizasyon sürecini görsel olarak takip eder
- **Sonuç Gösterimi**: Optimizasyon sonuçlarını düzenli şekilde gösterir

## Gelecek Geliştirmeler

- [ ] Grafik analizleri
- [ ] Veri dışa aktarma
- [ ] Otomatik raporlama
- [ ] Karşılaştırmalı sonuçlar
- [ ] Performans trendleri
