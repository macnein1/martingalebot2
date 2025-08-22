# Martingale Bot 2 - Optimization System

Bu proje, martingale stratejilerini optimize etmek için geliştirilmiş enterprise-seviye bir uygulamadır.

## 🏗️ Mimari Yapı

```
martingalebot2/
├── martingale_lab/          # Core optimization engine
│   ├── core/                # Core types and algorithms
│   ├── optimizer/           # Optimization engines
│   ├── adapters/           # Adapter patterns
│   ├── ui_bridge/          # UI communication layer
│   └── storage/            # Data persistence
└── ui/                     # Streamlit web interface
    ├── components/         # UI components
    ├── pages/             # Application pages
    └── utils/             # Utility functions
```

## 🚀 Kurulum

1. **Dependency'leri yükleyin:**
```bash
pip install -r ui/requirements.txt
```

2. **Uygulamayı başlatın:**
```bash
cd ui
streamlit run main.py
```

## 🎯 Özellikler

### Core Optimization Engine
- **Numba-accelerated** hesaplamalar
- **Batch processing** desteği
- **Multi-threading** optimizasyon
- **Gerçek zamanlı** progress tracking

### UI Features
- **Modern ve responsive** tasarım
- **Parametrik kontrol** arayüzü
- **Canlı progress** gösterimi
- **Detaylı sonuç** analizi
- **Export** özellikleri

### Optimization Parameters
- **Min/Max Overlap**: Yüzdelik overlap aralıkları
- **Min/Max Order**: Sipariş sayısı aralıkları
- **Risk Factor**: Risk toleransı (0.1-5.0)
- **Smoothing Factor**: Yumuşatma katsayısı (0.01-1.0)
- **Tail Weight**: Kuyruk ağırlığı (0.0-1.0)

## 📊 Kullanım

1. **Parameters**: Sol panelde optimizasyon parametrelerini ayarlayın
2. **Start**: "Optimizasyonu Başlat" butonuna tıklayın
3. **Monitor**: Sağ panelde progress ve sistem performansını izleyin
4. **Results**: Optimizasyon tamamlandığında sonuçları inceleyin

## 🔧 Teknik Detaylar

### Bridge Architecture
- **Service Layer**: Session-based optimization management
- **Payload Converters**: Type-safe data conversion
- **Response Builders**: Standardized API responses

### Performance
- **JIT Compilation**: Numba ile optimize edilmiş kernels
- **Parallel Processing**: Multi-core değerlendirme
- **Memory Efficient**: Streaming batch processing

## 📈 Sonuç Metrikleri

- **Total Score**: Genel optimizasyon skoru
- **Max Score**: Maksimum ihtiyaç skoru
- **Variance Score**: Varyans bazlı skor
- **Tail Score**: Kuyruk penaltı skoru
- **Performance Stats**: Hız ve verimlilik metrikleri

## 🛡️ Error Handling

- **Parameter validation** with detailed error messages
- **Session management** with automatic cleanup
- **Graceful degradation** for component failures
- **Mock services** for development testing

## 🔮 Next Steps

- [ ] Real-time WebSocket updates
- [ ] Advanced visualization charts
- [ ] Multi-strategy comparison
- [ ] Historical data integration
- [ ] Export to multiple formats
- [ ] Automated backtesting

---

**Enterprise-level** modular design with clear separation of concerns.
