# Advanced Features Implementation Plan

## Phase 1: Kelly Criterion Integration (2 saat)

### Amaç
Her order için matematiksel olarak optimal volume yüzdesi hesaplama

### Adımlar
1. `martingale_lab/core/kelly.py` modülü oluştur
2. Kelly fraction hesaplama fonksiyonu ekle
3. `evaluation_engine.py`'ye Kelly-adjusted score ekle
4. Test ve validasyon

### Kod Lokasyonları
- Yeni dosya: `martingale_lab/core/kelly.py`
- Güncelleme: `martingale_lab/optimizer/evaluation_engine.py`

### Risk
- Düşük risk, mevcut sisteme dokunmuyor
- Opsiyonel parametre olarak eklenecek

---

## Phase 2: Bayesian Optimization (3 saat)

### Amaç
Random search yerine akıllı parametre arama

### Adımlar
1. `scikit-optimize` kütüphanesi ekle
2. `martingale_lab/optimizer/bayesian_optimizer.py` oluştur
3. Gaussian Process tabanlı arama implementasyonu
4. `orchestrator`'a `--optimizer bayesian` flag ekle

### Kod Lokasyonları
- Yeni dosya: `martingale_lab/optimizer/bayesian_optimizer.py`
- Güncelleme: `martingale_lab/orchestrator/adaptive_orchestrator.py`
- Güncelleme: `martingale_lab/cli/optimize.py`

### Risk
- Düşük risk, opsiyonel özellik
- Default hala random search kalacak

---

## Phase 3: Portfolio Metrics (1 saat)

### Amaç
Risk-adjusted return metrikleri ekle

### Adımlar
1. `martingale_lab/core/portfolio_metrics.py` oluştur
2. Sortino ratio, Calmar ratio, Information ratio ekle
3. `evaluation_engine`'e entegre et
4. SQL sorgularına yeni metrikler ekle

### Kod Lokasyonları
- Yeni dosya: `martingale_lab/core/portfolio_metrics.py`
- Güncelleme: `martingale_lab/optimizer/evaluation_engine.py`
- Güncelleme: `sql_queries.sql`

### Risk
- Çok düşük risk, sadece yeni metrikler

---

## Phase 4: Smart Initial Generation (2 saat)

### Amaç
Random yerine daha akıllı başlangıç değerleri

### Adımlar
1. Historical iyi stratejilerden öğrenme
2. Markov Chain Monte Carlo (MCMC) sampling
3. Genetic algorithm başlangıç populasyonu
4. `evaluation_engine`'de initial generation güncelleme

### Kod Lokasyonları
- Yeni dosya: `martingale_lab/core/smart_init.py`
- Güncelleme: `martingale_lab/optimizer/evaluation_engine.py`

### Risk
- Orta risk, initial generation'ı değiştiriyor
- Fallback mekanizması eklenecek

---

## Test Stratejisi

### Unit Tests
```python
# test_kelly.py
def test_kelly_criterion():
    win_prob = 0.6
    win_loss_ratio = 1.5
    kelly_fraction = calculate_kelly(win_prob, win_loss_ratio)
    assert 0 < kelly_fraction < 1
```

### Integration Tests
```bash
# Mevcut sistemle karşılaştırma
python -m martingale_lab.cli.optimize --db baseline.db
python -m martingale_lab.cli.optimize --use-kelly --db kelly.db
python compare_results.py baseline.db kelly.db
```

### Performance Tests
```python
# Bayesian vs Random search
# 100 evaluation'da hangisi daha iyi sonuç buluyor?
```

---

## Başarı Kriterleri

1. **Kelly Criterion**
   - Volume distribution daha optimal
   - Score improvement > 5%

2. **Bayesian Optimization**
   - 50% daha az evaluation'da aynı score
   - Convergence hızı 2x

3. **Portfolio Metrics**
   - Yeni metrikler ile daha iyi strateji seçimi
   - SQL sorgularında kullanılabilir

4. **Smart Initial Generation**
   - İlk 10 evaluation'da en az 1 iyi sonuç
   - Inf score oranı < 5%

---

## Rollback Planı

Her feature için:
1. Feature flag ile kontrol
2. Eski kod path korunacak
3. A/B testing ile validasyon
4. Problem durumunda flag kapatma

```python
# config.py
FEATURES = {
    'use_kelly': False,
    'use_bayesian': False,
    'use_smart_init': False,
    'use_portfolio_metrics': True
}
```

---

## Zaman Çizelgesi

| Gün | Feature | Test | Deploy |
|-----|---------|------|--------|
| 1 | Kelly Criterion | ✓ | ✓ |
| 1 | Portfolio Metrics | ✓ | ✓ |
| 2 | Bayesian Optimization | ✓ | ✓ |
| 3 | Smart Initial Generation | ✓ | ✓ |
| 4 | Full Integration Test | ✓ | ✓ |

Total: 4 gün (part-time çalışma)

---

## Öncelik Sırası

1. **Portfolio Metrics** - En kolay, hemen fayda
2. **Kelly Criterion** - Orta zorluk, yüksek fayda
3. **Bayesian Optimization** - Zor ama çok faydalı
4. **Smart Initial Generation** - Opsiyonel iyileştirme