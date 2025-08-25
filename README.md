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

## Variable Orders + Wave-Shaped Martingale

We support two schedule generators for variable orders optimization:

- **anchors** (default): Sample log-volumes at A anchor points (A~4..8), interpolate between them, then run the repair pipeline.
- **blocks**: W wave blocks with base growth and amplitude; then repair.

**Hard constraints**: v0=0.01, indent0=0.0; v1 ≤ v0; front-cap on first K_front orders; local growth bands pre/post normalization; total sum=100%.

**Soft penalties**: second≤first, growth-band violations, frontload excess, total-variation, wave-smoothness; presets: `explore`, `robust`, `tight`.

### Example run

```bash
python -m martingale_lab.cli.optimize \
  --overlap-min 10 --overlap-max 11 \
  --orders-min 12 --orders-max 28 \
  --wave-mode anchors --anchors 6 \
  --g-pre-band 1.01,1.20 --g-post-band 1.01,1.30 \
  --first-volume 0.01 --first-indent 0.0 \
  --front-cap 5.0 --k-front 3 --isotonic-tail off \
  --penalty-preset robust \
  --batches 6 --batch-size 2000 \
  --workers 4 --workers-mode process \
  --prune-threshold 2.0 --grace-batches 2 --patience 10 \
  --log-every-batch 1 --log-eval-sample 0.0 \
  --db db_results/experiments.db --seed 42 \
  --notes "variable-orders anchors"
```

### Inspect best candidate (SQLite)

```sql
WITH latest AS (SELECT MAX(id) id FROM experiments),
best AS (
  SELECT r.payload_json
  FROM results r JOIN latest l ON r.experiment_id=l.id
  ORDER BY r.score ASC LIMIT 1
),
v AS (SELECT CAST(j.key AS INT) i, j.value v
      FROM best, json_each(json_extract(best.payload_json,'$.schedule.volume_pct')) j)
SELECT
  printf('%.5f',(SELECT v FROM v WHERE i=0)) AS v0,
  printf('%.5f',(SELECT v FROM v WHERE i=1)) AS v1,
  (SELECT CASE WHEN (SELECT v FROM v WHERE i=1) <= (SELECT v FROM v WHERE i=0) THEN 'OK' ELSE 'VIOL' END) AS v1_le_v0,
  printf('%.5f',(SELECT SUM(v) FROM v WHERE i IN (0,1,2))) AS first3_sum,
  printf('%.5f',(SELECT SUM(v) FROM v)) AS total_sum;
```

### Quick validation commands

**Anchors mode:**
```bash
python -m martingale_lab.cli.optimize \
  --overlap-min 9.8 --overlap-max 11.2 \
  --orders-min 8 --orders-max 32 \
  --wave-mode anchors --anchors 6 \
  --penalty-preset tight \
  --batches 3 --batch-size 1500 --workers 4 --workers-mode process \
  --db db_results/experiments.db --seed 42 --log-every-batch 1
```

**Blocks mode:**
```bash
python -m martingale_lab.cli.optimize \
  --overlap-min 9.8 --overlap-max 11.2 \
  --orders-min 8 --orders-max 32 \
  --wave-mode blocks --blocks 3 --wave-amp-min 0.05 --wave-amp-max 0.25 \
  --penalty-preset explore \
  --batches 3 --batch-size 1500 --workers 4 --workers-mode process \
  --db db_results/experiments.db --seed 42 --log-every-batch 1
```

**Grid view:**
```sql
WITH latest AS (SELECT MAX(id) id FROM experiments),
rows AS (
  SELECT r.score s,
         json_extract(r.payload_json,'$.overlap') ov,
         json_extract(r.payload_json,'$.orders')  od
  FROM results r JOIN latest l ON r.experiment_id=l.id
)
SELECT ov, od, MIN(s) best_score
FROM rows GROUP BY ov, od ORDER BY best_score ASC LIMIT 20;
```

## Gelecek Geliştirmeler

- [ ] Grafik analizleri
- [ ] Veri dışa aktarma
- [ ] Otomatik raporlama
- [ ] Karşılaştırmalı sonuçlar
- [ ] Performans trendleri
