# Jet Classifier (YOLO11s)

## Як запустити

1. Встанови залежності:

```bash
  pip install -r requirements.txt
```

2. Тренування моделі:

```bash
  python train.py
```

3. Web-інтерфейс:

```bash
  python interface.py
```

##  Структура

- `Jet Detection/data.yaml` — конфіг для YOLO
- `train/valid/test` — мають бути всередині `Jet Detection/`