# AML Mule Detection -- MLOps проект


## Структура

```
config/               конфиг обучения (пути, гиперпараметры, пороги мониторинга)
src/aml_mule/         пайплайн: data.py -> features.py -> train.py -> inference.py
api/                  FastAPI-сервис инференса
scripts/              генерация синтетических данных, CLI-проверка дрейфа
tests/                pytest: юнит-тесты + интеграционный тест train->inference->API
notebooks/            исходный research-ноутбук (EDA, обоснование метрик и архитектуры)
.github/workflows/    CI: тесты, lint, сквозной прогон обучения на синтетике, сборка Docker
Dockerfile            self-contained образ: обучится сам при первом запуске, если нет артефакта
```

## Почему модель устроена именно так

Коротко (подробности и все проверки -- в ноутбуке, разделы 7--11):

- `Is_Mule` восстанавливается из префикса `Customer_ID`; строки без ID
  удаляются
- Модель -- ансамбль "defense in depth": `primary_model` (RandomForest,
  `max_features=0.5`, чтобы не полагаться только на 2 доминирующих
  признака) + `fallback_model` (обучен без этих двух признаков вообще).
  Итоговый скор -- `max(primary, fallback)`.
- Метрика выбора модели -- PR-AUC.
  Порог подбирается по F2 (recall важнее precision).
- В ноутбуке (раздел 11) отдельно доказано: слабость на паре MULE/TRADER --
  свойство датасета, а не конкретного алгоритма

## Быстрый старт (локально)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# синтетические данные для пробного запуска (замените на реальный data.csv,
# когда он будет -- он никогда не коммитится в репозиторий)
PYTHONPATH=src python scripts/generate_synthetic_data.py --out data/data.csv

PYTHONPATH=src python -m aml_mule.train --config config/config.yaml

PYTHONPATH=src uvicorn api.main:app --reload
# -> http://localhost:8000/docs
```

## Тесты

```bash
PYTHONPATH=src pytest -q
```

Покрытие: восстановление меток, препроцессинг без утечки, feature
engineering, PSI-мониторинг, полный цикл train -> inference -> API на
синтетических данных.

## Docker

```bash
docker build -t aml-mule-detection .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts aml-mule-detection
```

Если `artifacts/aml_mule_pipeline_latest.joblib` отсутствует, контейнер
сам сгенерирует синтетические данныеы и обучится при первом запуске -- см. `entrypoint.sh`.

## API

- `GET /health` -- жив ли сервис, загружена ли модель
- `POST /predict` -- батч транзакций -> вероятность, метка, топ-3 SHAP-фактора на клиента
- `POST /monitor/drift` -- батч новых записей -> PSI по каждому признаку

Пример:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "records": [{"Account_Age_Days": 25, "Transaction_Volume_USD": 42000,
               "Avg_Time_Between_Trans_Min": 9, "Flags_Last_6M": 2,
               "Avg_End_Day_Balance": 120}]
}'
```

## Мониторинг дрейфа

```bash
PYTHONPATH=src python scripts/check_drift.py \
  --artifact artifacts/aml_mule_pipeline_latest.joblib \
  --batch data/new_batch.csv
```

PSI считается по каждому признаку модели относительно сэмпла train,
сохранённого прямо в артефакте (`reference_sample`). Скрипт возвращает
ненулевой код выхода при критическом дрейфе (PSI >= 0.2) -- годится как
шаг в cron/CI для алерта "пора пересмотреть модель".

## CI/CD

`.github/workflows/ci.yml`:
1. `test` -- lint (`ruff`) + `pytest`.
2. `smoke-train` -- генерирует синтетику и прогоняет обучение целиком
   end-to-end (реальные данные в репозиторий никогда не попадают), затем
   гоняет `check_drift.py` на них же (дрейфа быть не должно -- проверка,
   что скрипт вообще работает).
3. `docker-build` -- собирает образ, чтобы Dockerfile не протух незаметно.

