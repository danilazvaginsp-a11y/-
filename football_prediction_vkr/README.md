# Football Prediction VKR

Проект ВКР: применение методов машинного обучения для предсказания исходов футбольных матчей (победа хозяев / ничья / победа гостей) на данных трех лиг:
- English Premier League
- La Liga
- Bundesliga

## Структура

```text
football_prediction_vkr/
├── data/
│   └── dataset.csv
├── models/
│   └── catboost_model.joblib
├── notebooks/
│   └── 01_eda_and_prototyping.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── build_dataset.py
│   └── training.py
├── app.py
├── PRESENTATION_GUIDE.txt
├── requirements.txt
└── README.md
```

## Требования

- Python 3.12 (поддерживается) или Python 3.9+
- Исходные CSV-файлы матчей и таблиц статистики могут лежать в одном из мест:
  - `football_prediction_vkr/data/` (рекомендуется)
  - родительская папка проекта (`../` относительно `football_prediction_vkr`)
  - АПЛ: `E0 (1).csv`, `E0.csv`
  - Ла Лига: `SP1 (1).csv`, `SP1.csv`
  - Бундеслига: `D1 (1).csv`, `D1.csv`
  - xG-статистика Ла Лиги: `league-chemp (1).csv`, `league-chemp.csv`
  - xG-статистика Бундеслиги: `league-chemp (3).csv`, `league-chemp (2).csv`

## Установка (Git Bash)

```bash
python3 -m pip install -r ./requirements.txt
```

## Сборка единого датасета (Git Bash)

```bash
python3 ./src/build_dataset.py
```

Скрипт объединяет матчи за два сезона, добавляет метки лиги/сезона, подключает сезонные признаки `xG/xGA/xPTS` для домашних и гостевых команд и сохраняет `data/dataset.csv`.

## Обучение модели (Git Bash)

```bash
python3 ./src/training.py
```

После обучения артефакт модели будет сохранен в `models/catboost_model.joblib`.

## Запуск Streamlit-приложения (Git Bash)

```bash
python3 -m streamlit run ./app.py
```

## Что реализовано

- Ликбезопасная генерация rolling-признаков по лиге и команде
- Хронологическое разделение данных 80/20
- Обучение `CatBoostClassifier` с категориальными признаками (`home_team`, `away_team`, `league`, `season`)
- Метрики качества (`accuracy`, `log_loss`)
- Интеграция сезонных статистик `xG`, `xGA`, `xPTS` и их разностей
- Прогноз для выбранной лиги и команд
- Сравнение выбранных команд (таблица по последним 10 матчам)
- Статистика команды по последним матчам и график формы
- SHAP waterfall-график и текстовая расшифровка факторов прогноза

