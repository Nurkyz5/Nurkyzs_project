# Риск автоматизации профессий под воздействием ИИ

Инструмент для HR и L&D: оценивает, какие должности в штате попадают в зону риска автоматизации,
куда можно перевести сотрудников и каким навыкам их учить. Финальный проект курса Data Science.

## Как запустить

```bash
pip install -r requirements.txt
jupyter notebook final_project.ipynb     # весь анализ: EDA → модели → интерпретация
streamlit run app.py                     # демо для HR
```

Ноутбук сам пересобирает датасет (`src/build_dataset.py`) и сохраняет модель в `models/risk_model.joblib`,
которую использует демо.

## Структура

| Путь | Что это |
|---|---|
| `final_project.ipynb` | основной ноутбук |
| `notebook_src.py` | тот же ноутбук в виде скрипта (jupytext) |
| `app.py` | Streamlit-демо: аудит штата, карточка должности, «что если», рынок КР |
| `src/build_dataset.py` | сборка датасета из сырых файлов |
| `src/skills_ru.py` | русские названия признаков O*NET, обучаемые навыки, соответствие трендам WEF |
| `data/raw/` | исходные данные (см. ниже) |
| `data/processed/` | собранный датасет и оценённые должности КР |
| `models/` | обученная модель |

## Данные

| Файл | Источник |
|---|---|
| `frey_osborne.csv` | Frey & Osborne, *The Future of Employment* (2013/2017), 702 профессии; копия из [plotly/datasets](https://github.com/plotly/datasets/blob/master/job-automation-probability.csv) |
| `onet31_*.csv` | [O*NET 31.0](https://www.onetcenter.org/database.html) (оценки обновлены до 08/2026), шкала Importance (IM) — **основная версия** |
| `onet_2010_to_2019_crosswalk.csv` | [официальный кроссвок O*NET-SOC 2010 → 2019](https://www.onetcenter.org/taxonomy/2019/walk.html): перевод кодов Frey & Osborne в коды O*NET 31.0 |
| `onet24_*.txt` | [O*NET 24.0](https://www.onetcenter.org/database.html) (2019) — для сравнения версий и для 8 профессий, ещё не оценённых в 31.0 |
| `gpts_are_gpts_occ_level.csv` | Eloundou et al., *GPTs are GPTs* (2023), [github.com/openai/GPTs-are-GPTs](https://github.com/openai/GPTs-are-GPTs) |
| `anthropic_aei_occupations.csv` | [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex) (CC-BY), сведено по профессиям: `observed_exposure` (labor_market_impacts, март 2026), `robot_exposure_index` (robot_exposure, сентябрь 2026), доля автоматизации и использования (release 2025-03-27) |
| `hh_kg_*.csv` | парсинг hh, вакансии Кыргызстана (area=48), 20 октября 2026: 59 должностей, до 8 вакансий на должность |
| `lalafo_kg_*.csv` | парсинг lalafo.kg, раздел «Работа», 21 октября 2026: первая страница выдачи по 59 должностям, 551 релевантное объявление (без телефонов и имён) |
| — (цитата, без данных) | [Google AI & Economy ATLAS v1.0](https://arxiv.org/abs/2608.00038), июль 2026: 14,7 млн диалогов с Gemini; открытых данных по профессиям нет, используется как внешнее подтверждение выводов |
| `src/skills_ru.py` | тренды навыков: [WEF Future of Jobs 2025](https://www.weforum.org/publications/the-future-of-jobs-report-2025/in-full/3-skills-outlook/) и [LinkedIn Skills on the Rise 2026](https://news.linkedin.com/2026/Skills-on-the-rise-2026) / 2025 |

## Постановка

* Объект — профессия; признаки — важность 161 элемента O*NET (навыки, знания, способности, рабочие действия) + 7 инженерных.
* Цель — класс риска по вероятности Frey & Osborne: низкий < 0,3 ≤ средний < 0,7 ≤ высокий.
* Метрика — macro-F1 (классы несбалансированы, средний класс важен); 80/20 stratified split + 5-fold CV.

## Ограничения

Метки отражают оценку для США 2013 года, а не реальные сокращения; результат — относительный рейтинг уязвимости.
Профили навыков — американские (O*NET), соответствие должностям КР сделано вручную по кодам SOC.
Экспозиция к GenAI показывает, где ИИ изменит работу, а не обязательно сократит её.
