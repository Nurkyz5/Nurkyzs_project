"""Русские названия 35 навыков O*NET и список навыков, которым реально можно обучить сотрудника."""
SKILL_RU = {
    "Active Learning": "Активное обучение", "Active Listening": "Активное слушание",
    "Complex Problem Solving": "Решение сложных задач", "Coordination": "Координация с коллегами",
    "Critical Thinking": "Критическое мышление", "Equipment Maintenance": "Обслуживание оборудования",
    "Equipment Selection": "Выбор оборудования", "Installation": "Монтаж", "Instructing": "Обучение других",
    "Judgment and Decision Making": "Принятие решений", "Learning Strategies": "Методы обучения",
    "Management of Financial Resources": "Управление финансами",
    "Management of Material Resources": "Управление ресурсами",
    "Management of Personnel Resources": "Управление людьми", "Mathematics": "Математика",
    "Monitoring": "Контроль качества своей работы", "Negotiation": "Переговоры",
    "Operation Monitoring": "Мониторинг оборудования", "Operation and Control": "Управление оборудованием",
    "Operations Analysis": "Анализ требований", "Persuasion": "Убеждение", "Programming": "Программирование",
    "Quality Control Analysis": "Контроль качества", "Reading Comprehension": "Понимание текстов",
    "Repairing": "Ремонт", "Science": "Научные методы", "Service Orientation": "Клиентоориентированность",
    "Social Perceptiveness": "Эмпатия (социальная восприимчивость)", "Speaking": "Устная речь",
    "Systems Analysis": "Системный анализ", "Systems Evaluation": "Оценка систем",
    "Technology Design": "Проектирование решений", "Time Management": "Тайм-менеджмент",
    "Troubleshooting": "Поиск неисправностей", "Writing": "Письменная речь",
}
# навыки, которые компания может развивать через корпоративное обучение (без технических/ручных)
TRAINABLE = ["Active Learning", "Complex Problem Solving", "Coordination", "Critical Thinking", "Instructing",
             "Judgment and Decision Making", "Management of Financial Resources", "Management of Personnel Resources",
             "Mathematics", "Negotiation", "Operations Analysis", "Persuasion", "Programming",
             "Service Orientation", "Social Perceptiveness", "Speaking", "Systems Analysis", "Systems Evaluation",
             "Technology Design", "Time Management", "Writing", "Learning Strategies"]

# Соответствие навыков O*NET мировым трендам WEF Future of Jobs 2025 (навыки, растущие к 2030 году,
# ключевые навыки 2025 и навыки с падающей важностью). Источник:
# https://www.weforum.org/publications/the-future-of-jobs-report-2025/in-full/3-skills-outlook/
WEF_TREND = {
    "Programming": "растёт", "Technology Design": "растёт", "Systems Analysis": "растёт",
    "Systems Evaluation": "растёт", "Active Learning": "растёт", "Learning Strategies": "растёт",
    "Complex Problem Solving": "ключевой", "Critical Thinking": "ключевой", "Judgment and Decision Making": "ключевой",
    "Operations Analysis": "ключевой", "Social Perceptiveness": "ключевой", "Service Orientation": "ключевой",
    "Persuasion": "ключевой", "Negotiation": "ключевой", "Management of Personnel Resources": "ключевой",
    "Coordination": "ключевой", "Instructing": "ключевой",
    "Writing": "снижается", "Mathematics": "снижается",
}
WEF_WEIGHT = {"растёт": 1.5, "ключевой": 1.2, "снижается": 0.7}

# Соответствие навыков O*NET мировым трендам WEF Future of Jobs 2025 (навыки, растущие к 2030 году,
# ключевые навыки 2025 и навыки с падающей важностью). Источник:
# https://www.weforum.org/publications/the-future-of-jobs-report-2025/in-full/3-skills-outlook/
WEF_TREND = {
    "Programming": "растёт", "Technology Design": "растёт", "Systems Analysis": "растёт",
    "Systems Evaluation": "растёт", "Active Learning": "растёт", "Learning Strategies": "растёт",
    "Complex Problem Solving": "ключевой", "Critical Thinking": "ключевой", "Judgment and Decision Making": "ключевой",
    "Operations Analysis": "ключевой", "Social Perceptiveness": "ключевой", "Service Orientation": "ключевой",
    "Persuasion": "ключевой", "Negotiation": "ключевой", "Management of Personnel Resources": "ключевой",
    "Coordination": "ключевой", "Instructing": "ключевой",
    "Writing": "снижается", "Mathematics": "снижается",
}
WEF_WEIGHT = {"растёт": 1.5, "ключевой": 1.2, "снижается": 0.7}

KNOWLEDGE_RU = {
    "Administration and Management": "Администрирование и менеджмент", "Biology": "Биология",
    "Building and Construction": "Строительство", "Chemistry": "Химия", "Clerical": "Делопроизводство",
    "Communications and Media": "Коммуникации и медиа", "Computers and Electronics": "Компьютеры и электроника",
    "Customer and Personal Service": "Клиентский сервис", "Design": "Проектирование и дизайн",
    "Economics and Accounting": "Экономика и бухучёт", "Education and Training": "Педагогика",
    "Engineering and Technology": "Инженерия и технологии", "English Language": "Родной язык (грамотность)",
    "Fine Arts": "Искусство", "Food Production": "Пищевое производство", "Foreign Language": "Иностранный язык",
    "Geography": "География", "History and Archeology": "История", "Law and Government": "Право и госуправление",
    "Mathematics": "Математика (знания)", "Mechanical": "Механика", "Medicine and Dentistry": "Медицина",
    "Personnel and Human Resources": "Управление персоналом (HR)", "Philosophy and Theology": "Философия",
    "Physics": "Физика", "Production and Processing": "Производство", "Psychology": "Психология",
    "Public Safety and Security": "Безопасность", "Sales and Marketing": "Продажи и маркетинг",
    "Sociology and Anthropology": "Социология", "Telecommunications": "Телекоммуникации",
    "Therapy and Counseling": "Консультирование и терапия", "Transportation": "Транспорт",
}
ABILITY_RU = {
    "Arm-Hand Steadiness": "Твёрдость руки", "Auditory Attention": "Слуховое внимание",
    "Category Flexibility": "Гибкость классификации", "Control Precision": "Точность управления",
    "Deductive Reasoning": "Дедуктивное мышление", "Depth Perception": "Восприятие глубины",
    "Dynamic Flexibility": "Динамическая гибкость тела", "Dynamic Strength": "Динамическая сила",
    "Explosive Strength": "Взрывная сила", "Extent Flexibility": "Растяжка тела", "Far Vision": "Зрение вдаль",
    "Finger Dexterity": "Ловкость пальцев", "Flexibility of Closure": "Поиск образа в шуме",
    "Fluency of Ideas": "Беглость идей", "Glare Sensitivity": "Устойчивость к бликам",
    "Gross Body Coordination": "Координация тела", "Gross Body Equilibrium": "Равновесие",
    "Hearing Sensitivity": "Острота слуха", "Inductive Reasoning": "Индуктивное мышление",
    "Information Ordering": "Упорядочивание информации", "Manual Dexterity": "Ручная ловкость",
    "Mathematical Reasoning": "Математическое мышление", "Memorization": "Запоминание",
    "Multilimb Coordination": "Координация конечностей", "Near Vision": "Зрение вблизи", "Night Vision": "Ночное зрение",
    "Number Facility": "Быстрый счёт", "Oral Comprehension": "Понимание устной речи", "Oral Expression": "Устное изложение",
    "Originality": "Оригинальность", "Perceptual Speed": "Скорость восприятия", "Peripheral Vision": "Периферийное зрение",
    "Problem Sensitivity": "Чувствительность к проблемам", "Rate Control": "Контроль скорости",
    "Reaction Time": "Время реакции", "Response Orientation": "Выбор реакции", "Selective Attention": "Избирательное внимание",
    "Sound Localization": "Локализация звука", "Spatial Orientation": "Ориентация в пространстве",
    "Speech Clarity": "Чёткость речи", "Speech Recognition": "Распознавание речи", "Speed of Closure": "Быстрое схватывание",
    "Speed of Limb Movement": "Скорость движений", "Stamina": "Выносливость", "Static Strength": "Статическая сила",
    "Time Sharing": "Многозадачность", "Trunk Strength": "Сила корпуса", "Visual Color Discrimination": "Различение цветов",
    "Visualization": "Пространственное воображение", "Wrist-Finger Speed": "Скорость кисти и пальцев",
    "Written Comprehension": "Понимание письменных текстов", "Written Expression": "Письменное изложение",
}
ACTIVITY_RU = {
    "Analyzing Data or Information": "Анализ данных", "Assisting and Caring for Others": "Забота о людях",
    "Coaching and Developing Others": "Коучинг и развитие других",
    "Communicating with Persons Outside Organization": "Коммуникация с внешними контактами",
    "Communicating with Supervisors, Peers, or Subordinates": "Коммуникация внутри команды",
    "Controlling Machines and Processes": "Управление машинами и процессами",
    "Coordinating the Work and Activities of Others": "Координация работы других",
    "Developing Objectives and Strategies": "Разработка целей и стратегий", "Developing and Building Teams": "Построение команд",
    "Documenting/Recording Information": "Документирование и учёт",
    "Drafting, Laying Out, and Specifying Technical Devices, Parts, and Equipment": "Техническое черчение",
    "Establishing and Maintaining Interpersonal Relationships": "Построение отношений",
    "Estimating the Quantifiable Characteristics of Products, Events, or Information": "Количественные оценки",
    "Evaluating Information to Determine Compliance with Standards": "Проверка соответствия стандартам",
    "Getting Information": "Сбор информации", "Guiding, Directing, and Motivating Subordinates": "Руководство и мотивация",
    "Handling and Moving Objects": "Перемещение предметов", "Identifying Objects, Actions, and Events": "Распознавание объектов и событий",
    "Inspecting Equipment, Structures, or Material": "Осмотр оборудования", "Interacting With Computers": "Работа с компьютером",
    "Interpreting the Meaning of Information for Others": "Объяснение информации другим",
    "Judging the Qualities of Things, Services, or People": "Оценка качества", "Making Decisions and Solving Problems": "Решение проблем",
    "Monitor Processes, Materials, or Surroundings": "Мониторинг процессов", "Monitoring and Controlling Resources": "Контроль ресурсов",
    "Operating Vehicles, Mechanized Devices, or Equipment": "Управление транспортом и техникой",
    "Organizing, Planning, and Prioritizing Work": "Планирование работы", "Performing Administrative Activities": "Административная работа",
    "Performing General Physical Activities": "Физическая работа", "Performing for or Working Directly with the Public": "Работа с публикой",
    "Processing Information": "Обработка информации", "Provide Consultation and Advice to Others": "Консультирование",
    "Repairing and Maintaining Electronic Equipment": "Ремонт электроники", "Repairing and Maintaining Mechanical Equipment": "Ремонт механизмов",
    "Resolving Conflicts and Negotiating with Others": "Разрешение конфликтов", "Scheduling Work and Activities": "Составление графиков",
    "Selling or Influencing Others": "Продажи и влияние", "Staffing Organizational Units": "Подбор персонала",
    "Thinking Creatively": "Креативное мышление", "Training and Teaching Others": "Обучение других (деятельность)",
    "Updating and Using Relevant Knowledge": "Обновление профессиональных знаний",
}

def feature_ru(col):
    """Русское название признака по имени столбца (sk_/kn_/ab_/wa_/fe_)."""
    p, name = col[:3], col[3:]
    eng = {"fe_social": "Индекс: социальный интеллект", "fe_creative": "Индекс: креативность",
           "fe_manual": "Индекс: ручной труд", "fe_routine_info": "Индекс: рутинная обработка информации",
           "fe_complex_cognitive": "Индекс: сложное мышление", "fe_log_wage": "log(зарплата)",
           "fe_education": "Уровень образования"}
    if col in eng: return eng[col]
    return {"sk_": SKILL_RU, "kn_": KNOWLEDGE_RU, "ab_": ABILITY_RU, "wa_": ACTIVITY_RU}.get(p, {}).get(name, name)

# Соответствие навыков O*NET растущим навыкам LinkedIn («Skills on the Rise» 2025 и 2026,
# данные профилей и найма в 12 странах, дек. 2024 – нояб. 2025). Источники:
# https://news.linkedin.com/2026/Skills-on-the-rise-2026
# https://gulfnews.com/business/markets/linkedin-unveils-skills-on-the-rise-2025-15-must-have-skills-for-the-future-1.500067061
LINKEDIN_TREND = {
    "Complex Problem Solving": "Problem-Solving",
    "Systems Analysis": "Strategic Thinking",
    "Judgment and Decision Making": "Strategic Thinking",
    "Speaking": "Communication, Public Speaking",
    "Writing": "Communication",
    "Active Learning": "Adaptability, AI Literacy",
    "Programming": "LLM, Prompt Engineering, Code Review, Debugging",
    "Technology Design": "LLM, Prompt Engineering",
    "Service Orientation": "Customer Engagement",
    "Mathematics": "Statistical Data Analysis",
    "Operations Analysis": "Market Analysis",
    "Coordination": "Stakeholder Management, Cross-Functional Collaboration",
    "Management of Personnel Resources": "Team Management",
    "Instructing": "Mentorship",
    "Persuasion": "Business Development, Go-to-Market Strategy",
    "Negotiation": "Business Development",
    "Systems Evaluation": "Governance, Risk Management and Compliance",
}
LINKEDIN_WEIGHT = 1.15

def trend_weight(skill):
    """Итоговый вес навыка в рекомендациях: тренд WEF × бонус, если навык растёт по данным LinkedIn."""
    w = WEF_WEIGHT.get(WEF_TREND.get(skill), 1.0)
    return w * (LINKEDIN_WEIGHT if skill in LINKEDIN_TREND else 1.0)
