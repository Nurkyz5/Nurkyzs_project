"""Демо: аудит штата на риск автоматизации ИИ и план переобучения.
Запуск:  streamlit run app.py
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib, streamlit as st, altair as alt

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))
from skills_ru import SKILL_RU, TRAINABLE, WEF_TREND, WEF_WEIGHT, LINKEDIN_TREND, trend_weight

st.set_page_config(page_title="AI Risk для HR", layout="wide")
CLASS_RU = {"low": "низкий", "medium": "средний", "high": "высокий"}
COLORS = {"низкий": "#f5c4a8", "средний": "#eb6834", "высокий": "#a33a12"}


@st.cache_resource
def load():
    art = joblib.load(ROOT / "models" / "risk_model.joblib")
    occ = pd.read_csv(ROOT / "data" / "processed" / "occupations_model_ready.csv").set_index("soc")
    kg = pd.read_csv(ROOT / "data" / "processed" / "kg_positions_scored.csv")
    vac = pd.read_csv(ROOT / "data" / "raw" / "hh_kg_vacancies.csv")
    return art, occ, kg, vac


art, occ, kg, vac = load()
model, FEATS, COMP = art["model"], art["features"], art["composites"]
ONET = [f for f in FEATS if f[:3] in ("sk_", "kn_", "ab_", "wa_")]
mu, sd = occ[ONET].mean(), occ[ONET].std()
kg = kg[kg["soc"].isin(occ.index)].drop_duplicates("query")
POSITIONS = kg["query"].tolist()
SOC = dict(zip(kg["query"], kg["soc"]))


def add_composites(X):
    X = X.copy()
    for k, cols in COMP.items():
        X[k] = X[cols].mean(axis=1)
    return X


def proba(X):
    p = model.predict_proba(X[FEATS])
    return pd.DataFrame(p, columns=[CLASS_RU[c] for c in model.classes_], index=X.index)


def transition_targets(soc, k=3, group="bank"):
    cand = kg[(kg["pred_class"] == "low") & (kg["soc"] != soc) & (kg["group"] == group)].drop_duplicates("soc")
    a = ((occ.loc[soc, ONET] - mu) / sd).values
    b = ((occ.loc[cand["soc"], ONET] - mu) / sd).values
    sim = b @ a / (np.linalg.norm(b, axis=1) * np.linalg.norm(a))
    return cand.assign(similarity=sim).sort_values("similarity", ascending=False).head(k)


def skill_gap(soc_from, soc_to, k=4):
    gap = pd.Series({s: occ.loc[soc_to, "sk_" + s] - occ.loc[soc_from, "sk_" + s] for s in TRAINABLE})
    w = pd.Series({s: trend_weight(s) for s in TRAINABLE})
    top = (gap.clip(lower=0) * w).sort_values(ascending=False).head(k).index
    return pd.DataFrame({"навык": [SKILL_RU[s] for s in top], "нужно поднять на": gap[top].round(1).values,
                         "WEF 2025": [WEF_TREND.get(s, "—") for s in top],
                         "LinkedIn 2025–26": [LINKEDIN_TREND.get(s, "—") for s in top]})


def hh_skills(position, k=10):
    s = vac.loc[vac["query"] == position, "skills"].dropna().str.split("|").explode().str.strip()
    return s.value_counts().head(k)


def kgs(x):
    return "—" if pd.isna(x) else f"{x:,.0f} сом".replace(",", " ")


st.title("Риск автоматизации ИИ: аудит штата и план переобучения")
st.caption("Модель обучена на мировых данных (Frey & Osborne × O*NET 31.0, ~700 профессий); экспозиция к GenAI — OpenAI/UPenn 2023; "
           "реальное использование ИИ и роботы — Anthropic Economic Index 2025–2026; "
           "рынок — вакансии hh и lalafo по Кыргызстану, октябрь 2026; тренды навыков — WEF 2025 и LinkedIn 2025–2026.")

tab1, tab2, tab3, tab4 = st.tabs(["Аудит штата", "Должность", "Что если", "Рынок КР"])

# ---------------- 1. Аудит штата ----------------
with tab1:
    st.subheader("Загрузите штатное расписание или отредактируйте пример")
    demo = pd.DataFrame({"position": ["Операционист", "Кассир", "Оператор call-центра", "Кредитный специалист",
                                      "Кредитный аналитик", "Коллектор", "Бухгалтер", "Финансовый аналитик",
                                      "Риск-менеджер", "Комплаенс", "Аналитик данных", "Программист", "Юрист",
                                      "HR менеджер", "Маркетолог"],
                         "headcount": [70, 25, 40, 45, 12, 15, 18, 8, 7, 6, 6, 20, 6, 6, 6]})
    up = st.file_uploader("CSV со столбцами position, headcount", type="csv")
    staff = pd.read_csv(up) if up else demo
    staff = st.data_editor(staff, num_rows="dynamic", width="stretch",
                           column_config={"position": st.column_config.SelectboxColumn("Должность", options=POSITIONS),
                                          "headcount": st.column_config.NumberColumn("Сотрудников", min_value=0)})
    staff = staff.dropna().query("position in @POSITIONS")
    if len(staff):
        X = occ.loc[staff["position"].map(SOC)]
        p = proba(X)
        rep = staff.reset_index(drop=True).copy()
        rep["класс риска"] = p.idxmax(axis=1).values
        rep["P(высокий)"] = p["высокий"].round(2).values
        rep["GenAI"] = X["genai_exposure_human"].round(2).values
        rep["ИИ сейчас"] = X["observed_exposure"].round(2).values
        rep["роботы"] = X["robot_exposure_index"].round(2).values
        tg = [transition_targets(SOC[pos], 1) if c != "низкий" else None for pos, c in zip(rep["position"], rep["класс риска"])]
        rep["куда переводить"] = [t["query"].iloc[0] if t is not None and len(t) else "—" for t in tg]
        rep["чему учить"] = [", ".join(skill_gap(SOC[pos], t["soc"].iloc[0], 3)["навык"]) if t is not None and len(t) else "—"
                             for pos, t in zip(rep["position"], tg)]
        total = rep["headcount"].sum()
        by = rep.groupby("класс риска")["headcount"].sum().reindex(["высокий", "средний", "низкий"]).fillna(0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Сотрудников в зоне высокого риска", f"{by['высокий']:.0f}", f"{by['высокий'] / total:.0%} штата", delta_color="off")
        c2.metric("Средний риск", f"{by['средний']:.0f}", f"{by['средний'] / total:.0%} штата", delta_color="off")
        c3.metric("Высокая экспозиция к GenAI (> 0,5)", f"{rep.loc[rep['GenAI'] > 0.5, 'headcount'].sum():.0f}",
                  "задачи изменятся с ИИ", delta_color="off")
        st.dataframe(rep.sort_values("P(высокий)", ascending=False), width="stretch", hide_index=True)
        st.download_button("Скачать отчёт (CSV)", rep.to_csv(index=False).encode("utf-8-sig"), "ai_risk_audit.csv")

# ---------------- 2. Должность ----------------
with tab2:
    pos = st.selectbox("Должность", POSITIONS, index=POSITIONS.index("Операционист") if "Операционист" in POSITIONS else 0)
    soc = SOC[pos]
    p = proba(occ.loc[[soc]]).iloc[0]
    row = kg.set_index("query").loc[pos]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Класс риска", p.idxmax())
    c2.metric("Экспозиция к GenAI", f"{occ.loc[soc, 'genai_exposure_human']:.2f}")
    c3.metric("Вакансий в КР (hh + lalafo)", f"{int(row['vacancies_total'])}", f"hh {int(row['vacancies_kg'])} · lalafo {int(row['vacancies_lalafo'])}", delta_color="off")
    c4.metric("Медианная зарплата", kgs(row["median_salary_kgs"]))
    fmt = lambda v, f="{:.2f}": "—" if pd.isna(v) else f.format(v)
    d1, d2, d3 = st.columns(3)
    d1.metric("ИИ уже делает задач (2026)", fmt(occ.loc[soc, "observed_exposure"], "{:.0%}"),
              help="Anthropic Economic Index: доля задач профессии, реально выполняемых с Claude")
    d2.metric("Доля автоматизации в работе с ИИ", fmt(occ.loc[soc, "aei_automation_share"], "{:.0%}"),
              help="Сколько использования ИИ — «сделай за меня», а не «помоги мне»")
    d3.metric("Роботы (0–3, 2026)", fmt(occ.loc[soc, "robot_exposure_index"]),
              help="Anthropic: насколько физические задачи профессии уже могут делать роботы")
    pdf = p.reindex(["низкий", "средний", "высокий"]).rename_axis("класс").reset_index(name="вероятность")
    st.altair_chart(alt.Chart(pdf).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("вероятность:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("класс:N", sort=None),
        color=alt.Color("класс:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=None),
        tooltip=["класс", alt.Tooltip("вероятность:Q", format=".2f")]).properties(height=140), width="stretch")
    st.caption(f"Профиль навыков: O*NET {soc} — {occ.loc[soc, 'occupation']}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Куда переводить (близкие должности с низким риском)**")
        for _, t in transition_targets(soc, 3).iterrows():
            st.markdown(f"**{t['query']}** — сходство навыков {t['similarity']:.2f}")
            st.dataframe(skill_gap(soc, t["soc"], 3), hide_index=True, width="stretch")
    with right:
        st.markdown("**Что просят работодатели КР в вакансиях (hh)**")
        hs = hh_skills(pos)
        if len(hs):
            st.dataframe(hs.rename_axis("навык").reset_index(name="вакансий"), hide_index=True, width="stretch")
        else:
            st.info("В собранных вакансиях нет ключевых навыков для этой должности.")

# ---------------- 3. Что если ----------------
with tab3:
    pos3 = st.selectbox("Должность ", POSITIONS, index=POSITIONS.index("Кредитный специалист") if "Кредитный специалист" in POSITIONS else 0)
    soc3 = SOC[pos3]
    st.write("Двигайте уровень навыков (шкала O*NET 1–5) и смотрите, как меняется риск.")
    pick = ["Complex Problem Solving", "Critical Thinking", "Judgment and Decision Making", "Systems Analysis",
            "Programming", "Negotiation", "Persuasion", "Social Perceptiveness", "Service Orientation", "Active Learning"]
    X = occ.loc[[soc3], FEATS].copy()
    cols = st.columns(2)
    for i, s in enumerate(pick):
        X["sk_" + s] = cols[i % 2].slider(SKILL_RU[s], 1.0, 5.0, float(occ.loc[soc3, "sk_" + s]), 0.25, key=f"{pos3}-{s}")
    X = add_composites(X)
    p0, p1 = proba(occ.loc[[soc3]]).iloc[0], proba(X).iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Риск сейчас", p0.idxmax(), f"P(высокий) = {p0['высокий']:.2f}", delta_color="off")
    c2.metric("Риск после обучения", p1.idxmax(), f"{p1['высокий'] - p0['высокий']:+.2f} к P(высокий)", delta_color="inverse")
    st.caption("Модель — логистическая регрессия: изменение одного навыка при прочих равных. Это ориентир для приоритизации "
               "обучения, а не обещание, что должность перестанет автоматизироваться.")

# ---------------- 4. Рынок ----------------
with tab4:
    grp = st.radio("Группа", ["bank", "other", "все"], horizontal=True,
                   format_func=lambda g: {"bank": "банки и офис", "other": "другие отрасли", "все": "все"}[g])
    m = kg if grp == "все" else kg[kg["group"] == grp]
    m = m.dropna(subset=["genai_exposure"]).assign(класс=lambda d: d["pred_class"].map(CLASS_RU))
    pts = alt.Chart(m).mark_circle(opacity=0.85, stroke="white", strokeWidth=1).encode(
        x=alt.X("p_high:Q", title="P(высокий риск автоматизации)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("genai_exposure:Q", title="экспозиция к GenAI"),
        size=alt.Size("vacancies_total:Q", title="вакансий (hh + lalafo)", scale=alt.Scale(range=[40, 900])),
        color=alt.Color("класс:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values()))),
        tooltip=["query", "класс", alt.Tooltip("p_high:Q", format=".2f"), alt.Tooltip("genai_exposure:Q", format=".2f"),
                 "vacancies_kg", "vacancies_lalafo", alt.Tooltip("median_salary_kgs:Q", format=",.0f")])
    labels = pts.mark_text(align="left", dx=8, dy=-6, fontSize=11, color="#52514e").encode(text="query", size=alt.value(11))
    st.altair_chart((pts + labels).properties(height=520), width="stretch")
    st.caption("Справа — риск классической автоматизации, сверху — генеративного ИИ. Правый верхний угол — двойной риск.")
