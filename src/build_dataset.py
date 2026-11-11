"""Сборка датасета: Frey-Osborne (цель) + O*NET (признаки) + GenAI exposure.

По умолчанию берётся актуальная версия O*NET 31.0 (данные обновлены до 08/2026). Она размечена в
кодах O*NET-SOC 2019, а таблица Frey & Osborne — в SOC 2010, поэтому коды переводятся через официальный
кроссвок O*NET «2010 to 2019». Старая версия 24.0 (2019, коды SOC 2010) оставлена для сравнения:
python src/build_dataset.py 24
"""
import sys
import pandas as pd, numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

ONET_FILES = {"sk": "Skills", "kn": "Knowledge", "ab": "Abilities", "wa": "Work_Activities"}
# ручное сопоставление кодов SOC, которых нет в O*NET 24.0 напрямую
MANUAL = {"29-1111": ["29-1141"], "29-2037": ["53-3011"], "13-1078": ["13-1071", "13-1075"],
          "15-1179": ["15-1122", "15-1134", "15-1143"]}

GENAI_MANUAL = {"15-1179": ["15-1212", "15-1254", "15-1241"], "15-1132": ["15-1252"], "15-1142": ["15-1244"],
                "13-1078": ["13-1071"], "43-5081": ["53-7065"], "29-1111": ["29-1141"], "29-1060": ["29-12"],
                "27-3022": ["27-3023"], "19-3031": ["19-3033", "19-3034"], "53-3041": ["53-3054", "53-3053"],
                "15-1199": ["15-1299"]}

ONET31_FILES = [("sk", "essential_skills"), ("sk", "transferable_skills"), ("kn", "knowledge"),
                 ("ab", "abilities"), ("wa", "work_activities")]
# в версии 31.0 часть элементов переименована — возвращаем прежние названия, чтобы признаки совпадали
RENAME31 = {"Operations Monitoring": "Operation Monitoring", "Administrative": "Clerical",
            "Communicating with People Outside the Organization": "Communicating with Persons Outside Organization",
            "Inspecting Equipment, Structures, or Materials": "Inspecting Equipment, Structures, or Material",
            "Judging the Qualities of Objects, Services, or People": "Judging the Qualities of Things, Services, or People",
            "Monitoring Processes, Materials, or Surroundings": "Monitor Processes, Materials, or Surroundings",
            "Providing Consultation and Advice to Others": "Provide Consultation and Advice to Others",
            "Working with Computers": "Interacting With Computers"}

def load_onet31():
    """O*NET 31.0 → широкая таблица в кодах SOC 2010 (6 знаков) через кроссвок 2010→2019."""
    frames = {}
    for prefix, name in ONET31_FILES:
        d = pd.read_csv(RAW / f"onet31_{name}.csv")
        d = d[d["Scale ID"] == "IM"]
        d["Element Name"] = d["Element Name"].replace(RENAME31)
        w = d.pivot_table(index="O*NET-SOC Code", columns="Element Name", values="Data Value", aggfunc="mean")
        w.columns = [f"{prefix}_{c}" for c in w.columns]
        frames[prefix] = pd.concat([frames[prefix], w], axis=1) if prefix in frames else w
    wide = pd.concat(frames.values(), axis=1)
    cw = pd.read_csv(RAW / "onet_2010_to_2019_crosswalk.csv")
    cw["soc"] = cw["O*NET-SOC 2010 Code"].str[:7]
    m = cw[["soc", "O*NET-SOC 2019 Code"]].drop_duplicates().merge(wide, left_on="O*NET-SOC 2019 Code", right_index=True)
    return m.drop(columns="O*NET-SOC 2019 Code").groupby("soc").mean()

def soc2019_map():
    """SOC 2010 (6 знаков) → список кодов 2019 (6 знаков) по кроссвоку."""
    cw = pd.read_csv(RAW / "onet_2010_to_2019_crosswalk.csv")
    return cw.assign(a=cw["O*NET-SOC 2010 Code"].str[:7], b=cw["O*NET-SOC 2019 Code"].str[:7]) \
             .groupby("a")["b"].apply(lambda x: sorted(set(x))).to_dict()

def load_onet(version="31"):
    if version == "31":
        return load_onet31()
    frames = []
    for prefix, name in ONET_FILES.items():
        d = pd.read_csv(RAW / f"onet24_{name}.txt", sep="\t")
        d = d[d["Scale ID"] == "IM"].copy()          # шкала важности 1-5
        d["soc"] = d["O*NET-SOC Code"].str[:7]       # 11-1011.03 -> 11-1011
        w = d.pivot_table(index="soc", columns="Element Name", values="Data Value", aggfunc="mean")
        w.columns = [f"{prefix}_{c}" for c in w.columns]
        frames.append(w)
    return pd.concat(frames, axis=1)

def onet_row(code, onet):
    if code in onet.index:
        return onet.loc[code]
    if code in MANUAL:
        return onet.loc[[c for c in MANUAL[code] if c in onet.index]].mean()
    # агрегированные коды вида 29-1060 / 25-1000: среднее по всем кодам с тем же префиксом
    stem = code.rstrip("0")
    match = [c for c in onet.index if c.startswith(stem) and c != code]
    if code.endswith("0") and match:
        return onet.loc[match].mean()
    return None

def main(version="31"):
    fo = pd.read_csv(RAW / "frey_osborne.csv").rename(columns={
        "_ - code": "soc", "prob": "automation_prob", "education": "education",
        "occupation": "occupation", "numbEmployed": "employment", "median_ann_wage": "median_wage"})
    fo = fo[["soc", "occupation", "automation_prob", "education", "employment", "median_wage"]]
    onet = load_onet(version)
    rows = {c: onet_row(c, onet) for c in fo["soc"]}
    source = {c: f"O*NET {version}.0" for c, r in rows.items() if r is not None}
    if version == "31":
        # несколько профессий в новой таксономии ещё не оценены (например, 13-2051 Financial Analysts):
        # для них берём профиль из O*NET 24.0 и помечаем это в столбце onet_version
        onet24 = load_onet("24")
        for c, r in rows.items():
            if r is None:
                r24 = onet_row(c, onet24)
                if r24 is not None:
                    rows[c] = r24.reindex(onet.columns)
                    source[c] = "O*NET 24.0"
    feats = pd.DataFrame({c: r for c, r in rows.items() if r is not None}).T
    df = fo.merge(feats, left_on="soc", right_index=True, how="left")
    df["onet_version"] = df["soc"].map(source)

    # GenAI exposure (Eloundou et al., 2023): beta = доля задач, ускоряемых LLM (+0.5 * с инструментами)
    g = pd.read_csv(RAW / "gpts_are_gpts_occ_level.csv")
    g["soc"] = g["O*NET-SOC Code"].str[:7]
    g = g.groupby("soc")[["human_rating_beta", "dv_rating_beta"]].mean()
    g.columns = ["genai_exposure_human", "genai_exposure_gpt4"]
    # GenAI-данные размечены в SOC 2019: переводим коды через официальный кроссвок,
    # а для агрегированных кодов FO (например, 29-1060) — через ручной список
    xw = soc2019_map()
    for old_code in fo["soc"]:
        if old_code not in g.index and old_code in xw:
            hits = [c for c in xw[old_code] if c in g.index]
            if hits:
                g.loc[old_code] = g.loc[hits].mean()
    for old_code, new_codes in GENAI_MANUAL.items():
        if old_code not in g.index:
            hits = [c for c in g.index if any(c.startswith(n) for n in new_codes)]
            if hits:
                g.loc[old_code] = g.loc[hits].mean()
    df = df.merge(g, left_on="soc", right_index=True, how="left")

    # Anthropic Economic Index (2025–2026): реальное использование ИИ и роботов по профессиям.
    #   observed_exposure     — доля задач профессии, реально выполняемых с Claude (март 2026; автоматизация = 1, помощь = 0,5)
    #   robot_exposure_index  — насколько физические задачи профессии уже могут делать роботы, 0–3 (сентябрь 2026)
    #   aei_automation_share  — доля «автоматизации» среди использования ИИ (vs «помощь человеку»), март 2025
    aei = pd.read_csv(RAW / "anthropic_aei_occupations.csv", dtype={"soc2019": str}).set_index("soc2019")
    cols = ["observed_exposure", "robot_exposure_index", "aei_automation_share", "aei_usage_pct"]
    rows = {}
    for code in fo["soc"]:
        # коды в файле — частично SOC 2010, частично 2019: для каждого показателя берём прямое совпадение,
        # иначе коды 2019 по кроссвоку, иначе ручной список
        vals = {}
        for col in cols:
            ok = aei[col].dropna().index
            cand = [code] if code in ok else [c for c in xw.get(code, []) if c in ok]
            if not cand and code in GENAI_MANUAL:
                cand = [c for c in ok if any(c.startswith(n) for n in GENAI_MANUAL[code])]
            if cand:
                vals[col] = aei.loc[cand, col].mean()
        if vals:
            rows[code] = pd.Series(vals)
    df = df.merge(pd.DataFrame(rows).T.add_prefix(""), left_on="soc", right_index=True, how="left")

    # целевая переменная: пороги из статьи Frey & Osborne
    df["risk_class"] = pd.cut(df["automation_prob"], [0, 0.3, 0.7, 1.0],
                              labels=["low", "medium", "high"], include_lowest=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "occupations_dataset.csv", index=False)
    print(df.shape, "без признаков O*NET:", df["sk_Active Learning"].isna().sum())

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "31")
