from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "database"

TRAIN_RAW_PATH = DATA_DIR / "train_mod.csv"
TRAIN_COMPLETO_PATH = DATA_DIR / "train_mod_tratado_completo.csv"
TRAIN_ANOVA_PATH = DATA_DIR / "train_mod_tratado_anova.csv"
TEST_RAW_PATH = DATA_DIR / "test.csv"
TEST_COMPLETO_PATH = DATA_DIR / "test_tratado_completo.csv"
TEST_ANOVA_PATH = DATA_DIR / "test_tratado_anova.csv"

TARGET_PRECO = "Preco"
ANO_REF = 2023

COLUNAS_REMOVER_EXPORT = [
    "ID",
    "Faixa_Preco",
    "Classificacao_Veiculo",
    "Modelo",
    "Data_ultima_lavagem",
    "Adesivos_personalizados",
    "Codigo_concessionaria",
]

FEATURES_KM_NUM = [
    "Débitos",
    "Idade",
    "Volume_motor",
    "Cilindros",
    "Numero_proprietarios",
    "Historico_troca_oleo",
]

FEATURES_KM_CAT = [
    "Categoria",
    "Couro",
    "Combustivel",
    "Tipo_cambio",
    "Tração",
    "Portas",
]

FEATURES_KM = FEATURES_KM_NUM + FEATURES_KM_CAT


def normalizar_texto(texto):
    if pd.isna(texto):
        return texto

    texto = str(texto).strip().lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    return texto


def tratar_colunas_basicas(df):
    df = df.copy()

    if "Débitos" in df.columns:
        df["Débitos"] = pd.to_numeric(
            df["Débitos"].replace("-", np.nan),
            errors="coerce",
        ).fillna(0)

    if "Ano" in df.columns:
        df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce")

    if "Combustivel" in df.columns:
        mapeamento_combustivel = {
            "gasolina": "Gasolina",
            "gasol.": "Gasolina",
            "diesel": "Diesel",
            "dies.": "Diesel",
            "hibrido": "Híbrido",
            "gas natural": "Gás Natural",
        }
        combustivel_norm = df["Combustivel"].apply(normalizar_texto)
        df["Combustivel"] = combustivel_norm.map(mapeamento_combustivel).fillna(df["Combustivel"])

    if "Volume_motor" in df.columns:
        df["Volume_motor"] = (
            df["Volume_motor"]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
        )
        df["Volume_motor"] = pd.to_numeric(df["Volume_motor"], errors="coerce")

    if "Km" in df.columns:
        df["Km"] = (
            df["Km"]
            .astype(str)
            .str.replace(" km", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.strip()
        )
        df["Km"] = pd.to_numeric(df["Km"], errors="coerce")

    if {"Ano", "Km"}.issubset(df.columns):
        df["Idade"] = (ANO_REF - df["Ano"]).clip(lower=1)
        df.loc[(df["Km"] >= 4_000_000.0) | (df["Km"] < 10_000), "Km"] = pd.NA
        df["Km_por_ano"] = df["Km"] / df["Idade"]

        km_str = df["Km"].round().astype("Int64").astype("string")
        padrao_km_repetido = km_str.str.fullmatch(r"(\d)\1{4,}", na=False)
        valores_km_sentinela = df["Km"].isin([
            11_111, 22_222, 33_333, 44_444, 55_555, 66_666, 77_777, 88_888, 99_999,
            111_111, 222_222, 333_333, 444_444, 555_555, 666_666, 777_777, 888_888, 999_999,
            1_111_111, 2_222_222, 123_456, 1_234_567, 101_010, 121_212,
        ])

        regra_km_suspeito = (
            (df["Km"] >= 1_000_000) |
            (df["Km_por_ano"] > 70_000) |
            ((df["Idade"] >= 2) & (df["Km_por_ano"] < 1_000)) |
            padrao_km_repetido |
            valores_km_sentinela
        )
        df.loc[regra_km_suspeito, "Km"] = pd.NA
        df["Km_por_ano"] = df["Km"] / df["Idade"]

    if "Cor" in df.columns:
        mapeamento_cor = {
            "preto": "Preto",
            "branco": "Branco",
            "prata": "Prata",
            "cinza": "Cinza",
            "azul": "Azul",
            "azul ceu": "Azul",
            "red": "Vermelho",
            "vermelho": "Vermelho",
            "verde": "Verde",
            "marrom": "Marrom",
            "bege": "Bege",
            "amarelo": "Amarelo",
            "dourado": "Dourado",
            "laranja": "Laranja",
            "roxo": "Roxo",
            "rosa": "Rosa",
        }
        cor_norm = df["Cor"].apply(normalizar_texto)
        df["Cor"] = cor_norm.map(mapeamento_cor).fillna(df["Cor"])

    if "Data_ultima_lavagem" in df.columns:
        df["Data_ultima_lavagem"] = pd.to_datetime(
            df["Data_ultima_lavagem"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        data_ref = pd.Timestamp("2023-01-01")
        df["Dias_desde_ultima_lavagem"] = (data_ref - df["Data_ultima_lavagem"]).dt.days

    for col in [
        "Cilindros",
        "Airbags",
        "Numero_proprietarios",
        "Historico_troca_oleo",
        TARGET_PRECO,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["Rodas", "Codigo_concessionaria", "Classificacao_Veiculo", "Faixa_Preco"], errors="ignore")
    return df


def carregar_base_treino_para_km():
    if TRAIN_COMPLETO_PATH.exists():
        df_train = pd.read_csv(TRAIN_COMPLETO_PATH)
    else:
        df_train = pd.read_csv(TRAIN_RAW_PATH)
        df_train = tratar_colunas_basicas(df_train)

    missing = [col for col in FEATURES_KM + ["Km"] if col not in df_train.columns]
    if missing:
        raise ValueError(f"Colunas ausentes na base de treino para imputar Km: {missing}")

    return df_train


def treinar_imputador_km():
    df_train = carregar_base_treino_para_km()
    df_train_km = df_train[df_train["Km"].notna()].copy()

    preprocessor_km = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), FEATURES_KM_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_KM_CAT),
    ])

    model_km = Pipeline([
        ("prep", preprocessor_km),
        ("model", RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            random_state=42,
            n_jobs=1,
        )),
    ])

    model_km.fit(df_train_km[FEATURES_KM], df_train_km["Km"])
    return model_km


def imputar_km(df):
    df = df.copy()
    missing = [col for col in FEATURES_KM + ["Km"] if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes na base de teste para imputar Km: {missing}")

    km_faltante = df["Km"].isna()
    df["Km_imputado"] = km_faltante.astype(int)

    if km_faltante.any():
        model_km = treinar_imputador_km()
        km_previsto = np.clip(model_km.predict(df.loc[km_faltante, FEATURES_KM]), 0, None)
        df.loc[km_faltante, "Km"] = km_previsto

    df["Km_por_ano"] = df["Km"] / df["Idade"]
    return df


def marcar_preco_suspeito(df):
    df = df.copy()
    if TARGET_PRECO not in df.columns or not TRAIN_ANOVA_PATH.exists():
        return df

    preco_treino = pd.to_numeric(pd.read_csv(TRAIN_ANOVA_PATH)[TARGET_PRECO], errors="coerce")
    limite_superior = preco_treino.max()

    preco = pd.to_numeric(df[TARGET_PRECO], errors="coerce")
    preco_suspeito = preco.notna() & ((preco <= 0) | (preco > limite_superior))

    df["Preco_original"] = preco
    df["Preco_suspeito"] = preco_suspeito.astype(int)
    df.loc[preco_suspeito, TARGET_PRECO] = np.nan

    print(f"Precos suspeitos marcados como ausentes: {int(preco_suspeito.sum())}")
    print(f"Limite superior usado para Preco: {limite_superior:.2f}")

    return df


def alinhar_com_anova_treino(df):
    if not TRAIN_ANOVA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de treino ANOVA nao encontrado: {TRAIN_ANOVA_PATH}. "
            "Rode primeiro a exportacao da analise exploratoria."
        )

    df_train_anova = pd.read_csv(TRAIN_ANOVA_PATH)
    colunas_anova = df_train_anova.columns.tolist()
    feature_columns = [col for col in colunas_anova if col != TARGET_PRECO]

    df_modelagem = df.drop(
        columns=[c for c in COLUNAS_REMOVER_EXPORT if c in df.columns],
        errors="ignore",
    ).copy()

    if TARGET_PRECO in df_modelagem.columns:
        df_modelagem[TARGET_PRECO] = pd.to_numeric(df_modelagem[TARGET_PRECO], errors="coerce")

    df_modelagem = df_modelagem.replace([np.inf, -np.inf], np.nan)

    # Sem drop_first no teste: as colunas sao alinhadas pelas dummies selecionadas no treino.
    # Isso evita perder uma categoria quando ela aparece sozinha no test.csv.
    df_test_enc = pd.get_dummies(df_modelagem, drop_first=False)
    bool_cols = df_test_enc.select_dtypes(include="bool").columns
    df_test_enc[bool_cols] = df_test_enc[bool_cols].astype(int)
    df_test_enc = df_test_enc.select_dtypes(include="number")

    X_test_anova = df_test_enc.reindex(columns=feature_columns, fill_value=0)
    medianas_treino = df_train_anova[feature_columns].median(numeric_only=True)
    X_test_anova = X_test_anova.replace([np.inf, -np.inf], np.nan)
    X_test_anova = X_test_anova.fillna(medianas_treino).fillna(0)

    df_export = X_test_anova.copy()
    if TARGET_PRECO in df_modelagem.columns:
        df_export[TARGET_PRECO] = df_modelagem[TARGET_PRECO].values
    else:
        df_export[TARGET_PRECO] = np.nan

    return df_export[colunas_anova]


def tratar_test():
    df_raw = pd.read_csv(TEST_RAW_PATH)
    linhas_entrada = len(df_raw)

    df_tratado = tratar_colunas_basicas(df_raw)
    df_tratado = imputar_km(df_tratado)
    df_tratado = marcar_preco_suspeito(df_tratado)
    df_anova = alinhar_com_anova_treino(df_tratado)

    if len(df_anova) != linhas_entrada:
        raise RuntimeError("O tratamento alterou a quantidade de linhas do test.csv.")

    df_tratado.to_csv(TEST_COMPLETO_PATH, index=False)
    df_anova.to_csv(TEST_ANOVA_PATH, index=False)

    print(f"Linhas de entrada: {linhas_entrada}")
    print(f"Linhas exportadas: {len(df_anova)}")
    print(f"Colunas exportadas no formato ANOVA: {df_anova.shape[1]}")
    print(f"Km imputados: {int(df_tratado['Km_imputado'].sum())}")
    print(f"Arquivo completo: {TEST_COMPLETO_PATH}")
    print(f"Arquivo para validar o modelo: {TEST_ANOVA_PATH}")

    return df_anova


if __name__ == "__main__":
    tratar_test()
