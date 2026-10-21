import pandas as pd
import numpy as np
from scipy import stats


def get_categorical_frequencies(df, top_n=None):
    """Retorna a frequência das categorias das variáveis categóricas."""

    categorical_columns = df.select_dtypes(include="object").columns

    frequencies = []

    for column in categorical_columns:
        counts = df[column].value_counts()

        if top_n:
            counts = counts.head(top_n)

        for category, count in counts.items():
            frequencies.append({
                "variable": column,
                "category": category,
                "frequency": count
            })

    return pd.DataFrame(frequencies)


def get_missing_values(df):
    """Retorna a quantidade e porcentagem de valores ausentes."""

    missing = df.isnull().sum()

    missing_pct = (missing / len(df) * 100).round(2)

    missing_table = pd.DataFrame({
        "qtd_ausentes": missing,
        "pct_ausentes": missing_pct
    })

    missing_table = missing_table[
        missing_table["qtd_ausentes"] > 0
    ].sort_values(
        "qtd_ausentes",
        ascending=False
    )

    return missing_table


def get_correlation_matrix(df, variables):
    """Retorna a matriz de correlação de Pearson das variáveis selecionadas."""

    return df[variables].corr(method="pearson")

def detectar_outliers_iqr(df, coluna):
    """Detecta outliers de uma variável numérica usando o método IQR."""

    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)

    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[
        (df[coluna] < limite_inferior) |
        (df[coluna] > limite_superior)
    ]

    return {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "quantidade_outliers": len(outliers),
        "outliers": outliers
    }

def detectar_outliers_zscore(df, coluna, limite=3):
    """Detecta outliers usando o método Z-score."""

    zscores = np.abs(stats.zscore(df[coluna]))

    outliers = df[zscores > limite].copy()

    outliers["zscore"] = zscores[zscores > limite]

    return {
        "limite": limite,
        "quantidade_outliers": len(outliers),
        "outliers": outliers
    }


import pandas as pd

import pandas as pd

def verificar_valores_invalidos(df):
    limites = {
        "Age": (18, 90),
        "Height_cm": (100, 250),
        "Weight_kg": (20, 300),
        "BMI": (5, 100),
        "Waist_Circumference_cm": (30, 250),
        "Blood_Glucose": (0, 1000),
        "HbA1c": (0, 30),
        "Fasting_Blood_Sugar": (0, 1000),
        "Insulin_Level": (0, 10000),
        "Blood_Pressure_Systolic": (0, 400),
        "Blood_Pressure_Diastolic": (0, 300),
        "Total_Cholesterol": (0, 1000),
        "HDL": (0, 500),
        "LDL": (0, 1000),
        "Triglycerides": (0, 5000),
        "Heart_Rate": (0, 300),
        "Exercise_Hours_Per_Week": (0, 168),
        "Daily_Walking_Minutes": (0, 1440),
        "Sleep_Hours": (0, 24),
        "Daily_Water_Intake_L": (0, 100),
        "Diabetes_Risk_Score": (0, 100)
    }

    resultados = []

    for coluna, (minimo, maximo) in limites.items():
        if coluna not in df.columns:
            continue

        valores = df[coluna]

        invalidos = valores.notna() & (
            (valores < minimo) | (valores > maximo)
        )

        resultados.append({
            "Variavel": coluna,
            "Limite_Minimo": minimo,
            "Limite_Maximo": maximo,
            "Quantidade_Invalidos": int(invalidos.sum()),
            "Percentual_Invalidos": round(
                invalidos.sum() / valores.notna().sum() * 100, 2
            ) if valores.notna().sum() > 0 else 0
        })

    return pd.DataFrame(resultados)