import os
import matplotlib.pyplot as plt
import seaborn as sns


def save_plot(filename, output_dir="../reports/figures"):
    """Salva o gráfico atual no diretório de saída."""

    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, filename),
        bbox_inches="tight",
        dpi=300
    )



def plot_missing_values_heatmap(df):
    """Gera um mapa de calor dos valores ausentes."""

    cols_com_nulos = df.columns[df.isnull().any()]

    if len(cols_com_nulos) == 0:
        print("O conjunto de dados não possui valores ausentes.")
        return

    fig, ax = plt.subplots(figsize=(15, 9))

    sns.heatmap(
        df[cols_com_nulos].isnull(),
        ax=ax,
        cbar=True,
        cmap="viridis",
        yticklabels=False,
        xticklabels=cols_com_nulos,
        cbar_kws={
            "label": "Situação do dado",
            "ticks": [0.25, 0.75],
        },
        vmin=0,
        vmax=1,
    )

    # Rótulos mais legíveis no eixo X
    ax.set_xticklabels(
        cols_com_nulos,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )

    # Legenda da barra de cores
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(["Presente (0)", "Ausente (1)"])

    ax.set_title("Distribuição de Valores Ausentes por Variável",
                 fontsize=13, pad=14)
    ax.set_xlabel("Variáveis com valores ausentes", labelpad=10)
    ax.set_ylabel("Registros")

    fig.tight_layout()

    save_plot("missing_values_heatmap.png")

    plt.show()
    plt.close(fig)

def plot_numerical_boxplots(df):
    """Gera boxplots para todas as variáveis numéricas."""

    numerical_columns = df.select_dtypes(include="number").columns
    numerical_columns = numerical_columns.drop("Patient_ID")

    for column in numerical_columns:
        plt.figure(figsize=(8, 5))

        sns.boxplot(x=df[column])

        plt.title(f"Boxplot — {column}")
        plt.xlabel(column)

        plt.show()
        plt.close()

def plot_correlation_matrix(matriz_corr):
    """Gera um heatmap da matriz de correlação."""

    plt.figure(figsize=(15,13))

    sns.heatmap(
        matriz_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0,
        linewidths=0.5
    )

    plt.title("Matriz de Correlação de Pearson")

    plt.tight_layout()

    plt.show()
    plt.close()

def plot_pairplot(df, variables):
    """Gera uma matriz de dispersão entre variáveis numéricas."""

    pair_fig = sns.pairplot(
        df[variables],
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 25}
    )

    pair_fig.fig.suptitle(
        "Matriz de Dispersão entre Variáveis Numéricas",
        y=1.02,
        fontsize=12
    )

    pair_fig.savefig(
        "../reports/figures/scatter_matrix_numericas.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()
    plt.close()



def plot_categorical_distribution(
    df,
    column,
    title,
    filename,
    horizontal=True,
    include_missing=False
):
    # Conta as categorias, tratando ausências separadamente se solicitado
    if include_missing:
        counts = (
            df[column]
            .fillna("Ausente")
            .value_counts()
        )
    else:
        counts = df[column].value_counts()

    # Ordena da maior para a menor frequência
    counts = counts.sort_values(ascending=False)

    plt.figure(figsize=(10, max(5, len(counts) * 0.4)))

    if horizontal:
        ax = sns.barplot(
            x=counts.values,
            y=counts.index,
            color="steelblue"
        )
        ax.set_xlabel("Quantidade de Registros")
        ax.set_ylabel("Categoria")
    else:
        ax = sns.barplot(
            x=counts.index,
            y=counts.values,
            color="steelblue"
        )
        ax.set_xlabel("Categoria")
        ax.set_ylabel("Quantidade de Registros")
        plt.xticks(rotation=25, ha="right")

    ax.set_title(title)

    plt.tight_layout()
    save_plot(filename)
    plt.show()
    plt.close()
