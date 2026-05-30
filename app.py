import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# ---------------------------------------------------
# Configuración de página
# ---------------------------------------------------
st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="🌿",
    layout="wide"
)

# ---------------------------------------------------
# Carga de datos
# ---------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)

    df = iris.frame.copy()
    df["species"] = df["target"].map(
        dict(enumerate(iris.target_names))
    )

    return df

df = load_data()

# ---------------------------------------------------
# Título
# ---------------------------------------------------
st.title("🌿 Iris Dataset Dashboard")
st.markdown(
    "Análisis interactivo del dataset Iris utilizando Streamlit y visualizaciones con paleta Viridis."
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header("Filtros")

species_selected = st.sidebar.multiselect(
    "Seleccionar especie",
    options=sorted(df["species"].unique()),
    default=sorted(df["species"].unique())
)

filtered_df = df[df["species"].isin(species_selected)]

# ---------------------------------------------------
# Métricas
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Observaciones", len(filtered_df))

with col2:
    st.metric(
        "Sepal Length Prom.",
        f"{filtered_df['sepal length (cm)'].mean():.2f}"
    )

with col3:
    st.metric(
        "Petal Length Prom.",
        f"{filtered_df['petal length (cm)'].mean():.2f}"
    )

with col4:
    st.metric(
        "Especies",
        filtered_df["species"].nunique()
    )

st.divider()

# ---------------------------------------------------
# Gráficos principales
# ---------------------------------------------------
left, right = st.columns(2)

with left:

    fig_scatter = px.scatter(
        filtered_df,
        x="sepal length (cm)",
        y="petal length (cm)",
        color="species",
        color_discrete_sequence=px.colors.sequential.Viridis,
        size="petal width (cm)",
        title="Relación entre Sepal Length y Petal Length"
    )

    fig_scatter.update_layout(
        height=500,
        legend_title="Species"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

with right:

    fig_box = px.box(
        filtered_df,
        x="species",
        y="sepal width (cm)",
        color="species",
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Distribución de Sepal Width por Especie"
    )

    fig_box.update_layout(
        height=500,
        showlegend=False
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

# ---------------------------------------------------
# Histograma
# ---------------------------------------------------
fig_hist = px.histogram(
    filtered_df,
    x="petal length (cm)",
    color="species",
    nbins=25,
    color_discrete_sequence=px.colors.sequential.Viridis,
    title="Distribución de Petal Length"
)

fig_hist.update_layout(height=450)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# ---------------------------------------------------
# Matriz de correlación
# ---------------------------------------------------
numeric_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

corr = filtered_df[numeric_cols].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Viridis",
    title="Matriz de Correlación"
)

fig_corr.update_layout(height=550)

st.plotly_chart(
    fig_corr,
    use_container_width=True
)

# ---------------------------------------------------
# Tabla de datos
# ---------------------------------------------------
st.subheader("Datos")

st.dataframe(
    filtered_df,
    use_container_width=True
)
