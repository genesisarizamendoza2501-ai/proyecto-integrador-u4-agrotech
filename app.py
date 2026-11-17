import os
import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y CARGA DE CSS
# ==========================================
st.set_page_config(
    page_title="AgroTech Analytics | Optimización de Riego",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def cargar_estilos_css(nombre_archivo="style.css"):
    ruta_css = os.path.join(os.path.dirname(__file__), nombre_archivo)
    if os.path.exists(ruta_css):
        with open(ruta_css, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ No se encontró el archivo '{nombre_archivo}' en la carpeta del proyecto.")

cargar_estilos_css("style.css")

# Configuración global de gráficos para armonizar con el diseño oscuro
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#151E32",
    "figure.facecolor": "#0B0F19",
    "grid.color": "#1E293B",
    "text.color": "#F8FAFC",
    "axes.labelcolor": "#CBD5E1",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "axes.edgecolor": "#1E293B"
})

# ==========================================
# HITO 1: INGESTA DETERMINISTA EN DUCKDB
# ==========================================
@st.cache_data
def cargar_telemetria_duckdb():
    con = duckdb.connect(database=':memory:')
    
    # Semilla determinista (0.42 -> 42 en NumPy) para reproducibilidad exacta
    np.random.seed(42)
    N = 100
    
    parcela_id = np.arange(1, N + 1)
    volumen_riego = np.round(np.random.uniform(100.0, 300.0, N), 2)
    fertilizante_kg = np.round(np.random.uniform(50.0, 150.0, N), 2)
    
    prob_tratamiento = np.random.uniform(0.0, 1.0, N)
    tipo_tratamiento = np.where(prob_tratamiento < 0.50, 'Tratamiento_A', 'Tratamiento_B')
    
    # Ecuación físico-matemática de simulación
    beta_0 = 5.0
    beta_1 = 0.035
    beta_2 = 0.020
    efecto_b = np.where(tipo_tratamiento == 'Tratamiento_B', 1.8, 0.0)
    ruido = np.random.uniform(-0.8, 0.8, N)
    
    rendimiento_ton = np.round(
        beta_0 + (beta_1 * volumen_riego) + (beta_2 * fertilizante_kg) + efecto_b + ruido, 2
    )
    
    df_simulado = pd.DataFrame({
        'parcela_id': parcela_id,
        'volumen_riego': volumen_riego,
        'fertilizante_kg': fertilizante_kg,
        'tipo_tratamiento': tipo_tratamiento,
        'rendimiento_ton': rendimiento_ton
    })
    
    con.register('telemetria_invernadero', df_simulado)
    
    # Extracción SQL limpia (Fase 1)
    df_sql = con.execute("""
        SELECT 
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            tipo_tratamiento,
            rendimiento_ton
        FROM telemetria_invernadero
        ORDER BY parcela_id ASC
    """).fetchdf()
    
    con.close()
    return df_sql

df = cargar_telemetria_duckdb()

# ==========================================
# BARRA LATERAL: CONTROLES EJECUTIVOS
# ==========================================
with st.sidebar:
    st.markdown("## 🌿 AgroTech Intelligence")
    st.caption("Sistema de Soporte a Decisiones Agronómicas")
    st.markdown("---")
    
    st.subheader("🎛️ Filtros de Telemetría")
    tratamientos = df['tipo_tratamiento'].unique().tolist()
    seleccion_tratamiento = st.multiselect(
        "Tratamiento Bio-estimulante:",
        options=tratamientos,
        default=tratamientos
    )
    
    rango_riego = st.slider(
        "Volumen de Riego (m³/ha):",
        min_value=float(df['volumen_riego'].min()),
        max_value=float(df['volumen_riego'].max()),
        value=(float(df['volumen_riego'].min()), float(df['volumen_riego'].max()))
    )
    
    rango_fert = st.slider(
        "Fertilizante Aplicado (kg/ha):",
        min_value=float(df['fertilizante_kg'].min()),
        max_value=float(df['fertilizante_kg'].max()),
        value=(float(df['fertilizante_kg'].min()), float(df['fertilizante_kg'].max()))
    )
    
    st.markdown("---")
    st.info("📌 **Motor Analítico:** DuckDB + Statsmodels OLS + SciPy Inferential")

# Validación de filtros activos
if not seleccion_tratamiento:
    st.warning("⚠️ Selecciona al menos un tipo de tratamiento en el panel lateral para visualizar los datos.")
    st.stop()

df_filtrado = df[
    (df['tipo_tratamiento'].isin(seleccion_tratamiento)) &
    (df['volumen_riego'].between(rango_riego[0], rango_riego[1])) &
    (df['fertilizante_kg'].between(rango_fert[0], rango_fert[1]))
]

# ==========================================
# HITO 2 Y 3: MODELADO OLS E INFERENCIA
# ==========================================
X = sm.add_constant(df[['volumen_riego', 'fertilizante_kg']])
y = df['rendimiento_ton']
modelo_ols = sm.OLS(y, X).fit()

r2_ajustado = modelo_ols.rsquared_adj
rmse = np.sqrt(modelo_ols.mse_resid)

tabla_ols = pd.DataFrame({
    "Variable": ["Intercepto (β₀)", "Volumen de Riego (β₁)", "Fertilizante (β₂)"],
    "Coeficiente Estimado": modelo_ols.params.values,
    "Error Estándar": modelo_ols.bse.values,
    "Estadístico t": modelo_ols.tvalues.values,
    "P-Valor": modelo_ols.pvalues.values,
    "Significancia (α=0.05)": ["Significativo ✅" if p < 0.05 else "No Significativo ❌" for p in modelo_ols.pvalues.values]
})

grupo_a = df[df['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_b = df[df['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']

t_stat, p_valor_bilateral = stats.ttest_ind(grupo_b, grupo_a, equal_var=False)
p_valor_unilateral = p_valor_bilateral / 2 if t_stat > 0 else 1 - (p_valor_bilateral / 2)

diff_medias = grupo_b.mean() - grupo_a.mean()
se_diff = np.sqrt(grupo_b.var(ddof=1)/len(grupo_b) + grupo_a.var(ddof=1)/len(grupo_a))
df_welch = (grupo_b.var(ddof=1)/len(grupo_b) + grupo_a.var(ddof=1)/len(grupo_a))**2 / (
    (grupo_b.var(ddof=1)/len(grupo_b))**2/(len(grupo_b)-1) + (grupo_a.var(ddof=1)/len(grupo_a))**2/(len(grupo_a)-1)
)
ic_95 = stats.t.interval(0.95, df=df_welch, loc=diff_medias, scale=se_diff)

# ==========================================
# HITO 4: ENCABEZADO Y TARJETAS KPI EJECUTIVAS
# ==========================================
st.title("🌱 Optimización Inferencial del Rendimiento Agrícola")
st.markdown("Monitoreo de telemetría de invernadero, modelado predictivo **OLS** y validación inferencial de bio-estimulantes.")

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.metric(
        label="Rendimiento Promedio (Filtrado)",
        value=f"{df_filtrado['rendimiento_ton'].mean():.2f} ton/ha",
        delta=f"{len(df_filtrado)} parcelas activas"
    )
with col_kpi2:
    st.metric(
        label="R² Ajustado (Modelo OLS)",
        value=f"{r2_ajustado * 100:.2f}%",
        delta=f"RMSE: {rmse:.3f} ton/ha",
        delta_color="off"
    )
with col_kpi3:
    st.metric(
        label="Incremento Tratamiento B",
        value=f"+{diff_medias:.2f} ton/ha",
        delta=f"IC 95%: [{ic_95[0]:.2f}, {ic_95[1]:.2f}]"
    )
with col_kpi4:
    st.metric(
        label="P-Valor (Prueba t unilateral)",
        value=f"{p_valor_unilateral:.5f}",
        delta="H₁ Aceptada (p < 0.05)" if p_valor_unilateral < 0.05 else "H₀ No Rechazada",
        delta_color="normal" if p_valor_unilateral < 0.05 else "inverse"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# NAVEGACIÓN POR PESTAÑAS (ORDEN SECUENCIAL)
# ==========================================
tab_datos, tab_resultados, tab_graficas = st.tabs([
    "🗄️ 1. Base de Datos Telemetría (DuckDB)",
    "📐 2. Resultados Matemáticos e Inferencia",
    "📊 3. Dashboard Visual (4 Gráficas)"
])

# ------------------------------------------
# PESTAÑA 1: BASE DE DATOS TELEMETRÍA (DUCKDB)
# ------------------------------------------
with tab_datos:
    st.markdown("### 🗄️ Dataset de Telemetría Agroindustrial ($N = 100$ Parcelas)")
    st.markdown("Datos extraídos mediante consulta SQL desde **DuckDB** tras la simulación determinista.")
    
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Dataset Filtrado (CSV)",
        data=csv,
        file_name="telemetria_invernadero_u4.csv",
        mime="text/csv"
    )
    
    st.dataframe(
        df_filtrado,
        use_container_width=True,
        hide_index=True,
        column_config={
            "parcela_id": st.column_config.NumberColumn("ID Parcela", format="%d"),
            "volumen_riego": st.column_config.ProgressColumn(
                "Volumen Riego (m³/ha)",
                format="%.2f m³",
                min_value=100.0,
                max_value=300.0
            ),
            "fertilizante_kg": st.column_config.NumberColumn("Fertilizante (kg/ha)", format="%.2f kg"),
            "tipo_tratamiento": st.column_config.TextColumn("Tratamiento"),
            "rendimiento_ton": st.column_config.NumberColumn("Rendimiento (ton/ha)", format="%.2f ton")
        }
    )

# ------------------------------------------
# PESTAÑA 2: RESULTADOS MATEMÁTICOS E INFERENCIA
# ------------------------------------------
with tab_resultados:
    col_m1, col_m2 = st.columns([1.2, 1])
    
    with col_m1:
        st.markdown("### 📐Modelo de Regresión Lineal Múltiple (OLS)")
        st.latex(r"Rendimiento = \beta_0 + \beta_1(\text{Riego}) + \beta_2(\text{Fertilizante}) + \epsilon")
        st.dataframe(
            tabla_ols.style.format({
                "Coeficiente Estimado": "{:.4f}",
                "Error Estándar": "{:.4f}",
                "Estadístico t": "{:.3f}",
                "P-Valor": "{:.5f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"**Métricas Globales del Modelo:** $R^2$ = {modelo_ols.rsquared:.4f} | $R^2$ Ajustado = {r2_ajustado:.4f} | Error Cuadrático Medio (RMSE) = {rmse:.4f} ton/ha")

    with col_m2:
        st.markdown("### 🧪Prueba de Hipótesis (t-Student)")
        st.markdown("""
        * **Hipótesis Nula ($H_0$):** $\mu_B - \mu_A \le 0$ *(El bio-estimulante no incrementa el rendimiento)*
        * **Hipótesis Alternativa ($H_1$):** $\mu_B - \mu_A > 0$ *(El Tratamiento B produce mayor rendimiento)*
        """)
        
        df_prueba_t = pd.DataFrame({
            "Métrica Estadística": [
                "Media Tratamiento A (Estándar)",
                "Media Tratamiento B (Bio-estimulante)",
                "Diferencia de Medias (μ_B - μ_A)",
                "Intervalo de Confianza 95%",
                "Estadístico t (Welch)",
                "P-Valor (Cola derecha, α = 0.05)"
            ],
            "Valor Calculado": [
                f"{grupo_a.mean():.3f} ton/ha (n={len(grupo_a)})",
                f"{grupo_b.mean():.3f} ton/ha (n={len(grupo_b)})",
                f"+{diff_medias:.3f} ton/ha",
                f"[{ic_95[0]:.3f}, {ic_95[1]:.3f}] ton/ha",
                f"{t_stat:.4f}",
                f"{p_valor_unilateral:.5f}"
            ]
        })
        st.dataframe(df_prueba_t, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🏆 Veredicto Ejecutivo de Ingeniería")
    if p_valor_unilateral < 0.05:
        st.success(f"""
        **DECISIÓN APROBADA (Escalamiento Recomendado):** Con un estadístico $t = {t_stat:.3f}$ y un $p\\text{{-valor}} = {p_valor_unilateral:.5f} < 0.05$, se **rechaza la Hipótesis Nula ($H_0$)**. 
        El nuevo **Tratamiento_B** genera un incremento promedio comprobado de **+{diff_medias:.2f} ton/ha** (con un 95% de confianza de que el beneficio real está entre **{ic_95[0]:.2f}** y **{ic_95[1]:.2f} ton/ha**). 
        Se recomienda autorizar la inversión para su implementación en todo el invernadero.
        """)
    else:
        st.warning("No se encontró evidencia estadísticamente significativa para rechazar $H_0$.")

# ------------------------------------------
# PESTAÑA 3: DASHBOARD VISUAL (2x2)
# ------------------------------------------
with tab_graficas:
    r1_c1, r1_c2 = st.columns(2)
    
    # Gráfica 1: Heatmap de Correlación de Pearson
    with r1_c1:
        st.markdown("#### 1. Matriz de Correlación de Pearson ($r$)")
        fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
        corr_matrix = df_filtrado[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].corr()
        corr_matrix.columns = ['Riego (m³/ha)', 'Fertilizante (kg/ha)', 'Rendimiento (ton/ha)']
        corr_matrix.index = ['Riego (m³/ha)', 'Fertilizante (kg/ha)', 'Rendimiento (ton/ha)']
        sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".3f", linewidths=1, linecolor="#0B0F19", ax=ax1, vmin=0, vmax=1)
        st.pyplot(fig1)
        
    # Gráfica 2: Dispersión con Línea de Regresión OLS
    with r1_c2:
        st.markdown("#### 2. Dispersión y Ajuste OLS (Rendimiento vs Riego)")
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
        sns.regplot(
            data=df_filtrado, x='volumen_riego', y='rendimiento_ton', ax=ax2,
            scatter_kws={'alpha': 0.8, 'color': '#10B981', 's': 45},
            line_kws={'color': '#F97316', 'linewidth': 2.5, 'label': 'Recta OLS'}
        )
        ax2.set_xlabel("Volumen de Riego (m³/ha)")
        ax2.set_ylabel("Rendimiento (ton/ha)")
        ax2.legend()
        st.pyplot(fig2)

    st.markdown("---")
    r2_c1, r2_c2 = st.columns(2)

    # Gráfica 3: Diagrama de Cajas Comparativo (Boxplot)
    with r2_c1:
        st.markdown("#### 3. Comparación de Tratamientos (Boxplot + Medias)")
        fig3, ax3 = plt.subplots(figsize=(6.5, 4.2))
        sns.boxplot(
            data=df, x='tipo_tratamiento', y='rendimiento_ton', ax=ax3,
            palette={"Tratamiento_A": "#38BDF8", "Tratamiento_B": "#10B981"},
            width=0.4, boxprops=dict(alpha=0.85)
        )
        sns.stripplot(
            data=df, x='tipo_tratamiento', y='rendimiento_ton', ax=ax3,
            color='#F8FAFC', alpha=0.65, jitter=0.15, size=5
        )
        ax3.set_xlabel("Grupo Experimental")
        ax3.set_ylabel("Rendimiento (ton/ha)")
        st.pyplot(fig3)

    # Gráfica 4: Histograma y Densidad de Residuos
    with r2_c2:
        st.markdown("#### 4. Diagnóstico de Normalidad de Residuos")
        fig4, ax4 = plt.subplots(figsize=(6.5, 4.2))
        residuos = modelo_ols.resid
        sns.histplot(residuos, kde=True, color='#A855F7', ax=ax4, bins=15, edgecolor="#0B0F19")
        ax4.axvline(0, color='#F97316', linestyle='--', linewidth=1.5, label='Media = 0')
        ax4.set_xlabel("Residuo (Y_real - Y_predicho)")
        ax4.set_ylabel("Frecuencia")
        ax4.legend()
        st.pyplot(fig4)