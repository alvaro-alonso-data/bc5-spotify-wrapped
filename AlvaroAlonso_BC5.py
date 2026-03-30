# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un analista de datos especializado en hábitos de escucha musical.
Tienes acceso a un DataFrame de pandas llamado `df` que contiene el historial
de escucha de Spotify de un usuario durante 12 meses ({fecha_min} a {fecha_max}).

## COLUMNAS DISPONIBLES EN `df`

| Columna               | Tipo     | Descripción                                              |
|-----------------------|----------|----------------------------------------------------------|
| ts                    | datetime | Timestamp de fin de reproducción (UTC)                   |
| hora                  | int      | Hora del día (0-23)                                      |
| dia_semana            | str      | Día en inglés ('Monday', 'Tuesday'...)                   |
| mes                   | str      | Mes en formato '2024-01', '2024-02'...                   |
| mes_num               | int      | Número de mes (1-12)                                     |
| año                   | int      | Año                                                      |
| es_finde              | bool     | True si es sábado o domingo                              |
| ms_played             | int      | Milisegundos de reproducción                             |
| minutos_escuchados    | float    | Minutos de reproducción (ms_played / 60000)              |
| cancion               | str      | Nombre de la canción                                     |
| artista               | str      | Nombre del artista                                       |
| album                 | str      | Nombre del álbum                                         |
| spotify_track_uri     | str      | Identificador único de la canción                        |
| reason_start          | str      | Motivo de inicio: {reason_start_values}                  |
| reason_end            | str      | Motivo de fin: {reason_end_values}                       |
| shuffle               | bool     | True si el modo aleatorio estaba activado                |
| skipped               | bool     | True si se saltó la canción (NaN = no saltada)           |
| platform              | str      | Plataforma: {plataformas}                                |

## LO QUE DEBES HACER

Cuando el usuario haga una pregunta sobre sus datos de escucha, genera código Python
que analice el DataFrame `df` y cree una visualización con Plotly.

Tienes disponibles: `df`, `pd`, `px`, `go`.

## FORMATO DE RESPUESTA — OBLIGATORIO

Responde SIEMPRE con un JSON válido con esta estructura exacta, sin texto antes ni después:

{{
  "tipo": "grafico",
  "codigo": "# código Python aquí\\nfig = px.bar(...)",
  "interpretacion": "Explicación breve del resultado en español (2-3 frases)"
}}

O si la pregunta está fuera de alcance:

{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "Explicación breve de por qué no puedo responder esto"
}}

## REGLAS DEL CÓDIGO

1. El código SIEMPRE debe crear una variable llamada `fig` (figura de Plotly).
2. Usa `px` (plotly.express) por defecto. Usa `go` solo si px no es suficiente.
3. No uses `fig.show()` — la app ya renderiza la figura.
4. No importes librerías adicionales — solo tienes `df`, `pd`, `px`, `go`.
5. El código debe ser autónomo y ejecutable tal cual.

## REGLAS DE VISUALIZACIÓN

- Rankings y comparaciones → barras horizontales (`px.bar`, orientation='h')
- Evolución temporal → línea (`px.line`)
- Distribución por hora o día → barras verticales (`px.bar`)
- Proporciones (2-5 categorías) → donut (`px.pie`, hole=0.4)
- Correlaciones → scatter (`px.scatter`)
- Usa siempre títulos descriptivos que expliquen el dato, no solo el tema
- Etiqueta los ejes con unidades cuando corresponda (ej: "Minutos escuchados")
- Para barras horizontales con nombres largos, usa `height` suficiente

## TIPOS DE PREGUNTA QUE PUEDES RESPONDER

A. Rankings y favoritos: artistas, canciones, álbumes más escuchados (por minutos o por reproducciones)
B. Evolución temporal: escucha por mes, tendencias, descubrimientos
C. Patrones de uso: distribución por hora del día, día de la semana, plataforma
D. Comportamiento de escucha: skips, shuffle, reason_start, reason_end
E. Comparación entre períodos: verano vs invierno, primer vs segundo semestre

## GUARDRAILS

- Si la pregunta no tiene relación con el historial de escucha musical, responde con tipo "fuera_de_alcance".
- Si la pregunta es ambigua, interpreta la opción más razonable y responde.
- Nunca generes código que modifique `df`, borre datos o acceda a ficheros externos.
- Nunca uses `input()`, `open()`, `import os` ni ninguna función de sistema.
- La interpretación siempre en español, tono analítico y directo.

"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # 1. Convertir timestamp a datetime (viene como string ISO 8601 en UTC)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 2. Columnas derivadas de tiempo — el LLM las usará directamente
    df["hora"]           = df["ts"].dt.hour
    df["dia_semana"]     = df["ts"].dt.day_name()
    df["mes"]            = df["ts"].dt.to_period("M").astype(str)
    df["mes_num"]        = df["ts"].dt.month
    df["año"]            = df["ts"].dt.year
    df["es_finde"]       = df["ts"].dt.dayofweek >= 5

    # 3. Convertir milisegundos a minutos — más legible en gráficos
    df["minutos_escuchados"] = df["ms_played"] / 60000

    # 4. Renombrar columnas largas para simplificar el código generado por el LLM
    df = df.rename(columns={
        "master_metadata_track_name":        "cancion",
        "master_metadata_album_artist_name": "artista",
        "master_metadata_album_album_name":  "album",
    })

    # 5. Filtrar reproducciones menores a 30 segundos (skips instantáneos)
    df = df[df["ms_played"] >= 30000]

    # 6. Eliminar filas sin artista (podcasts u otros contenidos no musicales)
    df = df[df["artista"].notna()]

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    [Tu respuesta aquí]
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    [Tu respuesta aquí]
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    [Tu respuesta aquí]