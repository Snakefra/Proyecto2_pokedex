import streamlit as st
import pandas as pd
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import datetime

# Configuración de la página
st.set_page_config(
    page_title="Pokédex",
    page_icon="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png",
    layout="wide"
)

# CSS para personalizar la página
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://wallpapercave.com/wp/wp2899024.jpg"); 
    background-size: cover;
    background-attachment: fixed;
}
h1 {
    color: #ffcb05; 
    text-shadow: 2px 2px 5px #2a75bb; 
}
button {
    background-color: #2a75bb !important; 
    color: #fff !important; 
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #ffcb05 !important; 
    color: #2a75bb !important; 
}
div.stSelectbox div[data-baseweb="select"] {
    background-color: #ffcb05; 
    color: #2a75bb; 
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
}
div.stTextInput > div > input {
    background-color: #ffcb05; 
    color: #2a75bb; 
    border-radius: 10px;
    font-size: 16px;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Logo de Pokémon
st.image(
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png",
    width=100,
    caption="Pokédex"
)

# Cargar variables de entorno
load_dotenv()

# Cargar los datasets
datasets = {
    "Pokemon Database": pd.read_csv("Pokemon Database.csv"),
    "Alopez Data Mining": pd.read_csv("pokemon_alopez247.csv"),
    "Pokemon Descriptions": pd.read_csv("pokemon_descriptions.csv"),
    "All Pokemon Stats": pd.read_csv("PokemonDB.csv"),
    "Pokemon Games": pd.read_csv("pokemonGames.csv"),
    "Type Matchup Data": pd.read_csv("PokeTypeMatchupData.csv")
}

# Guardar historial en archivo
def guardar_historial(pregunta, respuesta):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}, {pregunta}, {respuesta}\n")

# Cargar historial desde archivo
def cargar_historial():
    try:
        with open("history.txt", "r") as f:
            return f.readlines()
    except FileNotFoundError:
        return []

# Crear un agente general
def crear_agente_general():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [
        Tool(
            name="Python REPL",
            func=PythonREPLTool().run,
            description="Useful for generating Python code and calculations."
        )
    ]
    return initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Interfaz principal
def main():
    st.title("Pokédex")
    st.markdown("Interactúa con diferentes datasets de Pokémon usando agentes avanzados.")

    # Preguntas predefinidas
    preguntas_predefinidas = [
        "¿Cuál es el Pokémon con la mayor estadística total?",
        "¿Qué juegos de Pokémon salieron en 2021?",
        "¿Cuáles son los tipos más efectivos contra Charizard?",
        "¿Cuál es la descripción física de Pikachu?",
        "¿Qué información existe sobre los Pokémon de la generación 1?"
    ]

    # Selección de preguntas predefinidas
    pregunta_seleccionada = st.selectbox("Selecciona una pregunta predefinida:", preguntas_predefinidas)

    if st.button("Ejecutar pregunta predefinida"):
        if pregunta_seleccionada == "¿Cuál es el Pokémon con la mayor estadística total?":
            df = datasets["All Pokemon Stats"]
            respuesta = df[df["Total"] == df["Total"].max()]
            st.write("Respuesta del agente:", respuesta)

        elif pregunta_seleccionada == "¿Qué juegos de Pokémon salieron en 2021?":
            df = datasets["Pokemon Games"]
            respuesta = df[df["gameReleaseYear"] == 2021]
            st.write("Juegos lanzados en 2021:", respuesta)

        elif pregunta_seleccionada == "¿Cuáles son los tipos más efectivos contra Charizard?":
            df = datasets["Type Matchup Data"]
            charizard_data = df[df["Name"] == "Charizard"]
            if not charizard_data.empty:
                respuesta = charizard_data.iloc[0, 2:].sort_values(ascending=False)
                st.write("Tipos efectivos contra Charizard:", respuesta)
            else:
                st.write("No se encontraron datos para Charizard.")

        elif pregunta_seleccionada == "¿Cuál es la descripción física de Pikachu?":
            df = datasets["Pokemon Descriptions"]
            respuesta = df[df["Name"] == "Pikachu"]["Description"].iloc[0]
            st.write("Descripción física de Pikachu:", respuesta)

        elif pregunta_seleccionada == "¿Qué información existe sobre los Pokémon de la generación 1?":
            df = datasets["Alopez Data Mining"]
            respuesta = df[df["Generation"] == 1]
            st.write("Información sobre la Generación 1:", respuesta)

    # Campo de texto para preguntas personalizadas
    st.markdown("### Realiza una pregunta personalizada")
    pregunta_personalizada = st.text_area("Escribe tu pregunta sobre Pokémon:")

    agente_general = crear_agente_general()

    if st.button("Ejecutar pregunta personalizada"):
        if pregunta_personalizada:
            try:
                respuesta_personalizada = agente_general.invoke({"input": pregunta_personalizada})
                st.write("Respuesta del agente:", respuesta_personalizada)
                guardar_historial(pregunta_personalizada, respuesta_personalizada)
            except Exception as e:
                st.error(f"Error procesando la pregunta: {e}")

    # Mostrar historial
    st.markdown("### Historial de Preguntas y Respuestas")
    historial = cargar_historial()
    if historial:
        for linea in historial:
            st.write(linea)
    else:
        st.write("No hay historial disponible.")

if __name__ == "__main__":
    main()
