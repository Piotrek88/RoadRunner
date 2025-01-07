import streamlit as st
import pandas as pd
import joblib 
import json
from dotenv import dotenv_values, load_dotenv
from openai import OpenAI
import numpy as np
import os
import instructor
from pydantic import BaseModel, Field, ValidationError
from IPython.display import display, Markdown
from itables import init_notebook_mode
from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse.decorators import observe
from datetime import datetime
import time
import pycaret

load_dotenv()
init_notebook_mode(all_interactive=True)
model_runner = joblib.load('marathon_pipeline_regression_model.pkl')

# Kod zabezpieczający klucz API
if "OPENAI_API_KEY" not in st.session_state:
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    else:
        st.info("Podaj klucz API aby korzystać z Marathon Road Runner")
        # Czeka, aż użytkownik wprowadzi klucz i zatwierdzi go przyciskiem
        api_input = st.text_input("Klucz API", type="password")
        if api_input:
            # Ustawienie klucza w sesji jako sposób na przeładowanie
            st.session_state["OPENAI_API_KEY"] = api_input
            st.session_state["key_submitted"] = True  # Dodatkowy stan, aby wywołać rerun

if not st.session_state.get("OPENAI_API_KEY"):
    st.stop()

# Ponowne załadowanie następuje, kiedy ten blok się wykona
if st.session_state.get("key_submitted"):
    # Usuń dodatkowy stan, co spowoduje załadowanie klienta z nowym kluczem
    del st.session_state["key_submitted"]
    st.experimental_rerun()  # Użyj tego, jeśli jest dostępne


openai_client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
instructor_openai_client = instructor.from_openai(openai_client)
llm_client = LangfuseOpenAI(api_key=st.session_state["OPENAI_API_KEY"])


@observe()
def get_info_langfuse_observed(user_input, model="gpt-4o"):
    prompt = """
    Wyciagnij wskazane przeze mnie dane z tekstu w odpowiednich formatach.
    Zwróc szczególna uwage na określenie Płci: jeżeli nie bedzie napisane: jestem kobieta/meżczyzna, wartość Płeć wywnioskuj z np. podanego imienia np. Grzegorz = Mężczyzna, Natalia = Kobieta lub zwrotu, urodziłam się = Kobieta, urodziłem się = Mężczyzna.
    Zwróc "5 km Czas" odczytująć dane 15:30 , 15;30, 15-30 jako "15:30" lub pisane słownie piętnascie minut i trzydziesci sekund oraz gdy ktoś poda dane np. na wiekszym dystansie czyli na 15km 48min to wyciagnij z tych danych średnia na 5km poprzez 48 / 3 = 16min.
    Zwróć wartość jako obiekt JSON z następującymi kluczami:
    "5 km Czas" - as a string,
    "Rocznik" -  ma byc interpretowany jako liczba całkowita integer,
    "Płeć" - as string.
    Przykład finalny JSON:
    {
    "5 km Czas": "14:28",
    "Rocznik": "1997"
    "Płeć": "Kobieta"
    }
    """   
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role":"user",
            "content": user_input,
        },
    ]
    chat_completion = llm_client.chat.completions.create(
        response_format={"type": "json_object"},
        messages=messages,
        model=model,
        name="dane_uzytkownika",
    )

    resp = chat_completion.choices[0].message.content
    try:
        output = json.loads(resp)
    except json.JSONDecodeError:
        output = {"Uwaga": "Nie udało się przeanalizować JSON. Odpowiedź: " + resp}
        return output

    required_fields = ["5 km Czas", "Rocznik", "Płeć"]
    missing_or_invalid_fields = [
        field for field in required_fields 
        if field not in output or not output[field] or (field == "Płeć" and output[field] == "Nieokreślono")
        ]

    if missing_or_invalid_fields:
        raise ValueError(f"Brakujące informacje: {', '.join(missing_or_invalid_fields)}")
    
    def convert_time_to_seconds(time_str):
        if isinstance(time_str, int):
            raise ValueError("Błąd formatu czasu.")
        time_parts = datetime.strptime(time_str, '%M:%S')
        total_seconds = time_parts.minute * 60 + time_parts.second
        return total_seconds
    
    def convert_time_to_tempo(total_time_str, distans_km):
        """Oblicza tempo biegu w minutach/kilometr i zwraca jako float."""
        total_seconds = convert_time_to_seconds(total_time_str)
        tempo_seconds_per_km = total_seconds / distans_km       
        tempo_minutes_per_km = tempo_seconds_per_km / 60        
        return tempo_minutes_per_km

    
    gender_mapping = {"Mężczyzna": 0, "Kobieta": 1}
    try:
        gender_value = gender_mapping[output["Płeć"]]
    except KeyError:
        raise ValueError(f"Nieokreślona wartość płci: {output['Płeć']}")

    
    ai_model_input = {
        "5 km Tempo": convert_time_to_tempo(output["5 km Czas"], 5),
        "Rocznik": output["Rocznik"],
        "Płeć": gender_value
    }

    
    return output, ai_model_input


def convert_seconds_to_hhmmss(total_seconds):
    total_seconds = int(round(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    return formatted_time


###MAIN###

# Kod zabezpieczający klucz API
if "OPENAI_API_KEY" not in st.session_state:
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    else:
        st.info("Podaj klucz API aby korzystać z Marathon Road Runner")
        st.session_state["OPENAI_API_KEY"] = st.text_input("Klucz API", type="password")
        if st.session_state["OPENAI_API_KEY"]:
            st.experimental_rerun()
if not st.session_state.get("OPENAI_API_KEY"):
    st.stop()


def main():
    st.title("Half Marathon Road Runner")
    
    user_input = st.text_area("Przedstaw się, napisz w którym roku się urodziłeś/aś oraz w jakim czasie przebiegniesz dystans 5km?", max_chars=120)

    if st.button("Wprowadź dane"):
        with st.spinner('Przetwarzanie danych...'):
            
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.05)  
                progress_bar.progress(percent_complete + 1)
        
        try:
            langfuse_output, ai_model_input=get_info_langfuse_observed(user_input)
            st.write("Dane poprawne.")
        except ValueError as e:
            st.error(f"Błąd: {e}. Proszę uzupełnić brakujące informacje.")
        else:
            input_df = pd.DataFrame([ai_model_input])   
            predicted_time = model_runner.predict(input_df)
            if isinstance(predicted_time, np.ndarray):
                tempo_seconds = predicted_time[0]  
                if tempo_seconds < 0:
                    st.error("Błąd: przewidywany czas ukończenia maratonu jest ujemny, co jest nieprawidłowe.")
                    return

            else:
                raise ValueError("Bład: Oczekiwano numpy.ndarray z model_runner.predict. Przepraszamy.")
            formatted_time = convert_seconds_to_hhmmss(tempo_seconds)
            st.success(f"Przewidywany czas ukończenia maratonu: {formatted_time}")
           


if __name__ == '__main__':
    main()





