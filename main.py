import whisper
import streamlit as st
import openai
import config
import librosa
from pydub import AudioSegment
import time
import joblib

openai.api_key = config.api_key

st.title("Lecture to Anki")


def transcribe(audio_file):
    audio_segment = AudioSegment.from_file(audio_file)
    audio_segment = audio_segment.set_frame_rate(16000)
    audio_segment.export("output.wav", format="wav")
    audio, sr = librosa.load("output.wav", sr=None)

    model = whisper.load_model("base")

    segment_length = 30
    num_segments = len(audio) // (segment_length * sr)

    transcription = ""
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        audio_segment = audio[start:end]
        result = model.transcribe(audio_segment)
        transcription += result["text"]

    return transcription


def generate_cards(transcription):
    prompt = f"Please generate 3 detailed, Basic-type Anki flashcards in the format 'Question|answer$' from the following text: {transcription}"

    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=1000, temperature=0.5,
                                        top_p=1, frequency_penalty=0, presence_penalty=0.6)

    raw_anki_cards = response["choices"][0]["text"]

    anki_cards = raw_anki_cards.split("$")

    for card in anki_cards:
        if card != "":
            card = card.split("|")
            approved_cards.append({"question": card[0], "answer": card[1]})


approved_cards = []
final_approved_cards = []
audio_file = st.file_uploader("Upload a lecture", type=["mp3", "mp4", "wav"])
if audio_file:
    with st.spinner("Uploading and transcribing lecture..."):
        try:
            transcription = joblib.load("transcription.pkl")
        except:
            transcription = transcribe(audio_file)
            joblib.dump(transcription, "transcription.pkl")
        generate_cards(transcription)

    st.success("Lecture transcribed!")

if len(approved_cards) > 0:
    for card in approved_cards:
        question, answer = card["question"], card["answer"]
        accepted = st.checkbox(f"Accept card? {question} {answer}", value=True)
        if accepted:
            final_approved_cards.append({"question": question, "answer": answer})
    approved_cards = []
    st.success("Cards generated!")

    # if st.button("Save Anki-friendly file"):
    #     with open("anki_cards.txt", "w") as f:
    #         for card in final_approved_cards:
    #             f.write(f"{card['question']}\t{card['answer']}\n")
    #     st.success("Anki-friendly file saved!")
