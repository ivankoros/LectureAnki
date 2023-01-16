import whisper
import streamlit as st
import openai
import config
import librosa
from pydub import AudioSegment
import time

openai.api_key = config.api_key

st.title("Lecture to Anki")
with st.spinner("Uploading and transcribing lecture..."):
    audio_file = st.file_uploader("Upload a lecture", type=["mp3", "mp4", "wav"])

approved_cards = []

if audio_file:
    audio_segment = AudioSegment.from_file(audio_file)
    audio_segment = audio_segment.set_frame_rate(16000)
    audio_segment.export("output.wav", format="wav")
    audio, sr = librosa.load("output.wav", sr=None)

    model = whisper.load_model("base")

    segment_length = 30
    num_segments = len(audio) // (segment_length * sr)

    start_time = time.time()
    transcription = ""
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        audio_segment = audio[start:end]
        result = model.transcribe(audio_segment)
        transcription += result["text"]

    end_time = time.time()
    st.spinner("Done!")
    st.write("Transcription time: ", end_time - start_time)

    prompt = f"Please generate 3 detailed, Basic-type Anki flashcards in the format 'Question|answer$' from the following text: {transcription}"

    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=1000, temperature=0.5, top_p=1, frequency_penalty=0, presence_penalty=0.6)

    raw_anki_cards = response["choices"][0]["text"]

    anki_cards = raw_anki_cards.split("$")

    for card in anki_cards:
        if card != "":
            card = card.split("|")
            approved_cards.append({"question": card[0], "answer": card[1]})
    st.write(approved_cards)

# Create a prompt to accept/deny each cards. If accepted, add to Anki-friendly file
final_approved_cards = []
count = 0
for card in approved_cards:
    question = card["question"]
    answer = card["answer"]
    option = st.selectbox('Keep this card?', ('Yes', 'Np'), key=count)
    if option == "Yes":
        final_approved_cards.append(card)
    count += 1
    if option == "No":
        continue

