import streamlit as st
import PyPDF2
import moviepy.editor as mp
from transformers import pipeline
from PyPDF2 import PdfWriter

# Set Streamlit page configuration
st.set_page_config(page_title="Video Transcript and Q&A")

# Function to extract transcript from video
def extract_transcript(video_file):
    video = mp.VideoFileClip(video_file)
    video_audio = video.audio
    temp_audio_file = "temp.wav"
    video_audio.write_audiofile(temp_audio_file)

    transcriber = pipeline("automatic-speech-recognition")
    transcript = transcriber(temp_audio_file)[0]['alternatives'][0]['text']
    
    return transcript

# Function to generate PDF with transcript
def generate_pdf(transcript):
    pdf_path = "transcript.pdf"
    pdf_writer = PdfWriter()
    pdf_writer.add_page()
    pdf_writer.set_font("Arial", size=12)
    pdf_writer.cell(0, 10, txt=transcript, ln=True)
    
    with open(pdf_path, "wb") as f:
        pdf_writer.write(f)
    
    return pdf_path

# Function to perform question-answering on the PDF
def perform_qa(pdf_path, question):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    qa_model = pipeline("question-answering", model=model_name)
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    context = pdf_bytes.decode("utf-8")
    answer = qa_model(question=question, context=context)
    
    return answer["answer"]

# Streamlit app
def main():
    st.header("Video Transcript and Q&A")
    
    # Upload video file
    video_file = st.file_uploader("Upload a video file (.mp4)", type="mp4")
    
    if video_file is not None:
        # Extract transcript from video
        transcript = extract_transcript(video_file)
        
        # Generate PDF with transcript
        pdf_path = generate_pdf(transcript)
        
        st.info("Transcript generated successfully!")
        
        # Display transcript and provide input for question
        st.subheader("Transcript")
        st.text_area("Transcript", transcript, height=300)
        
        question = st.text_input("Ask a question about the transcript")
        
        if question:
            # Perform question-answering on the PDF
            answer = perform_qa(pdf_path, question)
            st.subheader("Question-Answering Result")
            st.write(answer)

# Run the Streamlit app
if __name__ == "__main__":
    main()
