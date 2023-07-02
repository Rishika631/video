import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfWriter
import base64

# Set Streamlit page configuration
st.set_page_config(page_title="YouTube Video Summarizer and Q&A")

# Function to extract transcript from YouTube video
def extract_transcript(youtube_video):
    video_id = youtube_video.split("=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    transcript_text = ""
    for segment in transcript:
        transcript_text += segment['text'] + " "
    
    return transcript_text

# Function to summarize transcript
def summarize_transcript(transcript):
    # Split transcript into chunks of 1000 characters (for T5 model limitation)
    chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
    
    # Initialize summarization model
    summarization_model = pipeline("summarization")
    
    # Summarize each chunk and combine the summaries
    summary = ""
    for chunk in chunks:
        summarized_chunk = summarization_model(chunk, max_length=100, min_length=50, do_sample=False)[0]["summary_text"]
        summary += summarized_chunk + " "
    
    return summary

# Function to generate PDF with transcript summary
def generate_pdf(summary):
    pdf_path = "transcript_summary.pdf"
    pdf_writer = PdfWriter()
    pdf_writer.add_page()
    pdf_writer.set_font("Arial", size=12)
    
    pdf_writer.cell(0, 10, txt="Summary:", ln=True)
    pdf_writer.cell(0, 10, txt=summary, ln=True)
    
    with open(pdf_path, "wb") as f:
        pdf_writer.write(f)
    
    return pdf_path

# Function to perform question-answering on the PDF
def perform_qa(pdf_path, question):
    qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    context = pdf_bytes.decode("utf-8")
    answer = qa_model(question=question, context=context)
    
    return answer["answer"]

# Streamlit app
def main():
    st.header("YouTube Video Summarizer and Q&A")
    
    # Get YouTube video URL from user
    youtube_video = st.text_input("Enter the YouTube video URL:")
    
    if youtube_video:
        # Extract transcript from YouTube video
        transcript = extract_transcript(youtube_video)
        
        # Summarize transcript
        summary = summarize_transcript(transcript)
        
        # Generate PDF with transcript summary
        pdf_path = generate_pdf(summary)
        
        st.info("Transcript summary generated successfully!")
        
        # Display transcript summary
        st.subheader("Transcript Summary")
        st.text(summary)
        
        # Provide question input
        user_question = st.text_input("Ask a question about the video:")
        
        if user_question:
            # Perform question-answering on the PDF
            answer = perform_qa(pdf_path, user_question)
            
            st.subheader("Question-Answering")
            st.write(answer)

if __name__ == "__main__":
    main()
