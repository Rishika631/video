
from transformers import pipeline, T5Tokenizer
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfWriter
import nltk
from nltk.tokenize import sent_tokenize
import base64

# Set Streamlit page configuration
st.set_page_config(page_title="YouTube Video Transcript Summarizer and Q&A")

# Function to extract transcript from YouTube video
def extract_transcript(youtube_video):
    video_id = youtube_video.split("=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    transcript_text = ""
    for segment in transcript:
        transcript_text += segment['text'] + " "
    
    return transcript_text

# Function to summarize transcript
def summarize_transcript(transcript):
    # Tokenize transcript into sentences
    sentences = sent_tokenize(transcript)
    
    # Select a summary length based on the number of sentences
    summary_length = max(int(len(sentences) * 0.3), 1)
    
    # Join sentences back into a single string for summarization
    input_text = " ".join(sentences)
    
    # Use T5 model for summarization
    summarization_model = pipeline("summarization", model="t5-base", tokenizer="t5-base")
    
    # Generate summary
    summary = summarization_model(input_text, max_length=summary_length, min_length=summary_length)
    summary_text = summary[0]['summary_text']
    
    return summary_text

# Function to generate PDF with transcript and summary
def generate_pdf(transcript, summary):
    pdf_path = "transcript_summary.pdf"
    pdf_writer = PdfWriter()
    pdf_writer.add_page()
    pdf_writer.set_font("Arial", size=12)
    
    pdf_writer.cell(0, 10, txt="Transcript:", ln=True)
    pdf_writer.cell(0, 10, txt=transcript, ln=True)
    
    pdf_writer.cell(0, 10, txt="\nSummary:", ln=True)
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
    st.header("YouTube Video Transcript Summarizer and Q&A")
    
    # Get YouTube video URL from user
    youtube_video = st.text_input("Enter the YouTube video URL:")
    
    if youtube_video:
        # Extract transcript from YouTube video
        transcript = extract_transcript(youtube_video)
        
        # Summarize transcript
        summary = summarize_transcript(transcript)
        
        # Generate PDF with transcript and summary
        pdf_path = generate_pdf(transcript, summary)
        
        st.info("Transcript and Summary generated successfully!")
        
        # Display transcript and summary
        st.subheader("Transcript")
        st.text_area("Transcript", transcript, height=300)
        
        st.subheader("Summary")
        st.text(summary)
        
        # Provide
