# Import necessary libraries and modules
from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from youtube_comment_downloader import YoutubeCommentDownloader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re
import os

# Initialize Flask app and config
app = Flask(__name__)
load_dotenv()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load API keys from environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token or ""

# Initialize the language model
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-8b-8192",
    temperature=0.7
)

# Set up HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Set default text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Cache for storing processed video data
cache = {}

# Extract video ID from a YouTube URL
def extract_video_id(url):
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Home page route
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle transcript and comments processing
@app.route("/get_transcript", methods=["POST"])
def handle_transcript_request():
    try:
        # Get video URL and query from form
        video_url = request.form.get("video_url")
        query = request.form.get("query", "").strip()

        if not video_url:
            return jsonify({"error": "No video URL provided"}), 400

        # Extract video ID
        video_id = extract_video_id(video_url)

        # Use cache if available
        if video_id in cache:
            data = cache[video_id]
        else:
            data = {...}
            cache[video_id] = data

        if not video_id:
            return jsonify({"error": "Invalid YouTube URL format"}), 400

        # Fetch transcript from YouTube
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_texts = [entry['text'] for entry in transcript]

        # Fetch comments using downloader
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(video_url)
        comment_texts = [c['text'] for c in comments]

        # Return error if no transcript or comments
        if not transcript and not comment_texts:
            return jsonify({"error": "This video has no transcript or comments available for analysis."}), 400

        # Combine transcript and comment texts
        all_chunks = transcript_texts + comment_texts

        # Join all texts into one string
        all_text = " ".join(all_chunks) if isinstance(all_chunks, list) else all_chunks

        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
        split_chunks = text_splitter.split_text(all_text)
        final_chunks = split_chunks

        # Create vector store from text chunks
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectors = Chroma.from_texts(
            texts=final_chunks,
            embedding=embeddings,
            persist_directory=f"chroma_db/{video_id}"
        )
        vectors.persist()

        # Create the prompt template for LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a focused YouTube assistant. You have access to two sources: the video transcript (what was said) and the top viewer comments (opinions and reactions).\n\n"
            "Your job is to give short, clear, and accurate answers using ONLY the information provided. Do not guess or add anything not in the transcript or comments.\n\n"
            "Guidelines:\n"
            "1. Use the transcript for questions about what was said in the video.\n"
            "2. Use the comments for audience opinions.\n"
            "3. If both matter, combine them briefly.\n"
            "4. Mention timestamps (e.g., 'At 2:45...') for transcript quotes.\n"
            "5. Mention viewers (e.g., 'One comment said...') and likes if helpful.\n"
            "6. If the answer is not in the context, say: 'I couldn't find that in the transcript or comments.'\n"
            "7. Use short bullet points when listing things.\n"
            "8. If the word 'video' is in the question, use only the transcript and ignore comments."
            ),
            ("human", 
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{input}\n\n"
            "Give a clear and concise answer using the rules above."
            )
        ])

        # Create the retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever(search_kwargs={"k": 2})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # If user asked a question, get the answer
        if query:
            result = retrieval_chain.invoke({"input": query})
            return jsonify({"answer": result["answer"]})

        # If no query, confirm video loaded
        return jsonify({
            "answer": "Video loaded successfully! Ask me anything about it.",
            "video_id": video_id
        })

    # Handle and log any errors
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
