import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# 1. Φόρτωση API Key 
load_dotenv(override=True)
client = OpenAI()
MODEL = "gpt-4o-mini"

# 2. System Prompt (Ο ρόλος και οι οδηγίες του Chatbot)
SYSTEM_MESSAGE = """
You are an expert Bioinformatics & Genomics AI Assistant.
You help researchers and students understand sequence analysis, variant interpretation, and computational biology tools.
Provide concise, clear answers in Markdown format.
"""


# 3. Callback Function για το Gradio ChatInterface
def chat_callback(message: str, history: list[dict]):
    """Παίρνει το νέο μήνυμα του χρήστη (message) και το ιστορικό της συζήτησης (history),

    ανακατασκευάζει τη λίστα των μηνυμάτων και κάνει stream την απάντηση πίσω
    στο UI.
    """
    # Καθαρισμός του ιστορικού στη μορφή που απαιτεί το OpenAI API
    formatted_history = [
        {"role": item["role"], "content": item["content"]} for item in history
    ]

    # Σύνθεση του πλήρους payload: System + Ιστορικό + Νέο Μήνυμα
    messages = (
        [{"role": "system", "content": SYSTEM_MESSAGE}]
        + formatted_history
        + [{"role": "user", "content": message}]
    )

    # Κλήση του μοντέλου με streaming
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, stream=True
    )

    response_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        response_text += delta
        yield response_text  # Ενημερώνει δυναμικά το chatbox λέξη-λέξη


# 4. Στήσιμο του Gradio Chat UI
demo = gr.ChatInterface(
    fn=chat_callback,
    type="messages",  # Χρήση του σύγχρονου format λεξικών [{'role': ..., 'content': ...}]
    title="🧬 Bioinformatics AI Assistant",
    description="Ask questions about Next-Generation Sequencing (NGS), variant effects, or bioinformatics pipelines.",
    examples=[
        "What is the difference between a missense and a nonsense mutation?",
        "How does the Phred quality score work in FASTQ files?",
        "Can you suggest a pipeline for eDNA metabarcoding analysis?",
    ],
    flagging_mode="never",
)

# 5. Εκτέλεση της εφαρμογής
if __name__ == "__main__":
    demo.launch(inbrowser=True)

# Api Key: this code assumes that you have set your OpenAI API key in an environment variable or a .env file. Make sure to create a .env file with the following content:
# OPENAI_API_KEY=your_openai_api_key_here
    import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv(override=True)
client = OpenAI()
MODEL = "gpt-4o-mini"

# 2. System prompt defining the assistant's persona
SYSTEM_MESSAGE = """
You are an expert Bioinformatics & Genomics AI Assistant.
You help researchers and students understand sequence analysis, variant interpretation, and computational biology tools.
Provide concise, clear answers in Markdown format.
"""

# 3. Callback function for Gradio ChatInterface
def chat_callback(message: str, history: list[dict]):
    """
    Handles user input, maintains conversation history, and streams responses.
    
    Args:
        message (str): The current user query.
        history (list[dict]): Previous message dictionaries [{'role': ..., 'content': ...}].
    """
    # Format existing conversation history for OpenAI payload
    formatted_history = [
        {"role": item["role"], "content": item["content"]}
        for item in history
    ]
    
    # Construct full message payload: System Prompt + History + Current User Message
    messages = (
        [{"role": "system", "content": SYSTEM_MESSAGE}]
        + formatted_history
        + [{"role": "user", "content": message}]
    )
    
    # Send streaming request to OpenAI API
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )
    
    # Yield incremental response tokens for live UI update
    response_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        response_text += delta
        yield response_text

# 4. Build and configure Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat_callback,
    type="messages",
    title="🧬 Bioinformatics AI Assistant",
    description="Ask questions about Next-Generation Sequencing (NGS), variant effects, or bioinformatics workflows.",
    examples=[
        "What is the difference between a missense and a nonsense mutation?",
        "How does the Phred quality score work in FASTQ files?",
        "Can you suggest a pipeline for eDNA metabarcoding analysis?"
    ],
    flagging_mode="never"
)

# 5. Launch the application
if __name__ == "__main__":
    demo.launch(inbrowser=True)