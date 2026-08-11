import os
import sys
from dotenv import load_dotenv
from litellm import completion

# 1. Φόρτωση API Keys
load_dotenv(override=True)

# 2. Ορισμός Ρόλων / Προσωπικοτήτων
system_alex = """
You are Dr. Alex, a cautious clinical geneticist. 
You are skeptical about AI in diagnostics, emphasizing strict validation, potential errors, and regulatory risks.
Keep your response short (2-3 sentences max).
"""

system_blake = """
You are Dr. Blake, an optimistic AI Bioinformatician. 
You focus on efficiency, speed, and how LLMs can automate PubMed/ClinVar literature synthesis for variants.
Keep your response short (2-3 sentences max).
"""

system_charlie = """
You are Dr. Charlie, a neutral Bioethics researcher. 
You balance both sides, focusing on patient safety, data privacy, and the importance of human-in-the-loop oversight.
Keep your response short (2-3 sentences max).
"""

# 3. Συνάρτηση για την κλήση του κάθε Bot
def get_bot_response(model_name: str, bot_name: str, system_prompt: str, conversation: str) -> str:
    user_prompt = f"""
    You are in a panel discussion. Here is the transcript so far:
    
    {conversation}
    
    Respond naturally as {bot_name} to the points raised.
    """
    try:
        response = completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[{bot_name} Error: {e}]"


# 4. Εκτέλεση της Συζήτησης
def start_panel_discussion():
    topic = "Can LLMs reliably assist in classifying BRCA1 Variants of Uncertain Significance (VUS)?"
    
    conversation = f"--- PANEL DISCUSSION ---\nTopic: {topic}\n"
    print(f"🔬 Starting Panel Discussion: {topic}\n" + "="*60 + "\n")

    # Αρχικό έναυσμα από τον Dr. Alex (GPT-4o-mini)
    initial_statement = "Dr. Alex: Using LLMs for BRCA1 variant interpretation is risky. Clinical decisions require 100% accuracy, and models hallucinate."
    conversation += initial_statement + "\n"
    print(f"🔴 {initial_statement}\n")

    # Γύρος 1: Ο Dr. Blake (Gemini) απαντά
    blake_reply = get_bot_response("gemini/gemini-2.5-flash-lite", "Dr. Blake", system_blake, conversation)
    blake_line = f"Dr. Blake: {blake_reply}"
    conversation += blake_line + "\n"
    print(f"🟢 {blake_line}\n")

    # Γύρος 1: Ο Dr. Charlie (Llama 3.2 Local) τοποθετείται
    charlie_reply = get_bot_response("ollama/llama3.2", "Dr. Charlie", system_charlie, conversation)
    charlie_line = f"Dr. Charlie: {charlie_reply}"
    conversation += charlie_line + "\n"
    print(f"🔵 {charlie_line}\n")

    # Γύρος 2: Ο Dr. Alex (GPT-4o-mini) ανταπαντά
    alex_reply = get_bot_response("gpt-4o-mini", "Dr. Alex", system_alex, conversation)
    alex_line = f"Dr. Alex: {alex_reply}"
    conversation += alex_line + "\n"
    print(f"🔴 {alex_line}\n")


if __name__ == "__main__":
    start_panel_discussion()