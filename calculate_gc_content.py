import os
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from Bio import Entrez, SeqIO
from io import StringIO

load_dotenv(override=True)
client = OpenAI()
MODEL = "gpt-4o-mini"

# Ρύθμιση email για το NCBI Entrez API
Entrez.email = "bio_researcher@example.com"

# ==========================================
# 1. ΠΡΑΓΜΑΤΙΚΕΣ ΒΙΟΠΛΗΡΟΦΟΡΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ==========================================

def calculate_gc_content(sequence: str) -> str:
    """Υπολογίζει το ποσοστό GC σε μια αλληλουχία DNA/RNA."""
    seq = sequence.upper().strip()
    valid_bases = [b for b in seq if b in "ATGCU"]
    if not valid_bases:
        return "Error: Invalid or empty nucleotide sequence."
    gc_count = seq.count("G") + seq.count("C")
    gc_percent = (gc_count / len(valid_bases)) * 100
    return f"Sequence Length: {len(valid_bases)} bp | GC Content: {gc_percent:.2f}%"

def fetch_gene_summary(gene_symbol: str, organism: str = "Homo sapiens") -> str:
    """Αναζητά και αντλεί τη λειτουργική περίληψη ενός γονιδίου από το NCBI."""
    try:
        query = f"{gene_symbol}[Gene Name] AND {organism}[Organism]"
        search_handle = Entrez.esearch(db="gene", term=query, retmax=1)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        id_list = search_results.get("IdList", [])
        if not id_list:
            return f"No NCBI Gene entry found for '{gene_symbol}' in {organism}."
        
        gene_id = id_list[0]
        summary_handle = Entrez.esummary(db="gene", id=gene_id)
        summary_record = Entrez.read(summary_handle)
        summary_handle.close()
        
        doc_summary = summary_record["DocumentSummarySet"]["DocumentSummary"][0]
        name = doc_summary.get("Name", gene_symbol)
        desc = doc_summary.get("Description", "No description available")
        summary_text = doc_summary.get("Summary", "No summary provided by NCBI")
        
        return f"Gene: {name} (ID: {gene_id})\nDescription: {desc}\nNCBI Summary: {summary_text}"
    except Exception as e:
        return f"NCBI API Error: {str(e)}"

# ==========================================
# 2. ΟΡΙΣΜΟΣ TOOLS (JSON SCHEMA)
# ==========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_gc_content",
            "description": "Calculate the GC content percentage and length of a given DNA/RNA sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "The raw nucleotide sequence (e.g. ATGCGATCGATCG)."
                    }
                },
                "required": ["sequence"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_gene_summary",
            "description": "Fetch official summary, ID, and description of a gene from the NCBI Gene database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "The standard gene symbol, e.g. BRCA1, TP53, EGFR."
                    },
                    "organism": {
                        "type": "string",
                        "description": "The organism name, default is 'Homo sapiens'."
                    }
                },
                "required": ["gene_symbol"],
                "additionalProperties": False
            }
        }
    }
]

# ==========================================
# 3. TOOL DISPATCHER & CHAT ENGINE
# ==========================================

def handle_tool_calls(message) -> list[dict]:
    responses = []
    for tool_call in message.tool_calls:
        fn_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        if fn_name == "calculate_gc_content":
            result_str = calculate_gc_content(args.get("sequence", ""))
        elif fn_name == "fetch_gene_summary":
            result_str = fetch_gene_summary(
                gene_symbol=args.get("gene_symbol"),
                organism=args.get("organism", "Homo sapiens")
            )
        else:
            result_str = "Error: Unknown tool."
            
        responses.append({
            "role": "tool",
            "content": result_str,
            "tool_call_id": tool_call.id
        })
    return responses

SYSTEM_PROMPT = """
You are an expert Bioinformatics AI Agent.
You assist researchers with genomic analyses, variant evaluations, and sequence statistics.
When asked about specific genes or sequence properties, use your available tools to retrieve factual data.
Always provide structured and concise explanations in Markdown.
"""

def chat(message: str, history: list[dict]):
    formatted_history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + formatted_history + [{"role": "user", "content": message}]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    
    # Διαχείριση διαδοχικών Tool Calls
    while response.choices[0].finish_reason == "tool_calls":
        msg = response.choices[0].message
        tool_responses = handle_tool_calls(msg)
        messages.append(msg)
        messages.extend(tool_responses)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
        
    return response.choices[0].message.content

# ==========================================
# 4. GRADIO CHAT UI
# ==========================================

demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="🧬 BioTools AI - Genomic Assistant",
    description="Ask gene summaries or submit nucleotide sequences for automated GC calculation.",
    examples=[
        "Can you fetch the NCBI summary for the TP53 gene in humans?",
        "What is the GC content of this sequence: ATGCGATCGATCGATCGATCGATCGCGCGCG?",
        "Give me information about BRCA2 and calculate the GC of 'ATGGCTTAGC'."
    ],
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)