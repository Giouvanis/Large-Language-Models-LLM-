import os
import json
import sqlite3
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# 1. Setup & Environment
load_dotenv(override=True)
client = OpenAI()
MODEL = "gpt-4o-mini"
DB_NAME = "genomics.db"

# ==========================================
# 2. SQLITE DATABASE INITIALIZATION
# ==========================================

def init_db():
    """Initializes SQLite database with schema and sample genomic data."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Create Variants Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS variants (
                variant_id TEXT PRIMARY KEY,
                gene_symbol TEXT,
                hgvs_c TEXT,
                hgvs_p TEXT,
                clinical_significance TEXT,
                phenotype TEXT
            )
        """)
        
        # Insert initial benchmark records
        sample_data = [
            ("VAR001", "BRCA1", "c.5266dupC", "p.Gln1756ProfsTer74", "Pathogenic", "Hereditary Breast and Ovarian Cancer"),
            ("VAR002", "BRCA1", "c.181T>G", "p.Cys61Gly", "Pathogenic", "Hereditary Breast and Ovarian Cancer"),
            ("VAR003", "BRCA2", "c.5946del", "p.Ser1982RfsTer22", "Pathogenic", "Breast-Ovarian Cancer Familial"),
            ("VAR004", "TP53", "c.743G>A", "p.Arg248Gln", "Pathogenic", "Li-Fraumeni Syndrome"),
            ("VAR005", "EGFR", "c.2573T>G", "p.Leu858Arg", "Pathogenic", "Non-Small-Cell Lung Cancer"),
            ("VAR006", "BRCA1", "c.4837A>G", "p.Ser1613Gly", "Uncertain Significance", "Hereditary Breast and Ovarian Cancer")
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO variants 
            (variant_id, gene_symbol, hgvs_c, hgvs_p, clinical_significance, phenotype)
            VALUES (?, ?, ?, ?, ?, ?)
        """, sample_data)
        
        conn.commit()

init_db()

# ==========================================
# 3. PYTHON TOOL IMPLEMENTATIONS
# ==========================================

def query_variant(identifier: str) -> str:
    """Queries variant by Variant ID or HGVS coding DNA notation."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT variant_id, gene_symbol, hgvs_c, hgvs_p, clinical_significance, phenotype 
            FROM variants 
            WHERE variant_id = ? OR hgvs_c = ?
        """, (identifier.strip(), identifier.strip()))
        
        row = cursor.fetchone()
        if row:
            return (
                f"Variant Found: ID: {row[0]} | Gene: {row[1]} | DNA: {row[2]} | "
                f"Protein: {row[3]} | Classification: {row[4]} | Condition: {row[5]}"
            )
        return f"No record found in database for identifier '{identifier}'."

def query_gene_variants(gene_symbol: str) -> str:
    """Lists all cataloged variants for a specific gene."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT variant_id, hgvs_c, clinical_significance 
            FROM variants 
            WHERE UPPER(gene_symbol) = UPPER(?)
        """, (gene_symbol.strip(),))
        
        rows = cursor.fetchall()
        if not rows:
            return f"No variants cataloged for gene '{gene_symbol}'."
        
        results = [f"• {r[0]} ({r[1]}): {r[2]}" for r in rows]
        return f"Cataloged variants for {gene_symbol.upper()}:\n" + "\n".join(results)

def update_variant_pathogenicity(variant_id: str, new_classification: str) -> str:
    """Updates the clinical significance/classification of a variant."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT variant_id FROM variants WHERE variant_id = ?", (variant_id.strip(),))
        if not cursor.fetchone():
            return f"Error: Variant ID '{variant_id}' does not exist in database."
        
        cursor.execute("""
            UPDATE variants 
            SET clinical_significance = ? 
            WHERE variant_id = ?
        """, (new_classification.strip(), variant_id.strip()))
        conn.commit()
        return f"Successfully updated {variant_id} classification to '{new_classification}'."

# ==========================================
# 4. TOOL SCHEMAS FOR OPENAI FUNCTION CALLING
# ==========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_variant",
            "description": "Look up genomic variant details by Variant ID (e.g. VAR001) or HGVS c. notation (e.g. c.5266dupC).",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "The Variant ID or HGVS c. notation."
                    }
                },
                "required": ["identifier"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_gene_variants",
            "description": "Retrieve all variants associated with a given gene symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "The official gene symbol (e.g. BRCA1, TP53, EGFR)."
                    }
                },
                "required": ["gene_symbol"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_variant_pathogenicity",
            "description": "Update the clinical pathogenicity classification for an existing variant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {
                        "type": "string",
                        "description": "The Variant ID to update (e.g. VAR006)."
                    },
                    "new_classification": {
                        "type": "string",
                        "description": "The updated classification (e.g. Pathogenic, Likely Pathogenic, Benign, Uncertain Significance)."
                    }
                },
                "required": ["variant_id", "new_classification"],
                "additionalProperties": False
            }
        }
    }
]

# ==========================================
# 5. DISPATCHER & CHAT ENGINE
# ==========================================

def handle_tool_calls(message) -> list[dict]:
    responses = []
    for tool_call in message.tool_calls:
        fn_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        if fn_name == "query_variant":
            result = query_variant(args.get("identifier", ""))
        elif fn_name == "query_gene_variants":
            result = query_gene_variants(args.get("gene_symbol", ""))
        elif fn_name == "update_variant_pathogenicity":
            result = update_variant_pathogenicity(
                args.get("variant_id", ""),
                args.get("new_classification", "")
            )
        else:
            result = "Error: Tool execution failed."
            
        responses.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })
    return responses

SYSTEM_PROMPT = """
You are a Clinical Genomics Assistant with direct access to an internal SQLite genomic database.
You help clinical bioinformaticians query variant classifications, check gene listings, and update pathogenicity annotations.
Use your tools to query and update the database accurately.
Provide concise, professional responses in Markdown.
"""

def chat(message: str, history: list[dict]):
    formatted_history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + formatted_history + [{"role": "user", "content": message}]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    
    # Handle sequential / multiple tool calls
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
# 6. GRADIO CHAT INTERFACE
# ==========================================

demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="🧬 ClinDB AI - Genomic Variant Database Assistant",
    description="Query or update internal genomic records using natural language tool-calling connected to SQLite.",
    examples=[
        "What variants do we have for BRCA1 in our database?",
        "Look up variant VAR004 and tell me its clinical phenotype.",
        "Please update VAR006 classification from 'Uncertain Significance' to 'Likely Pathogenic' and verify it."
    ],
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)