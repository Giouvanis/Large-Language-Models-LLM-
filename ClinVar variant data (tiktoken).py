import tiktoken

# 1. Load the tokenizer for the specified model (e.g., gpt-4o-mini or gpt-4o)
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# 2. Real ClinVar variant description text
clinvar_text = (
    "NM_000059.4(BRCA2):c.5946del (p.Ser1982RfsTer22) is a pathogenic "
    "frameshift variant associated with hereditary breast and ovarian cancer."
)

# 3. Encode the text into token IDs
tokens = encoding.encode(clinvar_text)

# 4. Display high-level summary stats
print(f"📄 Raw Text:\n{clinvar_text}\n")
print(f"📊 Total Characters: {len(clinvar_text)}")
print(f"🔢 Total Tokens: {len(tokens)}")
print(f"🆔 Token IDs (First 10): {tokens[:10]}...\n")

# 5. Breakdown: Map each token ID back to its decoded string representation
print("--- Token-by-Token Breakdown ---")
for token_id in tokens[:12]:
    token_str = encoding.decode([token_id])
    print(f"ID: {token_id:<6} -> '{token_str}'")