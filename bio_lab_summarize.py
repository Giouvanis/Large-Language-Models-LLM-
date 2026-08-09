import json
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI

# 1. Configuration & Setup
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY missing! Please check your .env file.")

client = OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini"


# 2. Web Scraping Helper Functions
def fetch_website_links(url: str) -> list[str]:
    """Scrapes all hyperlink URLs from a given webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Convert relative links to absolute URLs
            if href.startswith("/"):
                href = url.rstrip("/") + href
            if href.startswith("http"):
                links.append(href)
        return list(set(links))  # Remove duplicates
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch links from {url}: {e}")
        return []


def fetch_website_contents(url: str) -> str:
    """Extracts raw text content from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, and navigation noise
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.extract()

        text = soup.get_text(separator=" ", strip=True)
        return text[:3000]  # Cap length per page to manage context window
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch content from {url}: {e}")
        return ""


# 3. LLM Step 1: Select Relevant Research Links (JSON Output)
def select_research_links(url: str) -> dict:
    """Uses LLM to evaluate scraped links and return only research-relevant ones in JSON."""
    all_links = fetch_website_links(url)

    system_prompt = """
    You are a bioinformatics assistant analyzing links from a research laboratory website.
    Identify the most relevant links for understanding their scientific work, such as 'Research', 
    'Publications', 'Projects', 'Members', or 'Tools/Software'.
    
    Respond STRICTLY in JSON format following this exact structure:
    {
        "relevant_links": [
            {"type": "research page", "url": "https://example.com/research"},
            {"type": "publications", "url": "https://example.com/pubs"}
        ]
    }
    """

    user_prompt = f"""
    Target URL: {url}
    
    Found links:
    {chr(10).join(all_links[:50])}  # Evaluate top 50 links
    """

    print(f"🔍 Analyzing {len(all_links)} links to find research topics...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    return json.loads(response.choices[0].message.content)


# 4. Pipeline Assembly
def build_lab_context(url: str) -> str:
    """Combines landing page text with content from LLM-selected research links."""
    context = f"## Landing Page Content:\n\n{fetch_website_contents(url)}\n\n"
    selected_links_json = select_research_links(url)

    links_list = selected_links_json.get("relevant_links", [])
    print(f"✅ Found {len(links_list)} relevant research sections.")

    for item in links_list:
        link_type = item.get("type", "page")
        link_url = item.get("url")
        print(f"   --> Scraping [{link_type}]: {link_url}")
        page_text = fetch_website_contents(link_url)
        context += f"## Section [{link_type}]:\n{page_text}\n\n"

    return context[:8000]  # Safe token cap for GPT prompt


# 5. LLM Step 2: Generate Executive Summary (Streaming)
def generate_lab_summary(lab_name: str, url: str):
    """Streams an executive research summary formatted in Markdown."""
    raw_context = build_lab_context(url)

    system_prompt = """
    You are a principal scientific writer in bioinformatics and computational biology.
    Analyze the scraped website content of a research lab and produce an Executive Research Summary.
    
    Structure the report in Markdown using the following headings:
    - ## Core Research Focus
    - ## Key Projects & Methodologies
    - ## Notable Software / Tools / Publications (if available)
    - ## Potential Collaborations & Applications
    
    Respond directly in Markdown. Do not enclose the whole output in a python code block.
    """

    user_prompt = f"""
    Research Laboratory: {lab_name}
    Website Data:
    {raw_context}
    """

    print(f"\n🤖 Generating Executive Research Summary for '{lab_name}'...\n")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        stream=True,
    )

    # Real-time streaming output (Works in Jupyter & Terminal)
    full_response = ""
    display_handle = display(Markdown(""), display_id=True)

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        update_display(
            Markdown(full_response), display_id=display_handle.display_id
        )


# --- Execution Block ---
if __name__ == "__main__":
    # Example target: A real bioinformatics lab / research site
    target_lab_name = "Behavioral Neuroscience / Bioinformatics Lab"
    target_url = "https://huggingface.co"  # Or substitute with any lab URL

    generate_lab_summary(target_lab_name, target_url)