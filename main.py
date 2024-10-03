import os
import arxiv
import concurrent.futures
import json
import tiktoken
from openai import OpenAI
from PyPDF2 import PdfReader
from datetime import datetime
import requests
import chromadb
import logging
import subprocess
import tempfile
import re
import os
##########################################
 # define different api keys and urls for base for openai endpoints
####################################
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_url = "" # not used yet
openai_embedding_model = "text-embedding-3-large"
openai_chat_model = "gpt-4o-mini"
###########
ollama_api_key = "ollama"
ollama_api_url = os.getenv("OLLAMA_BASE_URL")
ollama_embedding_model = "mxbai-embed-large"
ollama_chat_model = "llama3.1:8b"
###########
groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_url = "https://api.groq.com/openai/v1"
# not used for embedding at all, need switcher for embedding model to swap to ollama embedding model if using groq.

groq_chat_model = "mixtral-8x7b-32768"
###########




####################################################
#################################################
################################################
####################################################





# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# set keys/base urls
groq_client = OpenAI(api_key=groq_api_key, base_url=groq_api_url)
openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_url)
ollama_client = OpenAI(api_key=ollama_api_key, base_url=ollama_api_url)

GPT_MODEL = ollama_chat_model
client = ollama_client

directory = './data/papers'
output_dir = './data/research_papers'

for dir_path in [directory, output_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"Created directory: {dir_path}")
    else:
        logging.info(f"Directory already exists: {dir_path}")

# Initialize ChromaDB client and collection
knowledge_db = chromadb.Client()
paper_collection = knowledge_db.get_or_create_collection("scientific_papers")

def embedding_request(text):
    logging.info(f"Generating embedding for text: {text[:30]}...")
    response = ollama_client.embeddings.create(input=text, model=ollama_embedding_model)
    return response

def get_articles(query, top_k=5):
    logging.info(f"Searching for articles with query: {query}")
    search = arxiv.Search(
        query=query,
        max_results=top_k,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    result_list = []
    arxiv_client = arxiv.Client()
    for result in arxiv_client.results(search):
        logging.info(f"Processing article: {result.title}")
        result_dict = {
            "title": result.title,
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "article_url": result.entry_id
        }
        result_list.append(result_dict)
        embedding = embedding_request(result.title).data[0].embedding
        try:
            paper_collection.add(
                embeddings=[embedding],
                metadatas=[{
                    "title": result.title,
                    "pdf_url": result.pdf_url,
                    "summary": result.summary,
                    "date": str(result.published)
                }],
                ids=[result.entry_id]
            )
            logging.info(f"Added article to collection: {result.title}")
        except Exception as e:
            logging.error(f"Error adding article to collection: {e}")
    return result_list

def rank_related_papers(query, top_n=5):
    logging.info(f"Ranking papers related to query: {query}")
    query_embedding = embedding_request(query).data[0].embedding
    results = paper_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    if results['metadatas']:
        titles = [metadata['title'] for metadata in results['metadatas'][0]]
        logging.info(f"Ranked papers: {titles}")
        return titles
    else:
        logging.warning("No related papers found.")
        return []

def download_pdf(paper):
    logging.info(f"Downloading PDF for paper: {paper['title']}")
    pdf_url = paper['pdf_url']
    response = requests.get(pdf_url)
    pdf_path = os.path.join(directory, f"{paper['title'].replace('/', '_')}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    return pdf_path

def read_pdf(pdf_path):
    logging.info(f"Reading PDF file: {pdf_path}")
    reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pdf_text += page_text + "\n"
    return pdf_text

def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        j = min(i + n, len(tokens))
        chunk = tokenizer.decode(tokens[i:j])
        yield chunk
        i = j

def extract_chunk(content, template_prompt):
    prompt = template_prompt + content
    logging.info("Generating summary for a chunk of text.")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def process_paper(paper_title):
    logging.info(f"Processing paper: {paper_title}")
    paper_data = paper_collection.get(where={"title": paper_title})
    if paper_data['metadatas']:
        paper = paper_data['metadatas'][0]
    else:
        logging.warning(f"Paper titled '{paper_title}' not found in the collection.")
        return None

    pdf_path = download_pdf(paper)
    pdf_text = read_pdf(pdf_path)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = list(create_chunks(pdf_text, 1500, tokenizer))
    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_chunk, chunk, "Summarize this text: ") for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            summaries.append(future.result())
    full_summary = "\n".join(summaries)
    return {"title": paper_title, "summary": full_summary}

def generate_research_paper(summaries):
    logging.info("Generating research paper from summaries.")
    combined_summaries = ""
    for paper in summaries:
        combined_summaries += f"Title: {paper['title']}\nSummary: {paper['summary']}\n\n"

    # Adding examples to the prompt, using ###python instead of ```python
    example_paper = """
Example Research Paper:
Title: Sample Title
Abstract: This paper explores the use of meta-learning algorithms in artificial intelligence. We propose a new method that improves learning efficiency.

Introduction:
Meta-learning, or learning to learn, has become an important area in AI research. In this paper, we...

[Rest of the example paper]
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": f"""
Using the following summaries of recent research in AI science and meta-learning, draft a new research paper exploring new scientific hypotheses.

Include the following sections: Abstract, Introduction, Related Work, Proposed Method, Experiments, Results, Discussion, and Conclusion.

Follow the style and format of the example below.

{example_paper}

Research Summaries:
{combined_summaries}
"""
        }],
        temperature=0
    )
    paper_content = response.choices[0].message.content
    paper_filename = f"research_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    paper_filepath = os.path.join(output_dir, paper_filename)
    with open(paper_filepath, 'w', encoding='utf-8') as file:
        file.write(paper_content)
    logging.info(f"Research paper saved to: {paper_filepath}")
    return paper_filepath

def test_generated_code(code):
    logging.info("Testing generated code.")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tmp_file:
        tmp_file.write(code)
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(
            ['python', tmp_file_path],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info("Code executed successfully.")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing code: {e.stderr}")
        return False, e.stderr
    finally:
        os.remove(tmp_file_path)

def extract_code_snippets(text):
    logging.info("Extracting code snippets from text.")
    code_snippets = []
    # Using ###python instead of ```python in the regex pattern
    pattern = r"###python(.*?)###"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        code_snippets.append(match.strip())
    return code_snippets

def execute_code_snippets(reflection):
    logging.info("Extracting and testing code snippets from the reflection.")
    code_snippets = extract_code_snippets(reflection)
    successful_codes = []
    for code in code_snippets:
        success, output = test_generated_code(code)
        if success:
            successful_codes.append(code)
        else:
            logging.warning("Code snippet failed execution and will not be included.")
    return successful_codes

def append_code_to_paper(paper_filepath, codes):
    logging.info("Appending successful code snippets to the paper.")
    with open(paper_filepath, 'a', encoding='utf-8') as file:
        file.write("\n\n# Code Implementations\n")
        for idx, code in enumerate(codes, 1):
            file.write(f"\n## Code Snippet {idx}\n")
            file.write(f"###python\n{code}\n###\n")

def reflect_and_improve(paper_content):
    logging.info("Reflecting on the research paper to suggest improvements.")
    # Adding examples to the prompt, using ###python instead of ```python
    example_reflection = """
Example Reflection:
The proposed method could be enhanced by incorporating a regularization term to prevent overfitting. Additionally, experimenting with different learning rates might improve convergence.
Here is an example implementation:

###python
def train_model(data, learning_rate=0.01, regularization=0.001):
    # Training code here
###
"""

    reflection_prompt = f"""
Reflect on this research paper. Identify areas of improvement, propose follow-up experiments or studies, suggest code implementations, and provide code snippets where appropriate.

Follow the style and format of the example below.

{example_reflection}

Research Paper Content:
{paper_content}
"""
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": reflection_prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def recursive_improvement_cycle():
    logging.info("Starting the recursive improvement cycle.")
    query = "AI meta-learning algorithms"
    get_articles(query)
    iteration = 0
    while True:
        iteration += 1
        logging.info(f"--- Iteration {iteration} ---")
        ranked_papers = rank_related_papers(query)
        if not ranked_papers:
            logging.error("No papers found to process. Exiting the cycle.")
            break
        summaries = []
        for title in ranked_papers:
            paper_summary = process_paper(title)
            if paper_summary:
                summaries.append(paper_summary)
            else:
                logging.warning(f"Could not process paper: {title}")
        if not summaries:
            logging.error("No summaries generated. Exiting the cycle.")
            break
        paper_filepath = generate_research_paper(summaries)
        with open(paper_filepath, 'r', encoding='utf-8') as file:
            paper_content = file.read()
        reflection = reflect_and_improve(paper_content)
        code_snippets = execute_code_snippets(reflection)
        if code_snippets:
            append_code_to_paper(paper_filepath, code_snippets)
        else:
            logging.info("No successful code snippets to append.")
        # Modify query or other parameters based on reflection if needed
        # For continuous loop, add a condition to break the loop if desired
        # Here, we'll run indefinitely until manually stopped
    logging.info("Recursive improvement cycle completed.")
    return paper_filepath

if __name__ == "__main__":
    final_paper = recursive_improvement_cycle()
    if final_paper:
        logging.info(f"Final research paper available at: {final_paper}")
    else:
        logging.error("No final paper was generated.")
