import os
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew
from langchain_community.llms import TextGenWebUI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from chromadb import Client
from chromadb.config import Settings
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import stability_sdk
from stability_sdk import client
import io
from PIL import Image

# Download NLTK data for tokenization
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Get the API key
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')

# Initialize ChromaDB
chroma_client = Client(Settings(persist_directory="./chroma_db"))
collection = chroma_client.create_collection(name="crypto_news")

# Initialize TextGenWebUI
llm = TextGenWebUI(base_url="http://localhost:5000", max_new_tokens=250)

# Initialize Stability API client
stability_api = client.StabilityInference(
    key=STABILITY_API_KEY,
    verbose=True,
)

# Web scraping function
def scrape_crypto_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # This is a basic example. Adjust the selectors based on the actual website structure
    articles = soup.find_all('article')
    return [{'title': article.find('h2').text, 'content': article.find('p').text} for article in articles]

# Tokenization function
def tokenize_text(text):
    return word_tokenize(text)

# Store in ChromaDB function
def store_in_chroma(articles):
    for i, article in enumerate(articles):
        tokens = tokenize_text(article['content'])
        collection.add(
            documents=[article['content']],
            metadatas=[{"title": article['title']}],
            ids=[f"article_{i}"]
        )

# Tool for web scraping
scrape_tool = Tool(
    name="Crypto News Scraper",
    func=lambda x: scrape_crypto_news(x),
    description="Scrapes crypto news from a given URL"
)

# Create an agent for web scraping
scraper_agent = Agent(
    role='Web Scraper',
    goal='Scrape crypto news articles',
    backstory='You are an expert at finding and extracting crypto news from websites',
    tools=[scrape_tool],
    llm=llm
)

# Task for scraping
scrape_task = Task(
    description="Scrape crypto news from https://example-crypto-news-site.com",
    agent=scraper_agent
)

# Create a new agent for summarization
summarizer_agent = Agent(
    role='Content Summarizer',
    goal='Summarize crypto news articles',
    backstory='You are an expert at condensing complex information into concise summaries',
    llm=llm
)

# Function to retrieve content from ChromaDB
def get_chroma_content():
    results = collection.query(
        query_texts=[""],
        n_results=10  # Adjust this number based on how many articles you want to summarize
    )
    return results['documents'][0], results['metadatas'][0]

# Function to summarize content
def summarize_content(content, title):
    summary_prompt = PromptTemplate(
        input_variables=["title", "content"],
        template="Summarize the following crypto news article in 2-3 sentences:\n\nTitle: {title}\n\nContent: {content}"
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = summary_chain.run(title=title, content=content)
    return summary

# Task for summarization
summarize_task = Task(
    description="Retrieve articles from ChromaDB and generate summaries",
    agent=summarizer_agent
)

# Create a new agent for generating image prompts
image_prompt_agent = Agent(
    role='Image Prompt Generator',
    goal='Create expert image prompts based on crypto news summaries',
    backstory='You are an expert at crafting detailed and creative image prompts that capture the essence of textual information',
    llm=llm
)

# Function to generate image prompts
def generate_image_prompt(summary):
    prompt_template = PromptTemplate(
        input_variables=["summary"],
        template="""
        Based on the following summary of a crypto news article, create a detailed and vivid prompt for a text-to-image AI model. 
        The prompt should capture the key concepts and mood of the news in a visual manner. 
        Focus on symbolic representations, relevant imagery, and the overall atmosphere that the news conveys.

        Summary: {summary}

        Image Prompt:
        """
    )
    prompt_chain = LLMChain(llm=llm, prompt=prompt_template)
    image_prompt = prompt_chain.run(summary=summary)
    return image_prompt.strip()

# Task for generating image prompts
image_prompt_task = Task(
    description="Generate expert image prompts based on crypto news summaries",
    agent=image_prompt_agent
)

# Function to generate images using Stability AI
def generate_image(prompt):
    try:
        # Set up our initial generation parameters
        answers = stability_api.generate(
            prompt=prompt,
            seed=42,  # Change this for different results
            steps=30,
            cfg_scale=8.0,
            width=512,
            height=512,
            samples=1,
            sampler=stability_sdk.interfaces.gooseai.generation.SAMPLER_K_DPMPP_2M
        )

        # Process the result
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == stability_sdk.interfaces.gooseai.generation.FILTER:
                    print("NSFW content detected. Skipping image.")
                    return None
                if artifact.type == stability_sdk.interfaces.gooseai.generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    return img
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Create the crew
crew = Crew(
    agents=[scraper_agent, summarizer_agent, image_prompt_agent],
    tasks=[scrape_task, summarize_task, image_prompt_task],
    verbose=True
)

# Function to run the summarization process
def run_summarization():
    documents, metadatas = get_chroma_content()
    summaries = []
    for doc, meta in zip(documents, metadatas):
        summary = summarize_content(doc, meta['title'])
        summaries.append({"title": meta['title'], "summary": summary})
    return summaries

# Function to run the image prompt generation and image creation process
def run_image_prompt_generation(summaries):
    image_prompts = []
    for summary in summaries:
        prompt = generate_image_prompt(summary['summary'])
        image = generate_image(prompt)
        
        image_prompts.append({
            "title": summary['title'],
            "summary": summary['summary'],
            "image_prompt": prompt,
            "image": image
        })
    return image_prompts

# Main execution
if __name__ == "__main__":
    # Run the crew
    result = crew.kickoff()

    # Process and store the results
    articles = scrape_crypto_news("https://example-crypto-news-site.com")  # Replace with actual URL
    store_in_chroma(articles)

    print("Scraping completed and articles stored in ChromaDB.")

    # Run summarization
    summaries = run_summarization()

    # Run image prompt generation and image creation
    image_results = run_image_prompt_generation(summaries)

    # Save and display results
    for i, item in enumerate(image_results):
        print(f"Title: {item['title']}")
        print(f"Summary: {item['summary']}")
        print(f"Image Prompt: {item['image_prompt']}")
        if item['image']:
            filename = f"crypto_news_image_{i}.png"
            item['image'].save(filename)
            print(f"Image saved as: {filename}")
        else:
            print("Image generation failed.")
        print("---")