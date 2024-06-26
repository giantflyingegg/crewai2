import os
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from crewai import Agent, Task, Crew
from langchain_openai import OpenAI as LangChainOpenAI
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from chromadb import Client
from chromadb.config import Settings
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import io
from PIL import Image
import time
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set model url
model_url = "http://localhost:5000/v1"

# Download NLTK data for tokenization
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    raise

# Load environment variables
load_dotenv()

# Get the API key
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
if STABILITY_API_KEY is None:
    logger.error("STABILITY_API_KEY environment variable is not set")
    raise ValueError("STABILITY_API_KEY environment variable is not set")

# Initialize ChromaDB
try:
    chroma_client = Client(Settings(persist_directory="./chroma_db"))
    collection = chroma_client.create_collection(name="crypto_news")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    raise

# Initialize OpenAI client for local model
try:
    llm = LangChainOpenAI(
        base_url=model_url,
        api_key="dummy",
        model_name="mistral-7b-v0.1.Q2_K.gguf",
        temperature=0.7
    )

    chat_model = ChatOpenAI(
        base_url=model_url,
        api_key="dummy",
        model_name="mistral-7b-v0.1.Q2_K.gguf",
        temperature=0.7
    )
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Initialize Stability API client
try:
    stability_api = client.StabilityInference(
        key=STABILITY_API_KEY,
        verbose=True,
    )
except Exception as e:
    logger.error(f"Failed to initialize Stability API client: {e}")
    raise

def get_ai_response(prompt, model="mistral-7b-v0.1.Q2_K.gguf"):
    try:
        response = llm(prompt)
        return response
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return None

def get_article_urls_from_sitemap(sitemap_url):
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        urls = [url.text for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
        return urls
    except requests.RequestException as e:
        logger.error(f"Failed to fetch sitemap: {e}")
        return []
    except ET.ParseError as e:
        logger.error(f"Failed to parse sitemap XML: {e}")
        return []

def scrape_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1', class_='typography__StyledTypography-sc-owin6q-0').text.strip()
        content = ' '.join([p.text for p in soup.find_all('p', class_='typography__StyledTypography-sc-owin6q-0')])
        return {'title': title, 'content': content}
    except requests.RequestException as e:
        logger.error(f"Failed to scrape article {url}: {e}")
        return None
    except AttributeError as e:
        logger.error(f"Failed to parse article content {url}: {e}")
        return None

def scrape_crypto_news(sitemap_url):
    article_urls = get_article_urls_from_sitemap(sitemap_url)
    news_items = []
    for url in article_urls[:10]:  # Limit to 10 articles for example
        article = scrape_article(url)
        if article:
            news_items.append(article)
        time.sleep(1)  # Be polite, wait a second between requests
    return news_items

def tokenize_text(text):
    return word_tokenize(text)

def store_in_chroma(articles):
    try:
        for i, article in enumerate(articles):
            tokens = tokenize_text(article['content'])
            collection.add(
                documents=[article['content']],
                metadatas=[{"title": article['title']}],
                ids=[f"article_{i}"]
            )
    except Exception as e:
        logger.error(f"Failed to store articles in ChromaDB: {e}")
        raise

# Tool for web scraping
scrape_tool = Tool(
    name="Crypto News Scraper",
    func=lambda x: scrape_crypto_news(x),
    description="Scrapes crypto news from a given URL"
)

# Create an agent for web scraping
scraper_agent = Agent(
    role='Crypto News Web Scraper',
    goal='Extract the latest and most relevant crypto news articles',
    backstory="""You're a specialized web scraper with extensive knowledge of cryptocurrency markets.
    Your expertise lies in efficiently navigating financial news websites and identifying the most 
    impactful crypto-related stories. You're known for your ability to quickly parse through large 
    amounts of data and extract only the most relevant information.""",
    tools=[scrape_tool],
    llm=chat_model,
    verbose=True,
    allow_delegation=False
)

scrape_task = Task(
    description="""Scrape the latest crypto news articles from CoinDesk using the provided Crypto News Scraper tool.
    Use the following sitemap URL: https://www.coindesk.com/arc/outboundfeeds/news-sitemap-index/?outputType=xml
    Focus on extracting the most relevant and high-impact stories.""",
    expected_output="A list of 10 most relevant crypto news articles with titles and content",
    agent=scraper_agent
)

# Create a new agent for summarization
summarizer_agent = Agent(
    role='Crypto News Summarizer',
    goal='Create concise and informative summaries of crypto news articles',
    backstory="""You're a highly skilled content analyst specializing in cryptocurrency and blockchain technology.
    Your talent lies in distilling complex financial and technical information into clear, concise summaries 
    that capture the essence of each news article. You have a knack for identifying key trends and implications 
    in the crypto market.""",
    llm=chat_model,
    verbose=True,
    allow_delegation=False
)

summarize_task = Task(
    description="""Analyze and summarize the crypto news articles retrieved from ChromaDB. Your task is to:
    1. Read each article stored in ChromaDB.
    2. Create a concise summary (2-3 sentences) for each article, highlighting the main points.
    3. Identify key trends or significant market implications mentioned in the articles.
    4. Ensure that your summaries are clear, accurate, and capture the essential information.

    Remember to focus on the most important and impactful information in each article.""",
    expected_output="""A list of concise summaries for the top 10 crypto news articles. Each summary should include:
    - The article's title
    - A 2-3 sentence summary of the main points
    - Any significant trends or market implications identified
    
    The summaries should be clear, informative, and highlight the most crucial information from each article.""",
    agent=summarizer_agent
)

# Create a new agent for generating image prompts
image_prompt_agent = Agent(
    role='Crypto News Visualizer',
    goal='Transform crypto news summaries into vivid and relevant image prompts',
    backstory="""You're a creative genius with a deep understanding of both cryptocurrency concepts and visual storytelling.
    Your unique skill is translating complex financial news into compelling visual narratives. You excel at crafting image 
    prompts that not only represent the news accurately but also capture the mood and implications of crypto market events.""",
    llm=chat_model,
    verbose=True,
    allow_delegation=False
)

image_prompt_task = Task(
    description="Create vivid and conceptually rich image prompts that visually represent the key themes in each crypto news summary",
    expected_output="A list of 10 detailed image prompts, each corresponding to a crypto news summary, capturing the essence of the news in visual form",
    agent=image_prompt_agent
)

def get_chroma_content():
    try:
        results = collection.query(
            query_texts=[""],
            n_results=10
        )
        return results['documents'][0], results['metadatas'][0]
    except Exception as e:
        logger.error(f"Failed to retrieve content from ChromaDB: {e}")
        return [], []

def summarize_content(content, title):
    prompt = f"""Summarize the following crypto news article in 2-3 sentences, highlighting the main points and any significant market implications:

    Title: {title}

    Content: {content}

    Summary:"""
    return get_ai_response(prompt)

def generate_image_prompt(summary):
    prompt = f"""
    Based on the following summary of a crypto news article, create a detailed and vivid prompt for a text-to-image AI model. 
    The prompt should capture the key concepts and mood of the news in a visual manner. 
    Focus on symbolic representations, relevant imagery, and the overall atmosphere that the news conveys.

    Summary: {summary}

    Image Prompt:
    """
    return get_ai_response(prompt)

def generate_image(prompt):
    try:
        answers = stability_api.generate(
            prompt=prompt,
            seed=42,
            steps=30,
            cfg_scale=8.0,
            width=512,
            height=512,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2S_ANCESTRAL
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    logger.warning("NSFW content detected. Skipping image.")
                    return None
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    filename = f"generated_image_{int(time.time())}.png"
                    img.save(filename)
                    logger.info(f"Image saved as: {filename}")
                    return filename
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

# Create the crew
crew = Crew(
    agents=[scraper_agent, summarizer_agent, image_prompt_agent],
    tasks=[scrape_task, summarize_task, image_prompt_task],
    verbose=True
)

def run_summarization():
    documents, metadatas = get_chroma_content()
    summaries = []
    for doc, meta in zip(documents, metadatas):
        summary = summarize_content(doc, meta['title'])
        summaries.append({"title": meta['title'], "summary": summary})
    return summaries

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

if __name__ == "__main__":
    try:
        # Run the crew
        result = crew.kickoff()

        # Process and store the results
        articles = scrape_crypto_news("https://www.coindesk.com/arc/outboundfeeds/news-sitemap-index/?outputType=xml")
        store_in_chroma(articles)

        logger.info("Scraping completed and articles stored in ChromaDB.")

        # Run summarization
        summaries = run_summarization()

        # Run image prompt generation and image creation
        image_results = run_image_prompt_generation(summaries)

        # Save and display results
        for i, item in enumerate(image_results):
            logger.info(f"Title: {item['title']}")
            logger.info(f"Summary: {item['summary']}")
            logger.info(f"Image Prompt: {item['image_prompt']}")
            if item['image']:
                filename = f"crypto_news_image_{i}.png"
                Image.open(item['image']).save(filename)
                logger.info(f"Image saved as: {filename}")
            else:
                logger.warning("Image generation failed.")
            logger.info("---")
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
        logger.error(traceback.format_exc())