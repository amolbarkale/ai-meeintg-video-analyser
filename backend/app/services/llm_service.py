import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.core.config import settings
from . import prompts

logger = logging.getLogger(__name__)

def generate_meeting_insights(transcript: str) -> dict:
    """
    Generates a comprehensive set of insights from a transcript using OpenAI API.
    
    Returns a dictionary containing:
    - abstract_summary
    - key_points
    - action_items
    - sentiment_analysis
    """
    logger.info("Initializing OpenAI ChatOpenAI")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=settings.OPENAI_API_KEY
    )

    insights = {}

    # --- 1. Generate Abstract Summary ---
    try:
        logger.info("Generating abstract summary...")
        prompt_template = PromptTemplate(template=prompts.abstract_summary_prompt, input_variables=["transcript"])
        chain = LLMChain(llm=llm, prompt=prompt_template)
        result = chain.invoke({"transcript": transcript})
        insights['abstract_summary'] = result.get('text', '') if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error generating abstract summary: {e}")
        insights['abstract_summary'] = "Error: Could not generate summary."

    # --- 2. Generate Key Points ---
    try:
        logger.info("Generating key points...")
        prompt_template = PromptTemplate(template=prompts.key_points_prompt, input_variables=["transcript"])
        chain = LLMChain(llm=llm, prompt=prompt_template)
        result = chain.invoke({"transcript": transcript})
        insights['key_points'] = result.get('text', '') if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error generating key points: {e}")
        insights['key_points'] = "Error: Could not generate key points."

    # --- 3. Generate Action Items ---
    try:
        logger.info("Generating action items...")
        prompt_template = PromptTemplate(template=prompts.action_items_prompt, input_variables=["transcript"])
        chain = LLMChain(llm=llm, prompt=prompt_template)
        result = chain.invoke({"transcript": transcript})
        insights['action_items'] = result.get('text', '') if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error generating action items: {e}")
        insights['action_items'] = "Error: Could not generate action items."

    # --- 4. Generate Sentiment Analysis ---
    try:
        logger.info("Generating sentiment analysis...")
        prompt_template = PromptTemplate(template=prompts.sentiment_analysis_prompt, input_variables=["transcript"])
        chain = LLMChain(llm=llm, prompt=prompt_template)
        result = chain.invoke({"transcript": transcript})
        insights['sentiment_analysis'] = result.get('text', '') if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error generating sentiment analysis: {e}")
        insights['sentiment_analysis'] = "Error: Could not generate sentiment analysis."

    logger.info("All insights generated successfully.")
    
    # --- 5. NEW: Topic Modeling (Tags) ---
    try:
        logger.info("Generating topic tags...")
        prompt = PromptTemplate(template=prompts.topic_modeling_prompt, input_variables=["transcript"])
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.invoke({"transcript": transcript})
        insights['tags'] = result.get('text', '') if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error generating topic tags: {e}")
        insights['tags'] = "Error: Could not generate tags."

    # --- 6. NEW: Knowledge Graph ---
    try:
        logger.info("Generating knowledge graph data...")
        prompt = PromptTemplate(template=prompts.knowledge_graph_prompt, input_variables=["transcript"])
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.invoke({"transcript": transcript})
        insights['knowledge_graph'] = result.get('text', '') if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {e}")
        insights['knowledge_graph'] = '{"nodes": [], "edges": []}' # Default to empty graph on error

    logger.info("All insights generated successfully.")
    
    return insights