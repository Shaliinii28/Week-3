import asyncio
import os
import re
import logging
import nest_asyncio
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from bs4 import BeautifulSoup

# Patch event loop for Flask
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup Gemini LLM
os.environ["GOOGLE_API_KEY"] = ""  
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.7)

# Define async web browser tool
async def web_browser_tool(query: str, max_pages: int = 3) -> str:
    try:
        edge_options = Options()
        edge_options.add_argument("--headless")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=edge_options)

        search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        driver.get(search_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('http') and not href.startswith(('https://www.bing.com', 'http://www.bing.com')):
                links.append(href)

        content = []
        for i, link in enumerate(links[:max_pages]):
            try:
                driver.get(link)
                await asyncio.sleep(2)
                page_soup = BeautifulSoup(driver.page_source, 'html.parser')
                text_elements = page_soup.find_all(['p', 'h1', 'h2', 'h3'])
                page_text = ' '.join([elem.get_text().strip() for elem in text_elements])
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                if page_text:
                    content.append(f"Source {i+1} ({link}): {page_text[:1000]}...")
            except Exception as e:
                logger.error(f"Error fetching {link}: {str(e)}")
                continue

        driver.quit()
        return "\n\n".join(content) if content else "No relevant content found."
    except Exception as e:
        logger.error(f"Web browser tool error: {str(e)}")
        return "Error fetching web content."

# Define summarizer tool
def text_summarizer_tool(text: str, max_length: int = 200) -> str:
    try:
        summarizer_template = """You are a helpful summarization agent.
Please provide a concise summary of the following content, no longer than 200 words.

{text}"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template=summarizer_template
        )
        completion = llm.predict(prompt.format(text=text))
        return completion
    except Exception as e:
        logger.error(f"Text summarizer error: {str(e)}")
        return "Error summarizing text."

# LangChain tool definitions
tools = [
    Tool(
        name="WebBrowser",
        func=web_browser_tool,
        description="Fetches content from web pages based on a query. Returns raw text content.",
        coroutine=web_browser_tool
    ),
    Tool(
        name="TextSummarizer",
        func=text_summarizer_tool,
        description="Summarizes text to a specified length."
    )
]

# ReAct-style prompts
researcher_prompt_template = """You are an intelligent agent with access to the following tools:
{tools}

Available tool names: {tool_names}

You MUST respond in the following exact format:

Give a long answer

Give plain text answer only without additonal symbols. Just raw text

Repeat the above format if multiple tool invocations are needed.

Do NOT include any other text before or after. Do NOT break the format. Do NOT explain your steps outside this format.

Answer the user's query: {input}

Chat history:
{chat_history}

{agent_scratchpad}
"""

summarizer_prompt_template = """You are a helpful assistant.
{tools}

Available tool names: {tool_names}

Your job is to summarize the following content in no more than 200 words.

Content:
{input}

{agent_scratchpad}
"""

# PromptTemplates
researcher_prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "chat_history", "agent_scratchpad"],
    template=researcher_prompt_template,
)

summarizer_prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=summarizer_prompt_template,
)

# Create agents
researcher = create_react_agent(llm, tools, researcher_prompt)
summarizer = create_react_agent(llm, tools, summarizer_prompt)

# Agent Executors
researcher_executor = AgentExecutor(agent=researcher, tools=tools, verbose=True)
summarizer_executor = AgentExecutor(agent=summarizer, tools=tools, verbose=True)

def clean_agent_output(output: str) -> str:
    # Remove lines starting with [Thought], [Action], [Input], [Observation], etc.
    cleaned_lines = []
    for line in output.splitlines():
        if not re.match(r"^\[(Thought|Action|Input|Observation)\]:", line):
            cleaned_lines.append(line)
    # Join remaining lines and strip extra whitespace
    return "\n".join(cleaned_lines).strip()


# Round robin controller
class RoundRobinGroupChat:
    def __init__(self, agents: List, max_rounds: int = 2):
        self.agents = agents
        self.max_rounds = max_rounds
        self.chat_history = []

    async def run(self, query: str) -> str:
        self.chat_history.append(HumanMessage(content=query))

        last_output = None  # to store Summarizer output

        for round in range(self.max_rounds):
            for agent_name, agent_executor in self.agents:
                input_content = self.chat_history[-1].content
                if agent_name == "Researcher" and round > 0:
                    continue
                result = await agent_executor.ainvoke({
                    "input": input_content,
                    "chat_history": self.chat_history,
                    "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "agent_scratchpad": []
                })
                output = result['output']
                clean_output = clean_agent_output(output)
                self.chat_history.append(AIMessage(content=clean_output))
                logger.info(f"{agent_name} output: {clean_output[:100]}...")

                # Save only Summarizer output
                if agent_name == "Summarizer":
                    last_output = clean_output

        return last_output if last_output is not None else self.chat_history[-1].content


# Instance of group chat
group_chat = RoundRobinGroupChat(
    agents=[
        ("Researcher", researcher_executor),
        ("Summarizer", summarizer_executor)
    ],
    max_rounds=1
)

# Flask-compatible interface
def run_research_assistant(query: str) -> str:
    try:
        return asyncio.run(group_chat.run(query))
    except Exception as e:
        logger.error(f"Assistant error: {str(e)}")
        return f"Error running assistant: {str(e)}"
