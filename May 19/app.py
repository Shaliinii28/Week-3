# app.py
import asyncio
import re
from flask import Flask, request, render_template
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import google.generativeai as genai
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

app = Flask(__name__)

# Configuration
GENAI_API_KEY = ""  # Replace with your API key

# Edge Browser Configuration
EDGE_OPTIONS = EdgeOptions()
EDGE_OPTIONS.use_chromium = True
EDGE_OPTIONS.add_argument("--headless=new")
EDGE_OPTIONS.add_argument("--disable-gpu")
EDGE_OPTIONS.add_argument("--no-sandbox")

# Initialize Gemini
genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')

# AutoGen Agent Configuration
llm_config = {
    "config_list": [{
        "model": "gemini-2.0-flash-001",
        "api_key": GENAI_API_KEY,
        "api_type": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta"
    }],
    "cache_seed": None
}

async def web_browser_tool(url: str):
    """Async web browser using Selenium Edge"""
    def sync_fetch():
        driver = webdriver.Edge(
            service=Service(EdgeChromiumDriverManager().install()),
            options=EDGE_OPTIONS
        )
        try:
            driver.get(url)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )  # Added missing parenthesis here
            content = driver.find_element(By.TAG_NAME, "body").text
            return content[:10000]  # Limit content length
        finally:
            driver.quit()
    
    return await asyncio.to_thread(sync_fetch)
async def summarize_tool(text: str):
    """Async summarization using Gemini"""
    response = await asyncio.to_thread(
        lambda: gemini_model.generate_content(
            f"Summarize this concisely, focusing on key points:\n\n{text}"
        )
    )
    return response.text

# Initialize Agents
researcher = AssistantAgent(
    name="Researcher",
    system_message="""You are a web research expert. Your tasks:
    1. Use web_browser_tool to gather information
    2. Extract key information with sources
    3. Return raw data in format:
       SUMMARY_START
       [Content]
       SUMMARY_END""",
    llm_config=llm_config,
    human_input_mode="NEVER",
    function_map={"web_browser_tool": web_browser_tool}
)

summarizer = AssistantAgent(
    name="Summarizer",
    system_message="""You are a summarization expert. Your tasks:
    1. Process research data
    2. Generate concise markdown summary
    3. Format output as:
       SUMMARY_START
       [Summary]
       SUMMARY_END""",
    llm_config=llm_config,
    human_input_mode="NEVER",
    function_map={"summarize_tool": summarize_tool}
)

user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config=False,
    llm_config=llm_config
)

# Configure Group Chat
groupchat = GroupChat(
    agents=[user_proxy, researcher, summarizer],
    messages=[],
    max_round=6,
    speaker_selection_method="round_robin"
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

@app.route("/", methods=["GET", "POST"])
async def index():
    if request.method == "POST":
        query = request.form["query"]
        
        async def process_query():
            await user_proxy.a_initiate_chat(
                manager,
                message=f"Research and summarize: {query}"
            )
            
            # Extract summary from messages
            pattern = re.compile(r'SUMMARY_START(.*?)SUMMARY_END', re.DOTALL)
            summary = ""
            for msg in groupchat.messages:
                if match := pattern.search(msg.get('content', '')):
                    summary = match.group(1).strip()
                    break
            
            # Convert markdown to HTML-friendly format
            if summary:
                # Handle bullet points
                summary = summary.replace("*   ", "• ")
                # Remove residual markdown
                summary = summary.replace("**", "")
            
            return summary or "No summary could be generated"

        summary = await process_query()
        return render_template("result.html", summary=summary, query=query)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, use_reloader=False)