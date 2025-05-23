import os
import subprocess
import tempfile
import asyncio
from flask import Flask, render_template, request, jsonify
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, config_list_from_models
import google.generativeai as genai
import nest_asyncio
import platform
is_windows = platform.system() == 'Windows'

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

app = Flask(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Create proper LLM configuration for AutoGen
llm_config = {
    "config_list": [
        {
            "model": "gemini-2.0-flash-001",
            "api_key": GEMINI_API_KEY,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
            "api_type": "google"
        }
    ],
    "temperature": 0.3,
    "timeout": 120
}

class CodeTools:
    @staticmethod
    async def execute_python(code: str) -> dict:
        try:
            if is_windows:
                # Windows-specific implementation
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "temp_script.py")
                try:
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    
                    result = subprocess.run(
                        ["python", temp_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    return {
                        "success": result.returncode == 0,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                finally:
                    try:
                        os.remove(temp_path)
                        os.rmdir(temp_dir)
                    except Exception as e:
                        print(f"Windows cleanup error: {e}")
            else:
                # Unix/Linux/Mac implementation
                with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                
                try:
                    result = subprocess.run(
                        ["python", tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    return {
                        "success": result.returncode == 0,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        print(f"Unix cleanup error: {e}")
                        
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def run_linter(code: str) -> dict:
        try:
            if is_windows:
                # Windows-specific implementation
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "temp_lint.py")
                try:
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    
                    result = subprocess.run(
                        ["pylint", temp_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    score = 0
                    if "rated at" in result.stdout:
                        score_part = result.stdout.split("rated at")[1]
                        score = float(score_part.split("/")[0].strip())
                    
                    return {
                        "score": score,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                finally:
                    try:
                        os.remove(temp_path)
                        os.rmdir(temp_dir)
                    except Exception as e:
                        print(f"Windows cleanup error: {e}")
            else:
                # Unix/Linux/Mac implementation
                with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                
                try:
                    result = subprocess.run(
                        ["pylint", tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    score = 0
                    if "rated at" in result.stdout:
                        score_part = result.stdout.split("rated at")[1]
                        score = float(score_part.split("/")[0].strip())
                    
                    return {
                        "score": score,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        print(f"Unix cleanup error: {e}")
                        
        except Exception as e:
            return {"error": str(e)}
            
            
async def agent_process(user_query):
    try:
        print(f"Initializing agents for query: {user_query}")  # Debug log
        
        # Initialize your agents here
        coder = AssistantAgent(
            name="Coder",
            system_message="You are an expert Python developer...",
            llm_config=llm_config
        )

        debugger = AssistantAgent(
            name="Debugger",
            system_message="You are a senior software engineer. Analyze code for issues and suggest improvements.",
            llm_config=llm_config
        )

        user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False,
            llm_config=False
        )

        group_chat = GroupChat(
            agents=[user_proxy, coder, debugger],
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin"
        )

        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config
        )
        
        print("Agents initialized, starting chat...")  # Debug log
        await user_proxy.a_initiate_chat(
            manager,
            message=f"User request: {user_query}\n\nGenerate and validate Python code following best practices."
        )
        
        print(f"Chat completed with {len(group_chat.messages)} messages")  # Debug log
        return group_chat.messages
        
    except Exception as e:
        print(f"Error in agent_process: {str(e)}")  # Debug log
        raise

        @user_proxy.register_for_execution()
        @debugger.register_for_llm(description="Execute Python code and get results")
        async def python_executor(code: str) -> str:
            result = await CodeTools.execute_python(code)
            return str(result)  # Ensure consistent string response

        @user_proxy.register_for_execution()
        @debugger.register_for_llm(description="Lint Python code for quality checks")
        async def pylint_checker(code: str) -> str:
            result = await CodeTools.run_linter(code)
            return str(result)  # Ensure consistent string response

        return group_chat.messages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_request():
    user_query = request.json.get('query', '')
    
    async def run_agent():
        return await agent_process(user_query)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        chat_messages = loop.run_until_complete(run_agent())
        loop.close()
        
        # Process the conversation to extract the most relevant output
        output = ""
        for msg in chat_messages:
            if msg.get('name') == 'Coder' and msg.get('content'):
                content = msg['content']
                # Extract code blocks
                if '```python' in content:
                    output = content.split('```python')[1].split('```')[0].strip()
                    break
                elif '```' in content:
                    output = content.split('```')[1].split('```')[0].strip()
                    break
                else:
                    output = content
        
        if not output:
            output = "No code was generated. Please try a different query."

        return jsonify({
            "status": "success",
            "output": output,
            "conversation": chat_messages
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)