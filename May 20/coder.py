import autogen
import subprocess
import tempfile
import os
import asyncio
from pylint import lint
import platform
import nest_asyncio
from io import StringIO
import re

config_list = [
    {
        "model": "gemini-2.0-flash-001", 
        "api_key": "", 
        "api_type": "google"
    }
]

# Python Executor function
async def execute_python_code(code):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        loop = asyncio.get_event_loop()
        def run_python_sync():
            result = subprocess.run(
                ['python', temp_file_path],
                capture_output=True,
                text=True
            )
            return result.returncode, result.stdout, result.stderr

        returncode, stdout, stderr = await loop.run_in_executor(None, run_python_sync)
        os.unlink(temp_file_path)

        return {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
    except Exception as e:
        return {"error": str(e)}

async def run_pylint(code):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        from io import StringIO
        from pylint.lint import Run
        from pylint.reporters.text import TextReporter

        stdout = StringIO()
        reporter = TextReporter(stdout)

        loop = asyncio.get_event_loop()
        def run_pylint_sync():
            Run([temp_file_path], reporter=reporter, exit=False)
            return stdout.getvalue()

        pylint_output = await loop.run_in_executor(None, run_pylint_sync)
        os.unlink(temp_file_path)

        return {
            "pylint_output": pylint_output,
            "pylint_errors": ""  # Errors are included in pylint_output
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        stdout.close()

# Define the Coder agent
coder = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="You are a skilled Python coder. Write clean, functional Python code based on the task provided. Ensure the code is well-documented and follows PEP 8 guidelines."
)

# Define the Debugger agent
debugger = autogen.AssistantAgent(
    name="Debugger",
    llm_config={"config_list": config_list},
    system_message="""Analyze code quality. If code passes with pylint score >=7/10 and runs successfully, 
    reply with 'ALL CHECKS PASSED. PYLINT SCORE: X/10. TERMINATE'"""
)

# Define the UserProxy agent
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=25,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
    llm_config={"config_list": config_list},
)

@user_proxy.register_for_execution()
@coder.register_for_llm(name="execute_python_code", description="Execute Python code and return the result")
async def execute_python_code_tool(code: str):
    result = await execute_python_code(code)
    return {
        "name": "execute_python_code",
        "content": {
            "status": "success" if result.get('returncode', 1) == 0 else "error",
            "output": result.get('stdout', ''),
            "error": result.get('stderr', '')
        }
    }

@user_proxy.register_for_execution()
@debugger.register_for_llm(name="run_pylint", description="Run Pylint on Python code and return the linter output")
async def run_pylint_tool(code: str):
    result = await run_pylint(code)
    score_match = re.search(r'rated at (\d+\.?\d*)/10', result["pylint_output"])
    return {
        "name": "run_pylint",
        "content": {
            "score": float(score_match.group(1)) if score_match else 0.0,
            "report": result["pylint_output"]
        }
    }


group_chat = autogen.GroupChat(
    agents=[user_proxy, coder, debugger],
    messages=[],
    max_round=50,
    speaker_selection_method="round_robin"
)

group_chat_manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: (
        "TERMINATE" in x.get("content", "") and
        any(str(score) in x.get("content", "") for score in range(6, 11))
    )
)

# Main async function to run the program
async def main():
    # Example task: Write a Python function to calculate factorial
    initial_task = """
    Write a Python function to calculate the factorial of a number.
    After writing, execute the code and run pylint to ensure it’s correct and clean.
    If there are issues, debug and fix them. The final code should run without errors and have a pylint score of at least 6/10.
    """

    # Start the group chat
    await user_proxy.initiate_chat(
        group_chat_manager,
        message=initial_task
    )

# Run the program
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  
    asyncio.run(main())