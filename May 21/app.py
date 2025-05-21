# app.py
import os
import uuid
import asyncio
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

config_list = [
    {
        "model": "gemini-2.0-flash-001",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta"
    }
]

async def analyze_data(file_path):
    df = pd.read_csv(file_path)
    summary = df.describe().to_markdown()
    return f"Data Summary:\n{summary}"

async def generate_visualization(file_path, img_path, chart_type):
    df = pd.read_csv(file_path)
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) == 0:
        return None
        
    if chart_type == 'histogram':
        df[numeric_cols].hist()
    elif chart_type == 'bar':
        df[numeric_cols].mean().plot(kind='bar')
    elif chart_type == 'line':
        df[numeric_cols].plot()
    elif chart_type == 'box':
        df[numeric_cols].plot(kind='box')
    elif chart_type == 'scatter' and len(numeric_cols) >= 2:
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    else:
        df[numeric_cols].hist()  # default
    
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    return img_path

class DataFetcher(AssistantAgent):
    def __init__(self, name):
        super().__init__(
            name=name,
            llm_config={"config_list": config_list},
            system_message="Specialist in data retrieval and validation."
        )

class DataAnalyst(AssistantAgent):
    def __init__(self, name):
        super().__init__(
            name=name,
            llm_config={"config_list": config_list},
            system_message="Expert data analyst."
        )

async def run_analysis_pipeline(file_path, chart_type):
    unique_id = str(uuid.uuid4())
    img_path = os.path.join(app.config['STATIC_FOLDER'], f'plot_{unique_id}.png')
    
    fetcher = DataFetcher(name="Data_Fetcher")
    analyst = DataAnalyst(name="Data_Analyst")
    user_proxy = UserProxyAgent(name="User_Proxy", human_input_mode="NEVER")
    
    groupchat = GroupChat(
        agents=[user_proxy, fetcher, analyst],
        messages=[],
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})
    
    await user_proxy.a_initiate_chat(
        manager,
        message=f"Analyze this CSV file: {file_path}"
    )
    
    analysis_result = await analyze_data(file_path)
    visualization_path = await generate_visualization(file_path, img_path, chart_type)
    
    return analysis_result, visualization_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
async def analyze():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    chart_type = request.form.get('chart_type', 'histogram')
    
    if file.filename == '':
        return "No selected file", 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            analysis_result, viz_path = await run_analysis_pipeline(file_path, chart_type)
            plot_filename = os.path.basename(viz_path) if viz_path else None
            
            return render_template('results.html', 
                                analysis=analysis_result,
                                plot_url=plot_filename)
        except Exception as e:
            return f"Analysis Error: {str(e)}", 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return "Invalid file format. Please upload a CSV file.", 400

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)