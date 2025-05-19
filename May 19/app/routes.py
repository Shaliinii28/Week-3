from flask import Blueprint, render_template, request
from app.web_research_assistant import run_research_assistant

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        query = request.form['query']
        result = run_research_assistant(query)
    return render_template("index.html", result=result)
