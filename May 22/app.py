import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import tiktoken

# Initialize environment and async support
load_dotenv()
nest_asyncio.apply()

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB with default embeddings
default_ef = embedding_functions.DefaultEmbeddingFunction()

class DocumentSystem:
    def __init__(self):
        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=default_ef
        )
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')
        self.loop = asyncio.new_event_loop()
        
    def chunk_text(self, text, max_tokens=500):
        """Split text into manageable chunks"""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk = tokens[i:i + max_tokens]
            chunks.append(encoding.decode(chunk))
        
        return chunks

    def add_document(self, text: str):
        """Add a document to the collection"""
        try:
            chunks = self.chunk_text(text)
            
            # Generate IDs for each chunk
            existing_ids = self.collection.get()['ids']
            new_ids = [f"doc{len(existing_ids) + i + 1}" for i in range(len(chunks))]
            
            # Store the chunks
            self.collection.add(
                documents=chunks,
                ids=new_ids
            )
            return True
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            return False

    async def _get_answer(self, question: str):
        """Internal async method to get answer"""
        try:
            # Retrieve relevant document chunks
            results = self.collection.query(
                query_texts=[question],
                n_results=min(5, len(self.collection.get()['ids']))
            )
            
            if not results['documents']:
                return "No relevant information found in the documents."
                
            context = "\n\n".join([
                f"Document excerpt {i+1}:\n{chunk}" 
                for i, chunk in enumerate(results['documents'][0])
            ])
            
            # Generate answer using Gemini
            response = await self.model.generate_content_async(
                f"""Analyze the following document excerpts and provide a comprehensive answer to the question.
                If the answer cannot be determined from the context, say "I don't know based on the provided documents."
                
                Question: {question}
                
                Document excerpts:
                {context}
                
                Comprehensive answer:"""
            )
            return response.text
        except Exception as e:
            print(f"Error getting answer: {str(e)}")
            return "I encountered an error processing your question."

    def get_answer(self, question: str):
        """Synchronous wrapper for async answer generation"""
        return self.loop.run_until_complete(self._get_answer(question))

# Initialize document system
doc_system = DocumentSystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add', methods=['POST'])
def add_document():
    text = request.form.get('text', '').strip()
    
    if not text:
        return jsonify({'success': False, 'error': 'Text is required'})
    
    if doc_system.add_document(text):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to add document'})

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    answer = doc_system.get_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)