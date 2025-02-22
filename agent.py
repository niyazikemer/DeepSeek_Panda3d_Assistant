import os
from anthropic import Anthropic
from ollama import generate
import json
class Agent:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.document_analysis_tool = {
            "name": "analyze_relevance",
            "description": "Analyzes documents and returns relevant document numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "relevant_docs": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Array of document numbers (1-based indexing) that are most relevant"
                    }
                },
                "required": ["relevant_docs"]
            }
        }

    def analyze_documents(self, query, documents):
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""You are a document relevance analyzer.

Tool specification:
{self.document_analysis_tool}

Query: {query}

Task: Identify document numbers that:
- Contain direct answers or code examples
- Provide technical explanations
- Include implementation details
- Offer essential context
- Supporting context that helps understand the solution

Consider both:
- Individual document relevance
- How documents complement each other
- Technical accuracy and completeness
Documents to analyze:
{context}

IMPORTANT: Return ONLY a JSON object with relevant document numbers. No explanations or reasoning needed.
Example response format:
{{"relevant_docs": [1, 4, 7]}}
"""
        
        response = self.call_claude(prompt)
        
        try:
            # Parse the JSON directly - no splitting needed
            result = json.loads(response[0].text)
            relevant_docs = result.get("relevant_docs", [])
            is_enough = len(relevant_docs) > 4 
            
            # Get content of relevant documents
            related_docs_content = [documents[i-1] for i in relevant_docs]
            context_related_docs = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(related_docs_content)
            ])
            print(f"Relevant documents: {relevant_docs}")
            return is_enough, context_related_docs
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Error processing response: {e}")
            return False, ""

    def generate_improved_query(self, query, documents):
        context = documents
        prompt = f"""You are a search query optimization specialist. Your task is to improve technical documentation queries.

Original Question: {query}

Context from current search:
{context}

Your task is to generate an improved search query that will yield better results. Consider:

1. TECHNICAL PRECISION
- Add specific technical terms found in the context
- Include relevant programming concepts
- Specify exact framework or library names

2. SCOPE ADJUSTMENT
- Broaden if too specific and missing context
- Narrow down if too general
- Focus on implementation details if needed

3. QUERY STRUCTURE
- Use proper technical terminology
- Include key programming concepts
- Maintain search relevance

4. SEARCH OPTIMIZATION
- Consider main components in the code examples, imported libraries, or functions


Examples of improvements:
❌ "How to use Bullet?" 
✅ "Bullet physics library in panda3d controls bullet physics engine and uses Bullet Nodes"

❌ "Simple Scene implementation" 
✅ "a Panda3D Scene includes NodePath, Nodes, lighting, and camera setup for a 3D environment"

Return only the improved query without explanations or additional text.
"""

        improved_query = self.call_claude(prompt)
        print(f"Improved query: {improved_query}")
        return improved_query
    
    def call_claude(self, prompt, max_tokens=500, temperature=0.0):
        """Make a call to Claude API with given prompt and parameters."""
        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content
    
    def call_ollama_generate(self, text):
        response = generate(
            model='deepseek-r1:32b',
            prompt=text,
            options={'temperature': 0.65, 'top_p': 0.8, 'top_k': 50}
        )
        return response['response']  # 🠔 Change 'text' to 'response'