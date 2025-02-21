import os
from anthropic import Anthropic
from ollama import generate
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
        import json
        
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""You are an AI assistant that analyzes documents for relevance.
        You must use the provided tool to return your analysis.
        
        Tool specification:
        {self.document_analysis_tool}
        
        Query to evaluate against: {query}
        
        Task: Analyze these documents and identify the ones most relevant to the query:
        {context}
        
        Respond only with the tool output in JSON format.
        """
        
        # Call Claude using the new method
        response = self.call_claude(prompt)
        
        # For debugging
        print("Analysis response:", response)
        
        try:
            # Extract JSON part from the response
            json_str = response[0].text.split('\n\nReasoning:')[0].strip()
            result = json.loads(json_str)
            
            # get the number of relevant documents
            relevant_docs = result.get("relevant_docs", [])
            is_enough = len(relevant_docs) > 6 
            
            # content of the relevant documents
            related_docs_content = [documents[i-1] for i in relevant_docs]
            context_related_docs = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(related_docs_content)
            ])
            print("Related documents content:", context_related_docs)

            return is_enough, context_related_docs
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Error processing response: {e}")
            return False, ""

    def generate_improved_query(self, query, documents):
        # Combine the query and documents to generate an improved query
        context = documents
        prompt = f"""Generate an improved search query based on this context and question.
        Do not try to answer the question, only improve the search query.
        
        Context:
        {context}
        
        Original Question:
        {query}
        
        Task: Generate a modified version of the question that would help find more relevant documents.
        Return only the improved query text, no explanations.
        """
        
        # Call Claude instead of Ollama
        improved_query = self.call_claude(prompt)
        
        # Keep the Ollama call as a comment for reference
        # improved_query = self.call_ollama_generate(prompt)
        # return the content of the response
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