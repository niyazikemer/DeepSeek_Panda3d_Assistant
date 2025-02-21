from anthropic import Anthropic

class Agent:
    def __init__(self):
        self.client = Anthropic(api_key="your-api-key")
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

    def analyze_documents(self, query, documents):  # Added query parameter
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
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # For debugging
        print("Analysis response:", response.content)
        
        return False

    def generate_improved_query(self, query, documents):
        # Combine the query and documents to generate an improved query
        context = documents[0]
        prompt = f"Generate an improved query based on the  useful document and the question. Do not try to answer the question. This is the context:\n{context} and this is the question you need to improve \n {query}\n Please improve the question in a way that it is more likely to return useful documents."
        improved_query = self.call_ollama_generate(prompt)
        return improved_query

    def call_ollama_generate(self, text):
        response = generate(
            model='deepseek-r1:32b',
            prompt=text,
            options={'temperature': 0.65, 'top_p': 0.8, 'top_k': 50}
        )
        return response['response']  # 🠔 Change 'text' to 'response'