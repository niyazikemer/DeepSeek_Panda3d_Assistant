from ollama import generate

class Agent:
    def analyze_documents(self, documents):
        # Dummy method that always returns False for now
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