class Agent:
    def analyze_documents(self, documents):
        # Dummy method that sometimes returns False
        # For demonstration, let's assume it returns False if there are less than 5 documents
        return False

    def generate_improved_query(self, query, documents):
        # Generate an improved query suggestion based on the useful documents
        key_terms = self.extract_key_terms(query)
        useful_info = self.identify_useful_info(documents)
        improved_query = f"{key_terms} {useful_info}"
        return improved_query

    def extract_key_terms(self, query):
        # Dummy method to extract key terms from the query
        return query

    def identify_useful_info(self, documents):
        # Dummy method to identify useful information from the documents
        return " ".join([doc.page_content[:50] for doc in documents])