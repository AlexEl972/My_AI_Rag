### 3. Mise à Jour du Script Python


import os
import boto3
import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

class RAGChatbot:
    def __init__(self):
        aws_config = {
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region_name': os.getenv('AWS_DEFAULT_REGION')
        }
        
        s3 = boto3.client('s3', **aws_config)
        
        bucket = os.getenv('S3_BUCKET')
        document_key = os.getenv('S3_DOCUMENT_KEY')
        
        local_file = 'downloaded_document.txt'
        s3.download_file(bucket, document_key, local_file)
        
        with open(local_file, 'r') as f:
            self.document = f.read()
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="doc_collection")
        

        embedding = self.embedding_model.encode([self.document]).tolist()
        self.collection.add(
            embeddings=embedding,
            documents=[self.document],
            ids=["doc_1"]
        )
        
        self.model = os.getenv('OLLAMA_MODEL', 'mistral')
    
    def retrieve_context(self, query, top_k=2):
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]
    
    def chat(self, query, temperature=0.7, use_rag=True):
        context = self.retrieve_context(query) if use_rag else []
        
        prompt = (f"Contexte: {' '.join(context)}\n" if context 
                  else "") + f"Question: {query}"
        
        response = ollama.chat(
            model=self.model, 
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature}
        )
        
        return response['message']['content']

def main():
    chatbot = RAGChatbot()
    
    print("🤖 RAG Chatbot (Tapez 'exit' pour quitter)")
    print("Modes disponibles :")
    print("1. Chat avec RAG (default)")
    print("2. Chat sans RAG")
    print("3. Changer la température")
    
    mode_rag = True
    temperature = 0.7
    
    while True:
        try:
            query = input("\n> ")
            
            if query.lower() == 'exit':
                break
            
            if query == '1':
                mode_rag = True
                print("✅ Mode RAG activé")
                continue
            
            if query == '2':
                mode_rag = False
                print("❌ Mode RAG désactivé")
                continue
            
            if query == '3':
                temperature = float(input("Nouvelle température (0-1) : "))
                print(f"🌡️ Température réglée à {temperature}")
                continue
            
            response = chatbot.chat(
                query, 
                temperature=temperature, 
                use_rag=mode_rag
            )
            
            print("\n🤖 Réponse :", response)
        
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break

if __name__ == "__main__":
    main()