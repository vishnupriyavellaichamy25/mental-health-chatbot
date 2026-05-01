import faiss
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGPipeline:
    def __init__(self):
        """
        Initializes the RAG Pipeline, loads FAISS index, texts, 
        embedding model, and the LLM.
        """
        print("Loading FAISS index and texts...")
        self.index = faiss.read_index('data/mental_health.index')
        with open('data/texts.pkl', 'rb') as f:
            self.texts = pickle.load(f)
            
        print("Loading embedder 'all-MiniLM-L6-v2'...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading LLM 'mistralai/Mistral-7B-Instruct-v0.2'...")
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Embeds query, searches FAISS, filters by cosine score > 0.3,
        and returns top-k texts.
        """
        query_vector = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, k)
        
        retrieved_texts = []
        for i, score in enumerate(scores[0]):
            if score > 0.3:
                idx = indices[0][i]
                if idx < len(self.texts):
                    retrieved_texts.append(self.texts[idx])
                    
        return retrieved_texts

    def build_prompt(self, user_message: str, context_docs: list, chat_history: list) -> str:
        """
        Builds the prompt using Mistral instruct format [INST]...[/INST].
        Includes context documents and chat history.
        """
        context_str = "\\n".join(context_docs)
        
        prompt = "[INST] You are a compassionate mental health AI companion. "
        prompt += "Use the following context to help answer the user's question empathetically. "
        prompt += "If the context does not help, provide a supportive and general response.\\n\\n"
        
        if context_str:
            prompt += f"Context:\\n{context_str}\\n\\n"
            
        if chat_history:
            prompt += "Chat History:\\n"
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\\n"
            prompt += "\\n"
            
        prompt += f"User: {user_message}\\n[/INST]"
        return prompt

    def generate_response(self, prompt: str) -> str:
        """
        Generates response from the LLM.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract everything after [/INST]
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
            
        return response

    def rag_respond(self, user_message: str, chat_history: list) -> str:
        """
        Combines retrieve, build_prompt, and generate_response.
        """
        context_docs = self.retrieve(user_message, k=5)
        prompt = self.build_prompt(user_message, context_docs, chat_history)
        response = self.generate_response(prompt)
        return response
