# Import necessary libraries
import torch
from dotenv import load_dotenv
import os
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

class OutfitRecommender:
    def __init__(self):
        # Initialize system setup, models, vector store, and RAG pipeline
        self._setup_environment()
        self._initialize_models()
        self._setup_vector_store()
        self._setup_rag_chain()
    
    def _setup_environment(self):
        """Load environment variables, login to HuggingFace Hub, and detect GPU"""
        load_dotenv()
        login(token=os.getenv("HF_TOKEN"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _initialize_models(self):
        """Initialize embedding model and quantized LLM"""
        # Sentence embedding model for semantic similarity
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load tokenizer for the LLM
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set 4-bit quantization config for efficient GPU memory usage
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load quantized LLM with device mapping
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            device_map="auto",
            quantization_config=quant_config
        )
    
    def _setup_vector_store(self):
        """Connect to Qdrant vector store for retrieval"""
        self.collection_name = "styled_outfits"
        self.qdrant_url = os.getenv("QDRANT_URL")  
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Initialize Qdrant client and LangChain vector store wrapper
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )
        
        self.qdrant = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embed_model,
            content_payload_key="input_context",  # where text is stored
            metadata_payload_key="__all__"
        )

        # Configure retriever to fetch top-1 relevant item
        self.retriever = self.qdrant.as_retriever(search_kwargs={"k": 1})
    
    def _setup_rag_chain(self):
        """Define RAG pipeline combining retrieval and generation"""
        # Prompt template for instruct-style generation
        template = """<s>[INST] 
        You are a fashion stylist. Based on the following items, select a top, bottom, and shoes that best suit the user. 

        User: 
        {question}

        Available Items:
        {context}

        Explain your choices. 
        [/INST]"""

        self.prompt = PromptTemplate.from_template(template)
        
        # Build text generation pipeline using Hugging Face Transformers
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        self.hf = HuggingFacePipeline(pipeline=pipe)

        # Compose the final RAG chain: Retrieval → Prompt → Generation
        self.rag_chain = (
            RunnableLambda(self._custom_chain)
            | self.prompt
            | self.hf
        )
    
    def _custom_chain(self, inputs):
        """Retrieve documents and format context for the prompt"""
        docs = self.retriever.invoke(inputs["question"])
        formatted_context = self._format_documents(docs)
        return {
            "question": inputs["question"],
            "context": formatted_context
        }
    
    def _format_documents(self, docs):
        """Fetch and format structured clothing item descriptions"""
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[doc.metadata["_id"] for doc in docs],
            with_payload=True,
        )
        return "\n".join([
            f"Top: {record.payload['metadata']['top_description']}\n"
            f"Bottom: {record.payload['metadata']['bottom_description']}\n"
            f"Shoes: {record.payload['metadata']['shoes_description']}"
            for record in records if record.payload['metadata'].get("top_description")
        ])
    
    @staticmethod
    def _clean_response(text):
        """Remove instruction token wrappers from model output"""
        return text.split("[/INST]")[-1].strip()
    
    def recommend(self, query):
        """Main interface: takes user query and returns outfit recommendation"""
        # 1. Generate RAG response
        result = self.rag_chain.invoke({"question": query})
        response_text = self._clean_response(result)
        
        # 2. Retrieve original item details for context
        docs = self.retriever.invoke(query)
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[doc.metadata["_id"] for doc in docs],
            with_payload=True,
        )

        # 3. Return recommendation text + item metadata
        return {
            "outfit_recommendation": response_text,
            "outfit_details": records[0].payload if records else None
        }
