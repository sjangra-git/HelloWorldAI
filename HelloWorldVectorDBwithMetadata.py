import os
from openai import OpenAI
import chromadb
from pydantic import BaseModel, Field
from typing import Optional

"""
The Problem: Semantic Search vs. Hard Constraints
Look at your user query and your data:

User Query: "Show me all Adidas shoes on sale and their prices?"

The "Adidas A100" Document: "Adidas mens shoes model: A100... sale=no"

Because you are using text-embedding-3-small, the vector database looks for semantic similarity. The words "Adidas", "shoes", and "prices" are incredibly strong matches. Your vector DB is almost certainly going to return the A100 model because it looks like the right kind of data, completely ignoring the fact that sale=no directly violates the user's intent.

You tried to fix this using the System Prompt (CRITICAL: Only list specific shoe models...), forcing the LLM to do the heavy filtering. But relying on the LLM to clean up bad database results is risky and wastes tokens.

This gives you the absolute best of both worlds: 
strict, programmatic database filtering combined with the natural language understanding of an LLM.
"""
# 1. Define the schema you want the LLM to output
class DBFilter(BaseModel):
    brand: Optional[str] = Field(None, description="The shoe brand mentioned, e.g., 'Adidas', 'Nike', 'Puma'")
    on_sale: Optional[bool] = Field(None, description="True if the user is looking for items on sale or discount, False otherwise.")
    price: Optional[int] = Field(None, description="Price of the item mentioned.")

def extract_filters(user_query: str, openai_client) -> dict:
    """Uses GPT to parse the user query into structured DB filters."""

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a database query generator. Extract metadata filters from the user's request based on the schema provided. If a filter is not mentioned, leave it as null."
            },
            {"role": "user", "content": user_query}
        ],
        response_format=DBFilter,
    )

    # This guarantees a perfectly parsed Pydantic object matching your schema
    extracted = completion.choices[0].message.parsed
    return extracted.model_dump(exclude_none=True)

def run_openai_rag():
    # 1. Initialize the OpenAI Client
    # (It automatically reads the OPENAI_API_KEY environment variable)
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # 2. Initialize Local ChromaDB (Ephemeral/In-Memory for this example)
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection(name="local_knowledge_base")

    # 3. Your Private Local Data (Simulating text extracted from your files)
    # Upgrade your data structure to include metadata dictionaries
    local_documents = [
        {"text": "Adidas mens shoes model: A100, price=100", "metadata": {"brand": "Adidas", "on_sale": False, "price": 100}},
        {"text": "Nike mens shoes model: N200, price=200", "metadata": {"brand": "Nike", "on_sale": True, "price": 150}},
        {"text": "Puma mens shoes model: P300, price=300", "metadata": {"brand": "Puma", "on_sale": True, "price": 120}},
        {"text": "Adidas womens shoes model: A400, price=400", "metadata": {"brand": "Adidas", "on_sale": True, "price": 110}},
    ]

    # 4. Ingest and Embed Data using OpenAI
    print("Embedding and storing local documents...")
    for i, doc in enumerate(local_documents):
         # Generate semantic vector using OpenAI's embedding model
         embedding_response = openai_client.embeddings.create(
             model="text-embedding-3-small",
             input=doc["text"]
         )
         vector = embedding_response.data[0].embedding

         # Store the text along with its mathematical vector representation
         collection.add(
             ids=[f"doc_{i}"],
             embeddings=[vector],
             metadatas=[doc["metadata"]], ## This will pass metadata to the vector database
             documents=[doc["text"]]
         )

    # 5. The User Asks a Question
    user_query = "Show me all Adidas shoes on sale and their prices?"
    print(f"\nUser Query: {user_query}")

    # Create structured filters from the user query using the LLM --<<
    # USE LLM to convert human language into structured metadata filters that match your DB schema
    extracted_metadata = extract_filters(user_query, openai_client)
    print(f"Extracted Filters from LLM : {extracted_metadata}")

    # 6. Convert the Query into a Vector to Search the DB
    query_embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=user_query
    )
    query_vector = query_embedding_response.data[0].embedding

    # Construct ChromaDB's where clause dynamically
    chroma_where_clause = {}
    if extracted_metadata:
        if len(extracted_metadata) > 1:
            # If multiple filters exist, wrap them in an $and block
            chroma_where_clause = {
                "$and": [{key: {"$eq": value}} for key, value in extracted_metadata.items()]
            }
        else:
            # If only one filter exists
            key, value = list(extracted_metadata.items())[0]
            chroma_where_clause = {key: {"$eq": value}}

    # 7. Query the Vector DB for the closest match
    search_results = collection.query(
        query_embeddings=[query_vector],
        n_results=3,
        where=chroma_where_clause if chroma_where_clause else None
    )
    # Store all 3 search results in the retrieved context
    retrieved_context = "\n".join(search_results['documents'][0])
    print(f"Retrieved Context from DB:\n{retrieved_context}")

    # 8. Hand the Context and the Query to OpenAI's LLM
    ## Because you are using text-embedding-3-small, the vector database looks for semantic similarity.
    ## The words "Adidas", "shoes", and "prices" are incredibly strong matches.
    ## Your vector DB is almost certainly going to return the A100 model
    ## because it looks like the right kind of data, completely ignoring the fact that sale=no directly violates the user's intent.
    ## You tried to fix this using the System Prompt (CRITICAL: Only list specific shoe models...), forcing the LLM to do the heavy filtering.
    ## But relying on the LLM to clean up bad database results is risky and wastes tokens.

    print("\nGenerating answer with GPT-4o-mini...")
    completion = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": ("You are a helpful assistant."
                                       "Answer the user's question using ONLY the provided context. ")},
        {"role": "user", "content": f"Context: {retrieved_context}\n\nQuestion: {user_query}"}
    ],
    temperature=0.2
    )

    print(f"\nAI Assistant: {completion.choices[0].message.content}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
    else:
        run_openai_rag()