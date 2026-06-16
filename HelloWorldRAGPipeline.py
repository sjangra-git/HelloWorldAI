import os
from openai import OpenAI
import chromadb

def run_openai_rag():
    # 1. Initialize the OpenAI Client
    # (It automatically reads the OPENAI_API_KEY environment variable)
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # 2. Initialize Local ChromaDB (Ephemeral/In-Memory for this example)
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection(name="local_knowledge_base")

    # 3. Your Private Local Data (Simulating text extracted from your files)
    local_documents = [
        "Adidas mens shoes model: A100, price: $100, quantity: 2, color=blue, sale=no",
        "Nike mens shoes model: N200, price: $150, quantity: 1, color=black, sale=10% off",
        "Puma mens shoes model: P300, price: $120, quantity: 3, color=white, sale=15% off",
        "Adidas womens shoes model: A400, price: $110, quantity: 2, color=pink, sale=10% off",
        "Adidas summer sale 20% off on all shoes until August 31st, 2026"
    ]

    # 4. Ingest and Embed Data using OpenAI
    print("Embedding and storing local documents...")
    for i, doc in enumerate(local_documents):
        # Generate semantic vector using OpenAI's embedding model
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=doc
        )
        vector = embedding_response.data[0].embedding

        # Store the text along with its mathematical vector representation
        collection.add(
            ids=[f"doc_{i}"],
            embeddings=[vector],
            documents=[doc]
        )

    # 5. The User Asks a Question
    user_query = "Show me all Adidas shoes on sale and their prices?"
    print(f"\nUser Query: {user_query}")

    # 6. Convert the Query into a Vector to Search the DB
    query_embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=user_query
    )
    query_vector = query_embedding_response.data[0].embedding

    # 7. Query the Vector DB for the closest match
    search_results = collection.query(
        query_embeddings=[query_vector],
        n_results=3 # Fetch only the single most relevant paragraph
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

       You tried to fix this using the System Prompt (CRITICAL: Only list specific shoe models...), forcing the LLM to do the heavy filtering. But relying on the LLM to clean up bad database results is risky and wastes tokens.
    print("\nGenerating answer with GPT-4o-mini...")
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ("You are a helpful assistant."
            "Answer the user's question using ONLY the provided context. "
            "CRITICAL: Only list specific shoe models. If a piece of context is a general sale announcement ignore it ")},
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