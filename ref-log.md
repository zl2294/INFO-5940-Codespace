## Similarity Retrieval Design Choice
Instead of manually calling a cosine similarity function or using the older `similarity_search()` API, this application uses:
# cosine similarity search via vector retriever
retrieved = vs.as_retriever(search_kwargs={"k": 4}).invoke(user_q)

I chose to use the `as_retriever()` interface because it already performs cosine-similarity ranking internally without requiring additional manual computation. It is also more flexible since it supports advanced retrieval strategies such as MMR and metadata filtering. This method is currently recommended by LangChain for RAG implementations, and it keeps my code cleaner and easier to extend later if I want to upgrade retrieval logic.

it works
1. When documents are uploaded, each chunk gets converted into a vector embedding.
2. When I ask a question, the query is also embedded into the same vector space.
3. ChromaDB compares the query vector to all stored vectors using cosine similarity.
4. It selects the top-k most relevant chunks (k=4 in my system).
5. Those chunks are passed to the LLM so the answer is grounded only in the documents.

To support my retrieval process in this project, I added:
chromadb: stores document embeddings locally so the system can perform semantic search
langchain-chroma: allows LangChain to connect with Chroma for vector retrieval
These two libraries enable the document chunks to be retrieved based on similarity when a user asks a question. I used GenAI (ChatGPT) mainly to confirm the correct versions and compatibility, so everything can run smoothly inside the provided Codespace environment.