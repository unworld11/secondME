from query_rag import query_documents

query = "Who am i"
engine = query_documents(query)

print(engine)