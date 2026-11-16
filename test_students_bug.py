from app_students.rag import initialize_retriever, search
from app_students.llm import generate_answer
q = "Πώς μπορώ να εκδώσω πιστοποιητικό φοίτησης;"
initialize_retriever()
items = search(q, n_results=3)

for i, item in enumerate(items):
    print(i, item["document_title"], len(item["text"]), repr(generate_answer(q, [item])))


print('NEW PRINT')
print(repr(generate_answer(q, items[:2])))