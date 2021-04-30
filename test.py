from modules import Embedding

e = Embedding(2,5,False)

print(e.embedding_table.weight.data)
