from sentence_transformers import SentenceTransformer, util
import pickle
model = SentenceTransformer('newest_forty_matcher')
#model_2 = SentenceTransformer('twenty_matcher')
input = pickle.load(open("lines_unseen.p", "rb"))
i=0

embeddings={}
for id in input.keys():
    text = input[id]
    i+=1 
    entity_1 = f'Query: {text}'

    emb1 = model.encode(entity_1, convert_to_tensor=True)
    embeddings[id]=emb1
    if i % 1000 == 0:
        print(f"Processed {i} items...")

pickle.dump(embeddings, open("embeddings_unseen_newest.p", "wb"))