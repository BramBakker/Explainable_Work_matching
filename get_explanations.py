import pickle
from test_bilrp_func import get_explanation
from sentence_transformers import SentenceTransformer
from real_bilrp import E5ModelXAI
model = E5ModelXAI(r'new_forty_matcher')

with open("realpairs.txt") as texts:
    line_status=0
    pairs=[]
    for line in texts:
        line_status+=1
        if line_status==1:
            pair=[line.strip("\n")]
        if line_status==2:
            pair.append(line.strip("\n"))
        if line_status==3:
            line_status=0
            pairs.append(pair)

number=18
for p in pairs:
    print(p)
    number+=1

    name="explanations/BiLRP_newnewpair_"+str(number)
    get_explanation(p[0],p[1],model,name)


