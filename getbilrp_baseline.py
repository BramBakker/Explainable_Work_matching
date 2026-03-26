import pickle
from test_bilrp_func import get_explanation, get_baseline_explanation
from sentence_transformers import SentenceTransformer
from real_bilrp import E5ModelXAI
model = SentenceTransformer(r'new_forty_matcher')

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

number=0
for p in pairs:
    number+=1
    name="explanations/BiLRP_baseline_"+str(number)
    get_baseline_explanation(p[0],p[1],model,name)
