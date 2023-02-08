import sys
sys.path.insert(0, './')
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher, Indexer
from colbert.data import Queries, Collection
import torch
from sentence_transformers import SentenceTransformer
import os
import numpy as np
#CUDA_LAUNCH_BLOCKING=1




class Maxsime:
    dataroot = 'downloads/lotte'
    dataset = 'pooled'
    datasplit = 'dev'    
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300   # truncate passages at 300 tokens
    checkpoint = 'downloads/colbertv2.0'
    index_name = f'{dataset}.{datasplit}.{nbits}bits'
    collection = Collection(path=os.path.join(dataroot, dataset, datasplit, 'collection.tsv'))



    def __init__(self, model ="colbert"):
        if model !="colbert":
            self.type = "bert"
            self.model = SentenceTransformer(model)
        else:            
            self.type = "colbert"
            config = ColBERTConfig(doc_maxlen=self.doc_maxlen, nbits=self.nbits)
            #indexer = Indexer(checkpoint=self.checkpoint, config=config)
            #indexer.index(name=self.index_name, collection=self.collection, overwrite=True)
            with Run().context(RunConfig(experiment='maxsime')):
                self.searcher = Searcher(index=self.index_name)


    def get_tokens(self,sent):
        if self.type == "colbert":
            return self.searcher.checkpoint.doc_tokenizer.tokenize([sent], add_special_tokens=True)[0]
        else:
            return self.model.tokenizer.convert_ids_to_tokens(self.model.tokenizer.encode(sent))
        
    def get_embeddings(self,sent):
        if self.type == "colbert":
            return self.searcher.checkpoint.queryFromText([sent])
        else:
            return self.model.encode(sent, output_value="token_embeddings")
    
    def get_scores(self,Q,D):        
        if self.type == "colbert":
            return torch.mm(Q[0].float(),D[0].permute(1,0).float())
        else:
            return torch.nn.functional.normalize(Q).mm(torch.nn.functional.normalize(D).permute(1,0))

    
    def explain_match(self,query, doc_text):        
        query_tokens = self.get_tokens(query)
        Q = self.get_embeddings(query)        
        doc_tokens = self.get_tokens(doc_text)        
        D = self.get_embeddings(doc_text)        
        scores = self.get_scores(Q,D)
        try:
            matches = [(query_tokens[i],doc_tokens[scores.argmax(1)[i]], scores[i,scores.argmax(1)[i]].item()) for i in range(len(query_tokens)) if query_tokens[i] !="[MASK]"]
        except:
            return
        sorted_matches = sorted(matches, key=lambda tup: tup[-1], reverse=True)                
        explained_text = doc_tokens
        for i in range(len(query_tokens)) :
            if query_tokens[i] !="[MASK]":
                explained_text[scores.argmax(1)[i]] += f"[={query_tokens[i]}]"        
        return sorted_matches, " ".join(explained_text)
    
    def evaluate(self, matches1, matches2):
        if matches1 is None:
            return None
        if matches2 is None:
            return 0,0
        target = set([t[1] for t in matches1[0] if t[0] != "[D]"])        
        y = set([t[1] for t in matches2[0]])        
        word_acc = len(y & target) / len(target)       
        target = set([(t[0],t[1]) for t in matches1[0] if t[0] != "[D]"])
        y = set([(t[0],t[1]) for t in matches2[0]])        
        pair_acc = len(y & target) / len(target)        
        return word_acc, pair_acc

if __name__ == '__main__':    

    query = "Why do kittens like packets?"
    doc ="Cats enjoy boxes because they love hiding places. When they are inside a box they are covered on all sides but one. Which means they are safe and can keep an eye out on the one open side. Boxes also allow for the cats to quickly dart from the box if something of interest appears, and allows for a quick retreat if necessary."
    mxsm1 = Maxsime(model="colbert")
    mxsm2 = Maxsime(model="bert-base-uncased")

    results1 = mxsm1.explain_match(query,doc)
    results2 = mxsm2.explain_match(query,doc)
    print(mxsm1.evaluate(results1, results2))
    
    
    dataroot = 'downloads/lotte'
    #dataset = 'lifestyle'
    dataset = 'pooled'
    datasplit = 'dev'

    queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
    collection = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

    queries = Queries(path=queries)
    
    docs = []
    for q in queries:
        # Find the top-3 passages for this query
        results = mxsm1.searcher.search(q[1], k=1)
        for passage_id, passage_rank, passage_score in zip(*results):
            docs.append(mxsm1.searcher.collection[passage_id])
            break


    
    #print(mxsm1.evaluate(mxsm1.explain_match(queries[0],docs[0]), mxsm2.explain_match(queries[0],docs[0])))
    eval_results = [mxsm1.evaluate(mxsm1.explain_match(q[1],d), mxsm2.explain_match(q[1],d)) for q,d in zip(queries,docs)]

    a = np.array([t for t in eval_results if t!= None])
    print(a.shape)
    print(np.mean(a, axis=0))
    # Print out the top-k retrieved passages
    exit()
    
        

    

    query = "how much should i feed my 1 year old english mastiff?"
    doc = "I have a 2 1/2 year old bull mastiff. I have been feeding him Blue Buffalo since I got him at 8 weeks old. He is very lean and active for a bull mastiff. I feed him about 3-4 cups twice a day which averages about 130.00 a month. It is very important that you can afford this breed. I just had to take mine to the vet because he developed some sort of allergies on his skin, eyes and ears and the vet bill was $210.00 with all his medication. This wasn't an option I had to take him an get all his meds or he would have gotten worse. They're just like your children, you can expect things to come up and you need to be able to care for them."
    print(mxsm1.explain_match(query,doc))
    print(mxsm2.explain_match(query,doc))    
    query = "are zebra loaches safe with shrimp?"
    doc = "Amano shrimp are good tank mates for community fish. They'll ignore your fish altogether. And they eat algae 24x7, which never hurts. Amano shrimp require brackish water for breeding, so won't breed in most tanks. This also makes them difficult to find. Cherry shrimp (and their color varieties) will also be no threat to your fish. But, they are very small, so aggressive fish (barbs, for example) may go after them. They breed quite easily and rapidly, so if you want more of them make sure you have plenty of hiding places for the young shrimp. Cherry shrimp are also algae eaters, though being so small you'll need huge quantities of them to have an real impact if algae controll is a goal. Really, for most shrimp in tanks, the issue isn't if they are a danger to the fish, but if the fish will be a danger to the shrimps. Even larger shrimp may find their extremities and tails the target of nipping. The tetras shouldn't be a problem for the shrimp, but the loaches may go after them. Edit: One heads up about shrimp (and most aquarium invertebrates, actually). A lot of medications and chemicals you might use in a tank are poisonous to them. So once they are in there, you'll need to be extra careful with what you put into the tank, and check that it is safe for shrimp."
    print(mxsm1.explain_match(query,doc))
    print(mxsm2.explain_match(query,doc))