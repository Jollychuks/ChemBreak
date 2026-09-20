from chembreak22.targets import HuggingFaceTarget
class Tok:
    eos_token_id=2; pad_token_id=2
    def __call__(self,*a,**k):return {'input_ids':[1,2]}
    def encode(self,*a,**k):return [1]
    def decode(self,*a,**k):return 'x'

def test_chemdfm_round_format():
    t=HuggingFaceTarget({'id':'ChemDFM','template':'round_chat'}); t.tokenizer=Tok(); s=t._format('new',[{'role':'user','content':'u'},{'role':'assistant','content':'a'}]); assert '[Round 0]' in s and 'Human: new' in s
