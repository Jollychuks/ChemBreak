from chembreak20.targets import HuggingFaceTarget

class FakeTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        return 'CHAT:'+'|'.join(f"{m['role']}={m['content']}" for m in messages)+':ASSISTANT'

def test_chemdfm_round_format():
    t=HuggingFaceTarget({'id':'ChemDFM','template':'round_chat'}); t.tokenizer=FakeTokenizer()
    text=t._format('next',[{'role':'user','content':'u1'},{'role':'assistant','content':'a1'}])
    assert '[Round 0]\nHuman: u1\nAssistant: a1' in text and '[Round 1]\nHuman: next\nAssistant:' in text

def test_chemllm_uses_tokenizer_native_chat_template():
    t=HuggingFaceTarget({'id':'ChemLLM','template':'tokenizer_chat_template'}); t.tokenizer=FakeTokenizer()
    text=t._format('next',[{'role':'user','content':'u1'},{'role':'assistant','content':'a1'}])
    assert text=='CHAT:user=u1|assistant=a1|user=next:ASSISTANT'
