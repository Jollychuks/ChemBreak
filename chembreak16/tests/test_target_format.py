from chembreak16.targets import HuggingFaceTarget

def test_chemdfm_round_formatting():
    target=HuggingFaceTarget({'id':'ChemDFM'})
    text=target._format('third',[{'role':'user','content':'first'},{'role':'assistant','content':'second'}])
    assert text=='[Round 0]\nHuman: first\nAssistant: second\n[Round 1]\nHuman: third\nAssistant:'
