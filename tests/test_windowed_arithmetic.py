import unittest
from windowed_arithmetic_experiment import windows,contexts,INSTRUCTION
from tests.test_structured_calculation import CharacterTokenizer
class Tests(unittest.TestCase):
    def test_windows_and_early_eos(self):
        for n in [0,24,25,49,50]:
            ids=[5]*n+([0] if n<50 else [])
            entropy=[[1,1+i,1+2*i] for i in range(len(ids))]
            w=windows(dict(token_ids=ids,entropies=entropy),{0})
            self.assertEqual(w['early_1_25_complete'],n>=25)
            self.assertEqual(w['later_26_50_complete'],n>=50)
            if n>=25:self.assertEqual(w['early_1_25_plasticity_bits_per_token'],12)
            else:self.assertIsNone(w['early_1_25_plasticity_bits_per_token'])
            if n==50:
                self.assertEqual(w['later_26_50_plasticity_bits_per_token'],37)
                self.assertEqual(w['full_1_50_plasticity_bits_per_token'],24.5)
            else:self.assertIsNone(w['full_1_50_plasticity_bits_per_token'])
    def test_instruction_kept(self):
        t=CharacterTokenizer();q=dict(question='QUESTION',injection='CORRECTION')
        texts=[t.decode(x) for x in contexts(t,q,list(map(ord,'FIXEDA')),'native')]
        self.assertTrue(all(x.count(INSTRUCTION)==1 for x in texts))
        self.assertNotIn('CORRECTION',texts[1]);self.assertNotIn('FIXEDA',texts[2])
if __name__=='__main__':unittest.main()
