import unittest
from unittest.mock import patch
from sampled_a_arithmetic_experiment import generate_trajectory
from test_sampled_rlvr import Model,Tokenizer
class Tests(unittest.TestCase):
    def test_fresh_a_and_b_with_independent_seeds(self):
        a=dict(token_ids=[5]*20,content_token_count=20,ended_at_eos=False)
        b=dict(token_ids=[5]*50,content_token_count=50,ended_at_eos=False)
        with patch('sampled_a_arithmetic_experiment.generate_b',side_effect=[a,b]) as gen:
            result=generate_trajectory(Model(),Tokenizer(),dict(question='Q',injection='J'),'native',{0},42)
        self.assertEqual(result,(a,b,1000000000042,True))
        self.assertEqual(gen.call_args_list[0].args[-2:],(1000000000042,20))
        self.assertEqual(gen.call_args_list[1].args[-2:],(42,50))
        self.assertEqual(len(gen.call_args_list[0].args[1]),1)
        self.assertEqual(len(gen.call_args_list[1].args[1]),3)
    def test_early_a_skips_injection(self):
        a=dict(token_ids=[5,0],content_token_count=1,ended_at_eos=True)
        with patch('sampled_a_arithmetic_experiment.generate_b',return_value=a) as gen:
            _,b,_,complete=generate_trajectory(Model(),Tokenizer(),dict(question='Q',injection='J'),'native',{0},42)
        self.assertFalse(complete);self.assertEqual(b['token_ids'],[]);self.assertEqual(gen.call_count,1)
if __name__=='__main__':unittest.main()
