import unittest
from unittest.mock import patch
from sampled_a_arithmetic150_experiment import windows,summaries,generate_trajectory
from test_sampled_rlvr import Model,Tokenizer
class Tests(unittest.TestCase):
 def test_windows_and_eos(self):
  for n in [0,49,50,99,100,149,150]:
   ids=[5]*n+([0] if n<150 else [])
   e=[[0,i,2*i] for i in range(len(ids))];w=windows(dict(token_ids=ids,entropies=e),{0})
   self.assertEqual(w['early_1_50_complete'],n>=50)
   self.assertEqual(w['middle_51_100_complete'],n>=100)
   self.assertEqual(w['late_101_150_complete'],n>=150)
   if n==150:
    self.assertEqual(w['early_1_50_plasticity_bits_per_token'],24.5)
    self.assertEqual(w['middle_51_100_plasticity_bits_per_token'],74.5)
    self.assertEqual(w['late_101_150_plasticity_bits_per_token'],124.5)
 def test_eos_excluded_from_all_primary_windows(self):
  rows=[]
  for n in [75,150]:
   b=dict(token_ids=[5]*n+([0] if n<150 else []),entropies=[[0,1,2]]*(n+int(n<150)))
   rows.append(dict(id='q',rlvr_step=0,a_complete=True,b_attempted=True,b_ended_at_eos=n<150,included_primary=n==150,**windows(b,{0})))
  s=summaries(rows)[0]
  self.assertEqual(s['early_1_50_available_n'],2)
  for name in ['early_1_50','middle_51_100','late_101_150']:
   self.assertEqual(s[name+'_n'],1);self.assertEqual(s[name+'_plasticity_bits_per_token_mean'],1)
  s=summaries(rows[:1])[0]
  self.assertIsNone(s['early_1_50_plasticity_bits_per_token_mean'])
 def test_generation_caps(self):
  a=dict(token_ids=[5]*20,content_token_count=20,ended_at_eos=False)
  with patch('sampled_a_arithmetic150_experiment.generate_b',side_effect=[a,{}]) as g:
   generate_trajectory(Model(),Tokenizer(),dict(question='Q',injection='J'),'native',{0},12)
  self.assertEqual(g.call_args_list[0].args[-1],20);self.assertEqual(g.call_args_list[1].args[-1],150)
if __name__=='__main__':unittest.main()
