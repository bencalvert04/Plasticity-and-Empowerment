import unittest
from unittest.mock import patch
from sampled_a_normal150_experiment import contexts,generate_trajectory,windows,summaries,INSTRUCTION
from tests.test_structured_calculation import CharacterTokenizer
from tests.test_sampled_rlvr import Model,Tokenizer
from sampled_rlvr_experiment import continuation_prefix
class Tests(unittest.TestCase):
 def test_exact_original_correction_context(self):
  t=CharacterTokenizer();q=dict(question='Q',injection='I meant 30 apples, not 20.');a=[65]*20
  self.assertEqual(INSTRUCTION,'')
  self.assertEqual(contexts(t,q,a,'native'),[continuation_prefix(t,'Q',m,s,'native') for s,m in [(a,q['injection']),(a,''),([],q['injection'])]])
 def test_natural_a_and_150_cap(self):
  a=dict(token_ids=[5]*20,content_token_count=20,ended_at_eos=False)
  with patch('sampled_a_normal150_experiment.generate_b',side_effect=[a,{}]) as g:
   generate_trajectory(Model(),Tokenizer(),dict(question='Q',injection='J'),'native',{0},12)
  self.assertEqual(g.call_args_list[0].args[-1],20);self.assertEqual(g.call_args_list[1].args[-1],150)
 def test_all_primary_windows_exclude_early_eos(self):
  rows=[]
  for n in [75,150]:
   ids=[5]*n+([0] if n<150 else [])
   b=dict(token_ids=ids,entropies=[[0,1,2]]*len(ids))
   rows.append(dict(id='q',rlvr_step=0,a_complete=True,b_attempted=True,b_ended_at_eos=n<150,included_primary=n==150,**windows(b,{0})))
  s=summaries(rows)[0]
  self.assertEqual(s['early_1_50_available_n'],2)
  for name in ['early_1_50','middle_51_100','late_101_150']:
   self.assertEqual(s[name+'_n'],1);self.assertEqual(s[name+'_plasticity_bits_per_token_mean'],1)
  self.assertIsNone(summaries(rows[:1])[0]['early_1_50_plasticity_bits_per_token_mean'])
if __name__=='__main__':unittest.main()
