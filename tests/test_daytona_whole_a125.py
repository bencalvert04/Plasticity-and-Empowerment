import unittest
from unittest.mock import patch
import tempfile,json
from pathlib import Path
from daytona_for_ben.controller import verify
from daytona_for_ben.run_shard import contexts,eligible,POSITIONS
from structured_calculation_experiment import region_summary
class WholeA125Tests(unittest.TestCase):
 def test_source_ablation(self):
  a=list(range(125));q={'question':'Q','injection':'J'}
  with patch('daytona_for_ben.run_shard.continuation_prefix',side_effect=lambda t,q,j,a,m:(j,a)):
   full,empty_j,without_a=contexts(None,q,a,'m')
  self.assertEqual(full,('J',a));self.assertEqual(empty_j,('',a));self.assertEqual(without_a,('J',[]));self.assertEqual(len(a)-len(without_a[1]),125)
 def test_verifier_rejects_old_late_a_protocol(self):
  with tempfile.TemporaryDirectory() as folder:
   dest=Path(folder)
   (dest/'config.json').write_text(json.dumps({'protocol':'late_a75_125_b75_125_v1'}))
   (dest/'rollouts.jsonl').write_text('')
   with self.assertRaises(AssertionError):verify('step0_shard0',dest)
 def test_cap_without_eos_is_retained(self):self.assertTrue(eligible({'content_token_count':125,'ended_at_eos':False}))
 def test_eos_at_last_generation_position_is_short(self):self.assertFalse(eligible({'content_token_count':124,'ended_at_eos':True}))
 def test_window(self):
  self.assertEqual(POSITIONS,list(range(74,125)))
  es=[[0,99,99]]*74+[[1,3,4]]*51
  x=region_summary(es,POSITIONS)
  self.assertEqual(x['plasticity_bits_per_token'],2);self.assertEqual(x['empowerment_bits_per_token'],3)
 def test_shards(self):
  parts=[{i for i in range(50) if i%3==s} for s in range(3)]
  self.assertEqual([len(x)*10 for x in parts],[170,170,160]);self.assertEqual(len(set.union(*parts)),50);self.assertFalse(parts[0]&parts[1])
if __name__=='__main__':unittest.main()
