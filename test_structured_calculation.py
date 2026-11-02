import unittest
import torch
from test_sampled_rlvr import Model
from structured_calculation_experiment import score_regions, b_contexts, generate_b, FORMAT

class CharacterTokenizer:
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids if i)
    def apply_chat_template(self,messages,**kwargs):
        text=''.join('<'+m['role']+'>'+m['content']+'</end>' for m in messages)+'<assistant>'
        return list(map(ord,text))

class Tests(unittest.TestCase):
    def span(self,text,done=True):
        ids=list(map(ord,text))+([0] if done else [])
        return dict(token_ids=ids,entropies=[[1.,3.,4.] for _ in ids],ended_at_eos=done)
    def test_regions_labels_excluded_numeric_grading(self):
        text='Calculation: 720 / (4 + 2) / 60 = 2\nAnswer: 2'
        r=score_regions(CharacterTokenizer(),self.span(text),{0},'2')
        self.assertEqual(r['parse_status'],'ok');self.assertTrue(r['answer_correct'])
        self.assertEqual(''.join(text[i] for i in r['answer_token_indices']),'2')
        self.assertEqual(r['answer_token_count'],1)
        self.assertEqual(r['calculation_plasticity_bits_per_token'],2)
        self.assertEqual(r['answer_empowerment_bits_per_token'],3)
        self.assertTrue(all(i>=len('Calculation: ') for i in r['calculation_token_indices']))
        self.assertEqual(r['first20_plasticity_bits_per_token'],2)
    def test_truncation_format_and_source_issue(self):
        for text,done,status in [('Calculation: 1+1=2\nAnswer: 2',False,'truncated'),
                                 ('The result is 2',True,'format_mismatch'),
                                 ('Calculation: 1+1\nAnswer: 2\nAnswer: 3',True,'format_mismatch')]:
            r=score_regions(CharacterTokenizer(),self.span(text,done),{0},'2')
            self.assertEqual(r['parse_status'],status)
            self.assertIsNone(r['calculation_plasticity_bits_per_token']);self.assertIsNone(r['answer_correct'])
        r=score_regions(CharacterTokenizer(),self.span('Calculation: 1+1=2\nAnswer: 2'),{0},'2',True)
        self.assertEqual(r['grading_status'],'source_issue_excluded');self.assertIsNone(r['answer_correct'])
        self.assertIsNotNone(r['answer_plasticity_bits_per_token'])
    def test_numeric_formats_and_wrong_answer(self):
        for answer,expected,correct in [('1,200','1200',True),('1/2','0.5',True),('3','2',False)]:
            r=score_regions(CharacterTokenizer(),self.span('Calculation: 1+1\nAnswer: '+answer),{0},expected)
            self.assertEqual(r['answer_correct'],correct)
    def test_format_instruction_retained_in_all_contexts(self):
        t=CharacterTokenizer();a=list(map(ord,'SAVED_A'))
        contexts=b_contexts(t,'QUESTION',a,'CORRECTION','native')
        texts=[t.decode(ids) for ids in contexts]
        self.assertTrue(all(x.count(FORMAT)==1 for x in texts))
        self.assertIn('CORRECTION',texts[0]);self.assertNotIn('CORRECTION',texts[1]);self.assertIn('CORRECTION',texts[2])
        self.assertIn('SAVED_A',texts[0]);self.assertIn('SAVED_A',texts[1]);self.assertNotIn('SAVED_A',texts[2])
    def test_variable_b_cap_and_eos_shared_history(self):
        model=Model();prefixes=[[1,3,4,2]]*3
        b=generate_b(model,prefixes,{0},12,7)
        self.assertEqual(b['token_ids'],[5]*7);self.assertFalse(b['ended_at_eos'])
        for p in range(7):
            for k in range(3):self.assertEqual(model.calls[3*p+k],prefixes[k]+[5]*p)
        b=generate_b(Model('a'),prefixes,{0},12,256)
        self.assertEqual(b['token_ids'],[0]);self.assertTrue(b['ended_at_eos'])
        with self.assertRaises(ValueError):generate_b(Model(),prefixes,{0},12,0)

if __name__=='__main__':unittest.main()
