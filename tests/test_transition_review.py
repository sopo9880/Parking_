import json
import tempfile
import unittest
from pathlib import Path
from human_slot_review import sha
from transition_review_ui import EventReviews


class EventReviewTests(unittest.TestCase):
    def test_only_user_choices_become_human_events_and_resume_checks_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for n in ('video','slots','roi'):(p/n).write_text(n)
            data={'video_sha256':sha(p/'video'),'fit_slots_sha256':sha(p/'slots'),'roi_source_sha256':sha(p/'roi'),
                  'cases':[{'case_id':'A'},{'case_id':'B'}]}
            case=p/'cases.json';case.write_text(json.dumps(data));out=p/'reviews.json'
            reviews=EventReviews(out,case,p/'video',p/'slots',p/'roi');self.assertEqual(reviews.choices,{})
            reviews.choose('A','OCCLUSION');saved=json.loads(out.read_text(encoding='utf8'))
            self.assertEqual(saved['status'],'partial');self.assertEqual(list(saved['reviews']),['A'])
            self.assertEqual(saved['reviews']['A']['source'],'operator_review')
            self.assertEqual(EventReviews(out,case,p/'video',p/'slots',p/'roi').choices['A']['label'],'OCCLUSION')
            (p/'video').write_text('changed')
            with self.assertRaises(ValueError):EventReviews(out,case,p/'video',p/'slots',p/'roi')


if __name__=='__main__':unittest.main()
