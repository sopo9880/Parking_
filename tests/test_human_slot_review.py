import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from human_slot_review import ReviewSession, ReviewApp, raw_point, sha
from human_manual_replay import verified_snapshot
from manual_initialization import load_snapshot, write_snapshot


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.slots=[{'global_id':'G018','local_id':'A','cctv':'cctv1','point':[10,20]}, {'global_id':'G001','local_id':'B','cctv':'cctv1','point':[30,40]}]
        for name in ('video','slots','rois'): (self.root/name).write_text(name)
        self.pseudo=self.root/'pseudo.csv';write_snapshot(self.pseudo,900,{'G018':'OCCUPIED','G001':'UNKNOWN'})
    def tearDown(self): self.tmp.cleanup()
    def session(self):
        return ReviewSession(self.root/'review',900,self.slots,self.root/'video',self.root/'slots',self.root/'rois',self.pseudo)
    def test_ai_never_becomes_human_without_selection(self):
        s=self.session();self.assertEqual(s.choices,{})
        with self.assertRaises(ValueError):s.publish()
        s.select('G018','UNKNOWN'); self.assertEqual(self.session().choices,{'G018':'UNKNOWN'})
        with self.assertRaises(ValueError):s.publish()
        s.select('G001','EMPTY');out=s.publish()
        self.assertEqual(load_snapshot(out,900,['G018','G001']),{'G018':'UNKNOWN','G001':'EMPTY'})
        audit=json.loads(out.with_suffix('.audit.json').read_text(encoding='utf8'))
        self.assertEqual(audit['snapshot_sha256'],sha(out));self.assertEqual(audit['unknown_count'],1)
    def test_changed_source_blocks_resume(self):
        s=self.session();s.select('G018','EMPTY');(self.root/'video').write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'):self.session()
    def test_inverse_roi_mapping(self):
        roi={'mode':'perspective','points':[[100,100],[200,100],[200,200],[100,200]],'output_size':[101,101]}
        self.assertEqual(raw_point({'point':[50,50]},roi),(150,150))
    def test_replay_requires_complete_matching_audit_and_unchanged_snapshot(self):
        session=self.session();session.select('G018','UNKNOWN');session.select('G001','EMPTY');snapshot=session.publish()
        fit=self.root/'fit';fit.mkdir();(fit/'slots.json').write_text('slots');(fit/'worker_input.json').write_text('rois')
        manifest={'video_sha256':sha(self.root/'video'),'snapshot_sha256':sha(self.pseudo)}
        self.assertEqual(verified_snapshot(snapshot,900,{'G018','G001'},manifest,fit)['G018'],'UNKNOWN')
        snapshot.write_text('time_sec,global_slot_id,initial_state\n900,G018,OCCUPIED\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):verified_snapshot(snapshot,900,{'G018','G001'},manifest,fit)
    def test_ui_choices_are_explicit_and_auto_advance(self):
        import tkinter as tk
        slotfile=self.root/'slots.json';slotfile.write_text(json.dumps({'slots':self.slots}))
        roifile=self.root/'roi.json';roifile.write_text(json.dumps({'rois':{'cctv1':[0,0,80,80]}}))
        other=self.root/'pseudo0.csv';write_snapshot(other,0,{'G018':'EMPTY','G001':'OCCUPIED'})
        config={'video':str(self.root/'video'),'review_dir':str(self.root/'review'),
                'times':{'900':{'slots':str(slotfile),'roi_source':str(roifile),'pseudo':str(self.pseudo)},
                         '0':{'slots':str(slotfile),'roi_source':str(roifile),'pseudo':str(other)}}}
        class Cap:
            def release(self):pass
        win=tk.Tk();win.withdraw()
        try:
            with patch('human_slot_review.open_video',return_value=Cap()),patch('human_slot_review.read_frame_at',return_value=np.zeros((80,80,3),dtype='uint8')):
                app=ReviewApp(win,config);win.update()
                self.assertEqual(app.tree.selection()[0],'G018');self.assertEqual(app.session.choices,{})
                app.choose('UNKNOWN');win.update()
                self.assertEqual(app.session.choices,{'G018':'UNKNOWN'});self.assertEqual(app.tree.selection()[0],'G001')
                app.time.set('0:00');app.load_time();win.update()
                self.assertEqual(app.session.start,0);self.assertEqual(app.session.choices,{})
                app.time.set('15:00');app.load_time();win.update()
                self.assertEqual(app.session.choices,{'G018':'UNKNOWN'})
        finally:win.destroy()


if __name__=='__main__':unittest.main()
