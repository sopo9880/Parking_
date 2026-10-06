import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from localization import CATALOG, tr, set_language, refresh_tree
from update_helper import _merge_settings_defaults


class TranslationTests(unittest.TestCase):
    def tearDown(self):
        set_language('ko')

    def test_catalog_and_protected_identifiers(self):
        for language in ('ko','en'):
            set_language(language)
            canonical='SAFE_BASELINE CANDIDATE OCCUPIED EMPTY MANEUVERING LEAVING UNKNOWN'
            self.assertEqual(tr(canonical),canonical)
            files='Video_Profile.csv settings.json candidate_v164_metrics.csv UPLOAD_TO_CHATGPT.zip'
            self.assertEqual(tr(files),files)
            path=r'C:\Video\Profile\Saved.csv'
            self.assertEqual(tr('Saved to:\n'+path).split('\n')[-1],path)
        set_language('ko')
        for english,korean in CATALOG.items():
            self.assertEqual(tr(english),korean)
        self.assertEqual(set_language('bad-locale'),'ko')

    def test_update_merge_keeps_preference(self):
        defaults={'ui':{'language':'ko'},'validation':{'camera_count':3}}
        current={'ui':{'language':'en'},'validation':{'camera_count':5}}
        self.assertEqual(_merge_settings_defaults(defaults,current),current)
        self.assertEqual(_merge_settings_defaults(defaults,{})['ui']['language'],'ko')

    def test_language_does_not_invalidate_research_cache(self):
        settings=json.loads(app.SETTINGS_PATH.read_text(encoding='utf-8'))
        other=copy.deepcopy(settings);other['ui']['language']='en'
        for kind in ('detector','learning','evidence','segmentation','robustness'):
            args=(kind,'missing-video.mp4','missing-gt.csv',{},[])
            self.assertEqual(app._stable_signature(*args,settings),app._stable_signature(*args,other))


class LiveUITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'settings.json'
        self.settings=json.loads(app.SETTINGS_PATH.read_text(encoding='utf-8'))
        self.settings['auto_update']={'enabled':False,'check_on_startup':False}
        self.settings['ui']['language']='ko'
        self.path.write_text(json.dumps(self.settings),encoding='utf-8')
        self.mock=patch.object(app,'SETTINGS_PATH',self.path);self.mock.start()
        self.ui_state=patch.object(app,'UI_STATE_PATH',Path(self.tmp.name)/'ui_state.json');self.ui_state.start()
        # Disable startup network checks regardless of legacy config key spelling.
        self.updates=patch.object(app,'load_update_config',return_value={'enabled':False});self.updates.start()
        self.root=app.App();self.root.withdraw();self.root.update()

    def tearDown(self):
        self.root.destroy();self.updates.stop();self.ui_state.stop();self.mock.stop();self.tmp.cleanup();set_language('ko')

    def widgets(self,root):
        return [root]+[w for child in root.winfo_children() for w in self.widgets(child)]

    def test_open_windows_switch_without_losing_forms(self):
        self.root.video_var.set('Video_Profile.mp4')
        self.root.gt_var.set('ground_truth.csv')
        self.root.open_validation_center();self.root.open_settings();self.root.update()
        widgets=self.widgets(self.root)
        before_ids=[str(w) for w in widgets]
        before_text={str(w):w.cget('text') for w in widgets if 'text' in w.keys()}
        self.root.language_var.set('en');self.root._change_language();self.root.update()
        self.assertEqual(before_ids,[str(w) for w in self.widgets(self.root)])
        self.assertEqual(self.root.video_var.get(),'Video_Profile.mp4')
        self.assertEqual(self.root.gt_var.get(),'ground_truth.csv')
        values=[w.cget('text') for w in widgets if 'text' in w.keys()]
        self.assertIn('Save Validation Settings',values)
        self.assertIn('Download & Prepare',values)
        self.assertIn('Set slot points',values)
        self.assertEqual(json.loads(self.path.read_text())['ui']['language'],'en')
        # Saving only language must leave detector/validation/research settings identical.
        saved=json.loads(self.path.read_text())
        expected=copy.deepcopy(self.settings);expected['ui']['language']='en'
        self.assertEqual(saved,expected)
        self.root.language_var.set('ko');self.root._change_language();self.root.update()
        self.assertEqual(before_text,{str(w):w.cget('text') for w in widgets if 'text' in w.keys()})

    def test_restart_and_dynamic_status(self):
        self.root.language_var.set('en');self.root._change_language()
        self.root.destroy();self.root=app.App();self.root.withdraw();self.root.update()
        self.assertEqual(self.root.language_var.get(),'en')
        self.root.language_var.set('ko');self.root._change_language()
        self.root._progress(.1,'1/7 Detector tuning | CACHE HIT');self.root.update()
        self.assertIn('검출기 튜닝',self.root.status_var.get())
        self.assertIn('전체 경과',self.root.timing_var.get())

    def test_split_intervals_persist_with_profile(self):
        import validation_center as vc
        from tkinter import ttk
        self.root.language_var.set('en');self.root._change_language()
        with patch.object(vc,'APP_DIR',Path(self.tmp.name)), \
             patch.object(vc,'PROFILE_PATH',Path(self.tmp.name)/'profiles.json'), \
             patch.object(vc.messagebox,'showinfo'), patch.object(vc.messagebox,'showerror') as errors:
            self.root.open_validation_center();self.root.update()
            widgets=self.widgets(self.root)
            entries=[w for w in widgets if isinstance(w,ttk.Entry)]
            dev=next(w for w in entries if w.get()=='5:30-20:30')
            test=next(w for w in entries if w.get()=='0:00-5:30')
            dev.delete(0,'end');dev.insert(0,'6:00-20:30')
            test.delete(0,'end');test.insert(0,'0:00-6:00')
            buttons={w.cget('text'):w for w in widgets if isinstance(w,ttk.Button)}
            snapshot=Path(self.tmp.name)/'manual.csv'
            snapshot.write_text('time_sec,global_slot_id,initial_state\n0,G001,U\n900,G001,O\n')
            check=next(w for w in widgets if isinstance(w,ttk.Checkbutton) and w.cget('text')=='Use exact-time manual snapshots in restart experiments only')
            check.invoke()
            manual_entry=next(w for w in entries if w.master==check.master and w.grid_info().get('row')==22)
            manual_entry.delete(0,'end');manual_entry.insert(0,str(snapshot))
            buttons['Save Validation Settings'].invoke()
            manual=json.loads(self.path.read_text())['validation']['restart_experiments']['manual_init']
            self.assertEqual(manual,{'enabled':True,'path':str(snapshot),'source':'operator_snapshot'})
            self.assertEqual(json.loads(self.path.read_text())['validation']['restart_experiments']['starts_sec'],[0,330,660,900])
            cfg=json.loads(self.path.read_text())['validation']['split_experiments']
            self.assertEqual(cfg,{'enabled':True,'dev':[360,1230],'test':[0,360]})
            profile=next(w for w in widgets if isinstance(w,ttk.Combobox) and str(w.cget('state'))=='normal')
            profile.set('split-profile');buttons['Save Profile'].invoke()
            dev.delete(0,'end');dev.insert(0,'5:30-20:30')
            buttons['Load Profile'].invoke()
            self.assertEqual(dev.get(),'6:00-20:30')
            self.assertEqual(test.get(),'0:00-6:00')
            self.assertEqual(manual_entry.get(),str(snapshot))
            self.assertFalse(errors.called)

    def test_manual_editor_saves_one_time_without_overwriting_other_snapshots(self):
        import tkinter as tk
        from tkinter import ttk
        import manual_init_ui as ui
        from manual_initialization import load_snapshot,write_snapshot
        import numpy as np
        self.root.language_var.set('en');self.root._change_language()
        folder=Path(self.tmp.name);slots=folder/'slots.json';snapshot=folder/'manual.csv'
        slots.write_text(json.dumps({'slots':[{'local_id':'S1','global_id':'G1','cctv':'cctv1','point':[10,10]}]}))
        write_snapshot(snapshot,330,{'G1':'EMPTY'})
        variable=tk.StringVar(master=self.root,value=str(snapshot))
        class Video:
            def release(self): pass
        with patch.object(ui.filedialog,'askopenfilename',return_value=str(slots)),patch.object(ui.filedialog,'asksaveasfilename',return_value=str(snapshot)),patch.object(ui.messagebox,'showinfo'),patch.object(ui.messagebox,'showerror') as errors,patch.object(ui,'open_video',return_value=Video()),patch.object(ui,'read_frame_at',return_value=np.zeros((50,50,3),dtype=np.uint8)) as frames,patch.object(ui,'crop_roi',side_effect=lambda frame,roi:frame):
            ui.open_snapshot_editor(self.root,self.root,variable);self.root.update()
            widgets=self.widgets(self.root);buttons={w.cget('text'):w for w in widgets if isinstance(w,ttk.Button)}
            buttons['Load exact-time snapshot / video'].invoke()
            self.assertEqual(frames.call_args.args[1],900)
            tree=next(w for w in widgets if isinstance(w,ttk.Treeview));tree.selection_set('G1')
            buttons['O = occupied'].invoke();buttons['Save restart snapshots'].invoke()
            self.assertEqual(load_snapshot(snapshot,900,['G1']),{'G1':'OCCUPIED'})
            self.assertEqual(load_snapshot(snapshot,330,['G1']),{'G1':'EMPTY'})
            self.assertFalse(errors.called)

    def test_validation_messagebox_is_localized(self):
        self.root.video_var.set('')
        with patch.object(app.messagebox,'showerror') as show:
            self.assertFalse(self.root._validate_inputs())
            self.assertEqual(show.call_args.args[0],'영상 필요')
        self.root.language_var.set('en');self.root._change_language()
        with patch.object(app.messagebox,'showerror') as show:
            self.assertFalse(self.root._validate_inputs())
            self.assertEqual(show.call_args.args[0],'Video required')

if __name__=='__main__':
    unittest.main()
