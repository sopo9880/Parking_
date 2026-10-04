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
        # Disable startup network checks regardless of legacy config key spelling.
        self.updates=patch.object(app,'load_update_config',return_value={'enabled':False});self.updates.start()
        self.root=app.App();self.root.withdraw();self.root.update()

    def tearDown(self):
        self.root.destroy();self.updates.stop();self.mock.stop();self.tmp.cleanup();set_language('ko')

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
