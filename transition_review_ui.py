"""Local targeted event review. No AI proposal is converted into a human label."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk,messagebox
import cv2
from PIL import Image,ImageTk
from human_slot_review import sha,atomic_json,pictures
from utils import open_video,read_frame_at

LABELS={'PARKED':'계속 주차 중','DEPARTURE':'실제 출차','ARRIVAL':'실제 입차','OCCLUSION':'가림 또는 검출 실패','UNSURE':'판단 불가'}


class EventReviews:
    def __init__(self,path,cases_file,video,slots,roi):
        self.path=Path(path);self.cases_file=Path(cases_file);self.data=json.loads(self.cases_file.read_text(encoding='utf8'))
        for name,source in [('video_sha256',video),('fit_slots_sha256',slots),('roi_source_sha256',roi)]:
            if self.data[name]!=sha(source):raise ValueError('영상 또는 주차면 설정이 사례 목록과 다릅니다: '+name)
        self.signature=sha(self.cases_file);self.cases=self.data['cases'];self.ids={c['case_id'] for c in self.cases};self.choices={}
        if len(self.ids)!=len(self.cases):raise ValueError('Duplicate case IDs')
        if self.path.exists():
            stored=json.loads(self.path.read_text(encoding='utf8'))
            if stored['cases_sha256']!=self.signature:raise ValueError('저장된 검토의 사례 목록이 다릅니다')
            self.choices=stored['reviews']
            if set(self.choices)-self.ids or any(x['label'] not in LABELS for x in self.choices.values()):raise ValueError('Invalid stored review')

    def choose(self,case_id,label):
        if case_id not in self.ids or label not in LABELS:raise ValueError('Invalid review')
        self.choices[case_id]={'label':label,'source':'operator_review','saved_local_time':datetime.now().isoformat(),
                               'scope':'entire displayed clip; no exact transition time inferred'}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(self.path,{'status':'complete' if len(self.choices)==len(self.ids) else 'partial',
                               'cases_sha256':self.signature,'video_sha256':self.data['video_sha256'],'reviews':self.choices})


class EventApp:
    def __init__(self,win,video,slots_path,roi_path,cases_path,output):
        self.win=win;self.video=video;self.slots=json.loads(Path(slots_path).read_text(encoding='utf8'))['slots']
        self.rois=json.loads(Path(roi_path).read_text(encoding='utf8'))['rois'];self.model=EventReviews(output,cases_path,video,slots_path,roi_path)
        self.cap=open_video(video);self.timer=None;self.images=[];win.title('주차 전이 사례 확인 — 개발 영상');win.geometry('1450x850');win.minsize(1100,700)
        win.protocol('WM_DELETE_WINDOW',self.close)
        top=ttk.Frame(win);top.pack(fill='x',padx=10,pady=5)
        ttk.Label(top,text='AI가 선별한 미확인 사례입니다. 영상 확인 후 한 가지를 선택하세요.').pack(side='left')
        self.progress=ttk.Label(top);self.progress.pack(side='right')
        body=ttk.Frame(win);body.pack(fill='both',expand=True,padx=8)
        side=ttk.Frame(body);side.pack(side='left',fill='y');self.tree=ttk.Treeview(side,columns=('state',),show='tree headings',height=12)
        self.tree.heading('#0',text='시각 / 주차면');self.tree.column('#0',width=175);self.tree.heading('state',text='내 확인');self.tree.column('state',width=145)
        self.tree.pack(fill='y');self.tree.bind('<<TreeviewSelect>>',lambda e:self.select_case())
        right=ttk.Frame(body);right.pack(side='left',fill='both',expand=True,padx=10)
        head=ttk.Frame(right);head.pack(fill='x');self.title=ttk.Label(head,font=('Malgun Gothic',14,'bold'));self.title.pack(side='left')
        ttk.Label(head,text='카메라:').pack(side='left',padx=(20,4));self.camera=tk.StringVar();self.cam=ttk.Combobox(head,textvariable=self.camera,state='readonly',width=10);self.cam.pack(side='left');self.cam.bind('<<ComboboxSelected>>',lambda e:self.render())
        self.hint=ttk.Label(right);self.hint.pack(anchor='w',pady=4)
        panels=ttk.Frame(right);panels.pack(fill='both',expand=True)
        self.full=ttk.Label(ttk.LabelFrame(panels,text='전체 CCTV 화면'));self.full.master.pack(side='left',fill='both',expand=True);self.full.pack(fill='both',expand=True)
        self.zoom=ttk.Label(ttk.LabelFrame(panels,text='표시된 주차면 확대'));self.zoom.master.pack(side='left',fill='both',expand=True);self.zoom.pack(fill='both',expand=True)
        controls=ttk.Frame(right);controls.pack(fill='x',pady=4)
        ttk.Button(controls,text='▶ 재생',command=self.play).pack(side='left');ttk.Button(controls,text='일시정지',command=self.pause).pack(side='left')
        self.slider=tk.Scale(controls,orient='horizontal',resolution=1,showvalue=True,command=self.on_slide);self.slider.pack(side='left',fill='x',expand=True,padx=10)
        buttons=ttk.Frame(right);buttons.pack(fill='x',pady=8)
        for value in LABELS:ttk.Button(buttons,text=LABELS[value],command=lambda v=value:self.choose(v)).pack(side='left',padx=3,ipadx=4,ipady=4)
        ttk.Label(right,text='선택한 사례만 사람 확인으로 자동 저장됩니다. 정확한 출차 시각을 자동으로 만들지 않습니다.').pack(anchor='w')
        self.refresh();self.tree.selection_set(self.model.cases[0]['case_id']);self.select_case()

    def refresh(self):
        previous=self.tree.selection();self.tree.delete(*self.tree.get_children())
        for c in self.model.cases:
            value=self.model.choices.get(c['case_id']); label=LABELS[value['label']] if value else '미확인'
            self.tree.insert('','end',iid=c['case_id'],text=f"{int(c['event_sec'])//60}:{int(c['event_sec'])%60:02d}  {c['global_id']}",values=(label,))
        self.progress.configure(text=f'확인 {len(self.model.choices)} / {len(self.model.cases)}')
        if previous and previous[0] in self.model.ids:self.tree.selection_set(previous[0])

    def select_case(self):
        selection=self.tree.selection()
        if not selection:return
        self.pause();self.case=next(c for c in self.model.cases if c['case_id']==selection[0]);c=self.case
        cams=sorted({s['cctv'] for s in self.slots if str(s['global_id'])==c['global_id']});self.cam.configure(values=cams)
        preferred=next((s['cctv'] for s in self.slots if s['local_id']==c.get('local_id')),None);self.camera.set(preferred if preferred in cams else cams[0])
        self.slider.configure(from_=c['clip_start'],to=c['clip_end']);self.slider.set(c['event_sec'])
        self.title.configure(text=f"{int(c['event_sec'])//60}:{int(c['event_sec'])%60:02d} · {c['global_id']}")
        descriptions={'unconfirmed_release':'AI: 점유 → 미확정','possible_entry':'AI: 미확정 → 점유','moving_detection_unverified':'AI: 움직임 감지, 출차 여부 미확인',
                      'long_unknown':'AI: 장시간 미확정','camera_detection_disagreement':'AI: 카메라별 검출 불일치'}
        self.hint.configure(text=descriptions.get(c['kind'],c['kind'])+'  |  전후 영상을 확인해 주세요.');self.render()

    def on_slide(self,_):
        if hasattr(self,'case'):self.render()

    def render(self):
        if not hasattr(self,'case'):return
        t=float(self.slider.get());frame=read_frame_at(self.cap,t);full,zoom=pictures(frame,self.slots,self.rois,self.case['global_id'],self.camera.get());self.images=[]
        for target,img,size in ((self.full,full,(650,560)),(self.zoom,zoom,(440,560))):
            im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail(size,Image.Resampling.LANCZOS)
            if target is self.zoom:im=im.resize((min(440,im.width*2),min(560,im.height*2)))
            photo=ImageTk.PhotoImage(im);self.images.append(photo);target.configure(image=photo)

    def play(self):
        self.pause()
        if self.slider.get()>=self.case['clip_end']:self.slider.set(self.case['clip_start'])
        def step():
            if self.slider.get()>=self.case['clip_end']:self.timer=None;return
            self.slider.set(self.slider.get()+1);self.timer=self.win.after(1000,step)
        step()

    def pause(self):
        if self.timer is not None:self.win.after_cancel(self.timer);self.timer=None

    def choose(self,label):
        case_id=self.case['case_id'];self.model.choose(case_id,label);self.refresh()
        remaining=[c['case_id'] for c in self.model.cases if c['case_id'] not in self.model.choices]
        if remaining:self.tree.selection_set(remaining[0]);self.select_case()

    def close(self):self.pause();self.cap.release();self.win.destroy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    config=json.loads(args.config.read_text(encoding='utf8'));win=tk.Tk()
    try:EventApp(win,config['video'],config['slots'],config['roi'],config['cases'],config['output'])
    except Exception as exc:messagebox.showerror('화면을 열 수 없습니다',str(exc),parent=win);win.destroy();raise
    win.mainloop()


if __name__=='__main__':main()
