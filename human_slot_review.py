"""Local human review of exact-time CCTV frames; no count GT is read."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from manual_initialization import load_snapshot, write_snapshot
from utils import crop_roi, open_video, order_quad_points, read_frame_at

PRIORITY = ('G018', 'G027', 'G028', 'G029')
VALID = {'OCCUPIED', 'EMPTY', 'UNKNOWN'}
TIME_LABELS = {900:'15:00', 0:'0:00', 330:'5:30', 660:'11:00'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


class ReviewSession:
    def __init__(self, root, start, slots, video_path, slots_path, roi_path, pseudo_path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.start = int(start)
        self.ids = sorted({str(s['global_id']) for s in slots}, key=lambda x: (x not in PRIORITY, PRIORITY.index(x) if x in PRIORITY else x))
        if not self.ids: raise ValueError('No slots available')
        self.provenance = {'time_sec': self.start, 'video_sha256': sha(video_path), 'slots_sha256': sha(slots_path), 'roi_source_sha256': sha(roi_path), 'pseudo_sha256': sha(pseudo_path)}
        self.pseudo = load_snapshot(pseudo_path, self.start, self.ids)
        if self.pseudo is None: raise ValueError('Pseudo snapshot is missing the exact review time')
        self.choices = {}
        self.progress_path = self.root / f'progress_{self.start:04d}.json'
        if self.progress_path.exists():
            data = json.loads(self.progress_path.read_text(encoding='utf-8'))
            if data['provenance'] != self.provenance: raise ValueError('Saved review inputs changed; use a new review directory')
            self.choices = data['choices']
            if set(self.choices) - set(self.ids) or any(v not in VALID for v in self.choices.values()): raise ValueError('Invalid saved review')

    def select(self, gid, state):
        if gid not in self.ids or state not in VALID: raise ValueError('Invalid selection')
        self.choices[gid] = state
        self.save_progress()

    def save_progress(self):
        atomic_json(self.progress_path, {'status': 'in_progress', 'source': 'operator_review', 'provenance': self.provenance,
                                        'choices': self.choices, 'reviewed_count': len(self.choices), 'total': len(self.ids)})

    def publish(self):
        if set(self.choices) != set(self.ids): raise ValueError('모든 주차면을 확인한 후 저장해 주세요.')
        out = self.root / 'verified'; out.mkdir(exist_ok=True)
        csv_path = out / f'human_{self.start:04d}.csv'; audit_path = out / f'human_{self.start:04d}.audit.json'
        if csv_path.exists() or audit_path.exists():
            backup = self.root / 'backups' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'); backup.mkdir(parents=True)
            for path in (csv_path, audit_path):
                if path.exists(): shutil.copy2(path, backup / path.name)
        temp = out / f'human_{self.start:04d}.tmp'
        write_snapshot(temp, self.start, self.choices)
        file_hash = sha(temp)
        audit = {'status': 'human_review_complete', 'source': 'operator_review', 'provenance': self.provenance,
                 'reviewed_slot_ids': self.ids, 'reviewed_count': len(self.choices), 'unknown_count': sum(v == 'UNKNOWN' for v in self.choices.values()),
                 'snapshot_sha256': file_hash, 'saved_local_time': datetime.now().isoformat(), 'pseudo_is_human_label': False}
        os.replace(temp, csv_path)
        atomic_json(audit_path, audit)
        return csv_path


def raw_point(slot, roi):
    x, y = map(float, slot.get('manual_point', slot['point']))
    if isinstance(roi, dict) and roi.get('mode') == 'perspective':
        w, h = roi['output_size']; dst = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        matrix = cv2.getPerspectiveTransform(dst, order_quad_points(roi['points']).astype(np.float32))
        return tuple(np.rint(cv2.perspectiveTransform(np.float32([[[x, y]]]), matrix)[0, 0]).astype(int))
    return int(x + roi[0]), int(y + roi[1])


def pictures(frame, slots, rois, gid, camera):
    whole = frame.copy()
    for s in slots:
        if str(s['global_id']) != gid: continue
        roi = rois[s['cctv']]; x, y = raw_point(s, roi)
        cv2.circle(whole, (x, y), 15, (0, 0, 255), 3)
        cv2.putText(whole, f"{gid} {s['cctv']}", (x+18, max(25, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    matching = [s for s in slots if str(s['global_id']) == gid and s['cctv'] == camera]
    if not matching: raise ValueError('Selected camera does not show this slot')
    s = matching[0]; view = crop_roi(frame, rois[camera]); x, y = map(int, s.get('manual_point', s['point']))
    height, width = view.shape[:2]
    if not (0 <= x < width and 0 <= y < height): raise ValueError('Slot point is outside the CCTV view')
    x1, x2, y1, y2 = max(0,x-140), min(width,x+140), max(0,y-140), min(height,y+140)
    cv2.circle(view, (x,y), 7, (0,0,255), 2)
    return whole, view[y1:y2,x1:x2].copy()


class ReviewApp:
    def __init__(self, window, config):
        self.window=window; self.config=config; self.sessions={}; self.frames={}; self.images=[]
        window.title('주차면 초기 상태 확인 — 15:00 우선'); window.geometry('1450x850'); window.minsize(1100,700)
        header=ttk.Frame(window); header.pack(fill='x',padx=10,pady=5)
        ttk.Label(header,text='시점:').pack(side='left')
        supported=[TIME_LABELS[t] for t in (900,0,330,660) if str(t) in config['times']]
        if not supported:raise ValueError('No supported snapshot times in configuration')
        self.time=tk.StringVar(value=supported[0]); selector=ttk.Combobox(header,textvariable=self.time,values=supported,state='readonly',width=10);selector.pack(side='left');selector.bind('<<ComboboxSelected>>',lambda e:self.load_time())
        ttk.Label(header,text='AI 추정은 참고용입니다. 각 면을 직접 선택해야 검토 완료됩니다.').pack(side='left',padx=14)
        self.progress=ttk.Label(header,text='');self.progress.pack(side='right')
        body=ttk.Frame(window);body.pack(fill='both',expand=True,padx=8)
        left=ttk.Frame(body);left.pack(side='left',fill='y')
        self.tree=ttk.Treeview(left,columns=('ai','human'),show='headings',height=21,selectmode='browse')
        for col,title,width in [('ai','AI 참고',115),('human','내 확인',130)]: self.tree.heading(col,text=title);self.tree.column(col,width=width,anchor='center')
        self.tree.configure(show='tree headings');self.tree.heading('#0',text='주차면');self.tree.column('#0',width=80,anchor='center')
        self.tree.pack(side='left',fill='y');bar=ttk.Scrollbar(left,orient='vertical',command=self.tree.yview);bar.pack(side='left',fill='y');self.tree.configure(yscrollcommand=bar.set)
        self.tree.bind('<<TreeviewSelect>>',lambda e:self.show_slot())
        right=ttk.Frame(body);right.pack(side='left',fill='both',expand=True,padx=10)
        row=ttk.Frame(right);row.pack(fill='x');self.title=ttk.Label(row,text='',font=('Malgun Gothic',15,'bold'));self.title.pack(side='left')
        ttk.Label(row,text='화면 선택:').pack(side='left',padx=(20,3));self.camera=tk.StringVar();self.cam=ttk.Combobox(row,textvariable=self.camera,state='readonly',width=12);self.cam.pack(side='left');self.cam.bind('<<ComboboxSelected>>',lambda e:self.render())
        self.info=ttk.Label(right,text='',font=('Malgun Gothic',11));self.info.pack(anchor='w',pady=6)
        panels=ttk.Frame(right);panels.pack(fill='both',expand=True)
        full=ttk.LabelFrame(panels,text='전체 CCTV 화면 — 빨간 원이 선택한 주차면');full.pack(side='left',fill='both',expand=True)
        zoom=ttk.LabelFrame(panels,text='선택 카메라 확대 — 빨간 원은 기준점');zoom.pack(side='left',fill='both',expand=True)
        self.full=ttk.Label(full,anchor='center');self.full.pack(fill='both',expand=True);self.zoom=ttk.Label(zoom,anchor='center');self.zoom.pack(fill='both',expand=True)
        actions=ttk.Frame(right);actions.pack(fill='x',pady=10)
        for title,state in [('차량 있음 (O)','OCCUPIED'),('빈자리 (E)','EMPTY'),('판단 불가 (U)','UNKNOWN')]:
            ttk.Button(actions,text=title,command=lambda s=state:self.choose(s)).pack(side='left',padx=5,ipadx=8,ipady=5)
        ttk.Button(actions,text='검토 완료 저장',command=self.publish).pack(side='right',padx=5,ipadx=8,ipady=5)
        ttk.Label(right,text='선택은 즉시 임시 저장됩니다. 판단이 어려우면 판단 불가를 누르세요. 검토 완료 저장은 모든 면을 확인한 후 가능합니다.').pack(anchor='w')
        self.load_time()

    def load_time(self):
        reverse={label:sec for sec,label in TIME_LABELS.items()};self.start=reverse[self.time.get()]
        data=self.config['times'][str(self.start)]
        if self.start not in self.sessions:
            self.slots=json.loads(Path(data['slots']).read_text(encoding='utf-8'))['slots']
            self.rois=json.loads(Path(data['roi_source']).read_text(encoding='utf-8'))['rois']
            self.sessions[self.start]=ReviewSession(self.config['review_dir'],self.start,self.slots,self.config['video'],data['slots'],data['roi_source'],data['pseudo'])
            cap=open_video(self.config['video'])
            try: self.frames[self.start]=read_frame_at(cap,self.start)
            finally: cap.release()
        self.slots=json.loads(Path(data['slots']).read_text(encoding='utf-8'))['slots'];self.rois=json.loads(Path(data['roi_source']).read_text(encoding='utf-8'))['rois']
        self.session=self.sessions[self.start];self.refresh();self.tree.selection_set(self.session.ids[0]);self.show_slot()

    def refresh(self):
        previous=self.tree.selection();self.tree.delete(*self.tree.get_children())
        label={'OCCUPIED':'차량 있음','EMPTY':'빈자리','UNKNOWN':'판단 불가'}
        for gid in self.session.ids:
            human=label[self.session.choices[gid]] if gid in self.session.choices else '미확인'
            self.tree.insert('','end',iid=gid,text=gid,values=(label[self.session.pseudo[gid]],human))
        self.progress.configure(text=f'확인 {len(self.session.choices)} / {len(self.session.ids)}')
        if previous and previous[0] in self.session.ids:self.tree.selection_set(previous[0])

    def show_slot(self):
        selection=self.tree.selection()
        if not selection:return
        self.gid=selection[0]; cams=sorted({s['cctv'] for s in self.slots if str(s['global_id'])==self.gid})
        self.cam.configure(values=cams);self.camera.set(cams[0]);self.render()

    def render(self):
        if not self.tree.selection():return
        gid=self.tree.selection()[0];self.title.configure(text=f'{self.time.get()}  ·  {gid}')
        label={'OCCUPIED':'차량 있음','EMPTY':'빈자리','UNKNOWN':'판단 불가'}
        self.info.configure(text=f"AI 참고: {label[self.session.pseudo[gid]]}   |   내 확인: {label[self.session.choices[gid]] if gid in self.session.choices else '미확인'}")
        full,zoom=pictures(self.frames[self.start],self.slots,self.rois,gid,self.camera.get())
        self.images=[]
        for target,image,size in ((self.full,full,(650,560)),(self.zoom,zoom,(440,560))):
            im=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));im.thumbnail(size,Image.Resampling.LANCZOS)
            if target is self.zoom: im=im.resize((min(440,im.width*2),min(560,im.height*2)))
            photo=ImageTk.PhotoImage(im);self.images.append(photo);target.configure(image=photo)

    def choose(self,state):
        if not self.tree.selection():return
        gid=self.tree.selection()[0];self.session.select(gid,state);self.refresh()
        index=self.session.ids.index(gid)
        remaining=[g for g in self.session.ids[index+1:]+self.session.ids[:index] if g not in self.session.choices]
        if remaining:self.tree.selection_set(remaining[0]);self.tree.see(remaining[0]);self.show_slot()
        else:self.render()

    def publish(self):
        try:
            result=self.session.publish();messagebox.showinfo('저장 완료',f'사람 확인 결과를 저장했습니다.\n{result}\nUNKNOWN은 그대로 보존됩니다.',parent=self.window)
        except Exception as exc:messagebox.showerror('저장할 수 없습니다',str(exc),parent=self.window)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'));window=tk.Tk()
    try: ReviewApp(window,config)
    except Exception as exc: messagebox.showerror('화면을 열 수 없습니다',str(exc),parent=window);window.destroy();raise
    window.mainloop()


if __name__=='__main__': main()
