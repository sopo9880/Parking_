"""One-time operator snapshot editor; never reads count GT."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import cv2
from PIL import Image,ImageTk
from localization import tr
from manual_initialization import load_snapshot,write_snapshot,read_snapshots
from utils import load_json,open_video,read_frame_at,crop_roi

def open_snapshot_editor(parent,app,path_var):
    root=Path(__file__).resolve().parent
    slots_path=filedialog.askopenfilename(parent=parent,title=tr('Select slot geometry JSON'),initialdir=str(root),filetypes=[('JSON','*.json')])
    if not slots_path: return
    slots=load_json(slots_path,{}).get('slots',[])
    ids=sorted({str(s['global_id']) for s in slots})
    if not ids: raise ValueError('No global slot IDs')
    win=tk.Toplevel(parent);win.title(tr('Manual restart snapshot'));win.geometry('1100x760')
    time_var=tk.StringVar(value='15:00');camera=tk.StringVar(value=slots[0]['cctv'])
    values={gid:'UNKNOWN' for gid in ids};preview=ttk.Label(win)
    rois=load_json(Path(slots_path).parent/'worker_input.json',{}).get('rois') or load_json(root/'rois.json',{})
    top=ttk.Frame(win);top.pack(fill='x')
    ttk.Label(top,text=tr('Restart time (mm:ss)')).pack(side='left')
    ttk.Entry(top,textvariable=time_var,width=10).pack(side='left')
    ttk.Combobox(top,textvariable=camera,values=sorted({s['cctv'] for s in slots}),state='readonly',width=12).pack(side='left')
    def timestamp():
        from validation_center import parse_time_list
        times=parse_time_list(time_var.get(),include_zero=True)
        if len(times)!=1: raise ValueError('Specify one restart time')
        return times[0]
    actions=ttk.Frame(win);actions.pack(fill='x')
    tree=ttk.Treeview(win,columns=('slot','views','state'),show='headings',height=14,selectmode='extended')
    for col,label in [('slot','Global slot'),('views','Camera / local slot'),('state','Initial O/E/U')]: tree.heading(col,text=tr(label))
    tree.column('slot',width=95);tree.column('views',width=270);tree.column('state',width=150)
    tree.pack(side='left',fill='y',padx=5,pady=5);preview.pack(side='right',fill='both',expand=True)
    def refresh():
        tree.delete(*tree.get_children())
        for gid in ids:
            views='; '.join(s['cctv']+'/'+s['local_id'] for s in slots if str(s['global_id'])==gid)
            tree.insert('','end',iid=gid,values=(gid,views,values[gid]))
    def display():
        try:
            t=timestamp();snap=load_snapshot(path_var.get(),t,ids) if Path(path_var.get()).is_file() else None
            values.update(snap or {g:'UNKNOWN' for g in ids});refresh()
            cap=open_video(app.video_var.get().strip())
            try: frame=read_frame_at(cap,t)
            finally: cap.release()
            if frame is None: raise ValueError('No video frame at selected time')
            roi=rois.get(camera.get())
            image=crop_roi(frame,roi) if roi else frame
            if roi:
                for s in slots:
                    if s['cctv']!=camera.get(): continue
                    x,y=map(int,s.get('manual_point',s['point']));state=values[str(s['global_id'])]
                    color={'OCCUPIED':(0,0,230),'EMPTY':(0,190,0),'UNKNOWN':(0,190,230)}[state]
                    cv2.circle(image,(x,y),5,color,-1);cv2.putText(image,str(s['global_id']),(x+5,y),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
            im=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));im.thumbnail((520,560))
            preview.image=ImageTk.PhotoImage(im);preview.configure(image=preview.image)
        except Exception as exc: messagebox.showerror(tr('Evaluation error'),tr(str(exc)),parent=win)
    def set_state(state):
        selected=tree.selection()
        for gid in selected: values[gid]=state;tree.item(gid,values=(gid,tree.item(gid,'values')[1],state))
    def save():
        try:
            t=timestamp();target=filedialog.asksaveasfilename(parent=win,title=tr('Save restart snapshots'),defaultextension='.csv',initialfile='manual_init_snapshots.csv')
            if not target: return
            # Preserve other exact-time snapshots when saving into the same file.
            old,_=read_snapshots(target) if Path(target).is_file() else ([],[])
            retained=[r for r in old if float(r['time_sec'])!=t]
            write_snapshot(target,t,values)
            if retained:
                import csv
                with Path(target).open('a',encoding='utf-8',newline='') as f:
                    writer=csv.writer(f)
                    for row in retained: writer.writerow([row['time_sec'],row['global_slot_id'],row['initial_state']])
            path_var.set(target);messagebox.showinfo(tr('Saved'),tr('Only this start state is used; later frames are automatic.'),parent=win)
        except Exception as exc: messagebox.showerror(tr('Evaluation error'),tr(str(exc)),parent=win)
    for label,command in [('Load exact-time snapshot / video',display),('O = occupied',lambda:set_state('OCCUPIED')),('E = empty',lambda:set_state('EMPTY')),('U = unknown',lambda:set_state('UNKNOWN')),('Save restart snapshots',save)]:
        ttk.Button(top if label=='Load exact-time snapshot / video' else actions,text=tr(label),command=command).pack(side='left',padx=2)
    refresh()
