"""Repair actual facial displacement seams while retaining all UV islands."""
import bpy, numpy as np, pathlib, math, json, os
from mathutils import Vector
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
p=pathlib.Path("output").resolve()
def load(path):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    o=next(o for o in bpy.context.scene.objects if o.type=="MESH" and not o.name.startswith("Hair_"))
    o.data.calc_loop_triangles()
    return o,np.array([tuple(o.matrix_world@v.co) for v in o.data.vertices]),np.array([tuple(t.vertices) for t in o.data.loop_triangles])
_,a,tri0=load(p/"base/xuanyuan_scalp_fit_candidate.blend")
body,current,tri=load(p/"candidate/xuanyuan_facial_relief_candidate.blend")
assert a.shape==current.shape and np.array_equal(tri0,tri)
m=body.data;inv=body.matrix_world.inverted()
hair=[o for o in bpy.context.scene.objects if o.type=="MESH" and o.name.startswith("Hair_")]
lo=a.min(0);hi=a.max(0);h=hi[2]-lo[2];cx=(lo[0]+hi[0])/2
uv=m.uv_layers.active;uv_before=np.array([tuple(l.uv) for l in uv.data])
d=current-a;ids=np.flatnonzero(a[:,2]>lo[2]+.84*h)
parent=np.arange(len(a))
def find(i):
    while parent[i]!=i:
        parent[i]=parent[parent[i]];i=parent[i]
    return i
for i,j in cKDTree(a[ids]).query_pairs(h*0.000005):
    r,s=find(ids[i]),find(ids[j])
    if r!=s:parent[s]=r
groups={}
for i in ids:groups.setdefault(find(i),[]).append(i)
groups=[np.array(q) for q in groups.values() if len(q)>1]
def splits(d):return sum(np.max(np.ptp(d[g],axis=0))>1e-7 for g in groups)
split_before=splits(d)
for g in groups:d[g]=d[g].mean(0)
assert split_before>0
def normals(q):return np.cross(q[tri[:,1]]-q[tri[:,0]],q[tri[:,2]]-q[tri[:,0]])
n0=normals(a);valid=np.linalg.norm(n0,axis=1)>h*h*1e-12
for scale in (1,.75,.5,.25,.125):
    b=a+d*scale;n1=normals(b)
    flip=int(((n0[valid]*n1[valid]).sum(1)<0).sum())
    if flip==0:break
assert flip==0
split_after=splits(b-a)
assert split_after==0
assert np.count_nonzero(np.linalg.norm(b-current,axis=1)>1e-6)>0
scene=bpy.context.scene;scene.render.engine="CYCLES";scene.cycles.samples=16
scene.render.resolution_x=480;scene.render.resolution_y=480;scene.render.resolution_percentage=100
scene.view_settings.view_transform="Standard"
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type="ORTHO";cam.data.ortho_scale=h*.205;scene.camera=cam
c=Vector((cx,(lo[1]+hi[1])/2,lo[2]+h*.91))
sheet=Image.new("RGB",(1920,1000),"white");draw=ImageDraw.Draw(sheet)
for stage in range(2):
    for vert,q in zip(m.vertices,b if stage else current):vert.co=inv@Vector(q)
    m.update()
    for j,ang in enumerate((0,45,90,315)):
        r=math.radians(ang);cam.location=c+Vector((math.sin(r)*h*3,-math.cos(r)*h*3,0))
        cam.rotation_euler=(c-cam.location).to_track_quat("-Z","Y").to_euler()
        path=p/f"face_{stage}_{ang}.png";scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        sheet.paste(Image.open(path).convert("RGB"),(480*j,500*stage));draw.text((480*j+5,500*stage+483),f"{'AFTER' if stage else 'BEFORE'} {ang}",fill="black")
sheet.save(p/"face_seam_comparison.png")
bpy.data.objects.remove(cam,do_unlink=True)
assert np.array_equal(uv_before,np.array([tuple(l.uv) for l in uv.data]))
for o in scene.objects:o.select_set(False)
for o in [body]+hair:o.select_set(True)
bpy.context.view_layer.objects.active=body
bpy.ops.export_scene.gltf(filepath=str(p/"xuanyuan_face_seam_candidate.glb"),export_format="GLB",use_selection=True)
bpy.ops.wm.save_as_mainfile(filepath=str(p/"xuanyuan_face_seam_candidate.blend"))
report={"source_run":46139011839,"split_groups_before":split_before,"split_groups_after":split_after,"changed_vertices":int((np.linalg.norm(b-current,axis=1)>1e-6).sum()),"max_displacement_m":float(np.max(np.abs(b-a))),"new_inverted_faces":flip,"hair_objects_preserved":len(hair),"uv_unchanged":True,"backtracking_scale":scale,"limitations":["existing face texture and fused fringe remain","eyes and lips not independent geometry","not final avatar"],"final_avatar":False}
(p/"face_seam_report.json").write_text(json.dumps(report,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)));print(report,flush=True);os._exit(0)
