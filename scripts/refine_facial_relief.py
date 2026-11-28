"""Localized facial relief correction on latest actual mesh; preserve UV and hair."""
import bpy, numpy as np, pathlib, math, json, os
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix, diags
p=pathlib.Path("output").resolve()
bpy.ops.wm.open_mainfile(filepath=str(p/"base/xuanyuan_scalp_fit_candidate.blend"))
body=next(o for o in bpy.context.scene.objects if o.type=="MESH" and not o.name.startswith("Hair_"))
hair=[o for o in bpy.context.scene.objects if o.type=="MESH" and o.name.startswith("Hair_")]
m=body.data; M=body.matrix_world.copy(); inv=M.inverted()
a=np.array([tuple(M@v.co) for v in m.vertices],float)
lo=a.min(0);hi=a.max(0);h=hi[2]-lo[2];cx=(lo[0]+hi[0])/2
assert h>1
x=(a[:,0]-cx)/h;z=(a[:,2]-lo[2])/h;depth=-a[:,1]/h
# Original neutral face and side profile in reference sheet: chin at roughly
# 0.866 body height, nose tip at 0.902, eyes at 0.919. Smooth windows only
# affect the visible facial relief; no mesh primitives or replacement face.
def smoothstep(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)
def gauss(v,c,s):return np.exp(-.5*((v-c)/s)**2)
uv=m.uv_layers.active
assert uv is not None
uv_before=np.array([tuple(l.uv) for l in uv.data])
# Sample existing baked skin colour from original mesh's UV image.
images=[n.image for mat in m.materials if mat and mat.use_nodes for n in mat.node_tree.nodes if n.type=="TEX_IMAGE" and n.image]
assert images
im=max(images,key=lambda q:q.size[0]*q.size[1])
w,hh=im.size;channels=im.channels
pixels=np.empty(w*hh*channels,np.float32);im.pixels.foreach_get(pixels)
pixels=pixels.reshape(hh,w,channels)
loopvid=np.array([l.vertex_index for l in m.loops])
u=np.clip((uv_before[:,0]*(w-1)).round().astype(int),0,w-1)
v=np.clip((uv_before[:,1]*(hh-1)).round().astype(int),0,hh-1)
rgb=pixels[v,u,:3];skinloop=((rgb[:,0]>.12)&(rgb[:,0]>rgb[:,1]*1.06)&(rgb[:,1]>rgb[:,2]*1.015)).astype(float)
count=np.bincount(loopvid,minlength=len(a));score=np.bincount(loopvid,weights=skinloop,minlength=len(a))/np.maximum(count,1)
edge=np.array([tuple(e.vertices) for e in m.edges])
row=np.r_[edge[:,0],edge[:,1]];col=np.r_[edge[:,1],edge[:,0]]
A=coo_matrix((np.ones(len(row)),(row,col)),shape=(len(a),len(a))).tocsr()
A=diags(1/np.maximum(np.asarray(A.sum(1)).ravel(),1))@A
skin=np.maximum(score,.65*(A@score))
skin=np.maximum(skin,.45*(A@skin))
# Frontal outer surface only: ray test distinguishes face from hair/back.
bvh=BVHTree.FromPolygons([Vector(q) for q in a],[tuple(f.vertices) for f in m.polygons])
visible=np.zeros(len(a))
region=(abs(x)<.046)&(z>.855)&(z<.937)&(depth>.004)
for i in np.flatnonzero(region):
    q=a[i];hit=bvh.ray_cast(Vector((q[0],lo[1]-h,q[2])),Vector((0,1,0)))[0]
    if hit is not None:
        visible[i]=max(0,1-abs(hit.y-q[1])/(h*.010))
mask=visible*skin*smoothstep((z-.855)/.011)*smoothstep((.937-z)/.012)*smoothstep((.046-abs(x))/.013)
# Smooth local depth irregularities, then add bounded continuous relief.
# The nose bridge joins the tip, eye sockets recede, and lip/chin retain
# the reference's restrained male profile rather than enlarging the head.
d=depth.copy()
for _ in range(5):
    d=d+.40*mask*((A@d)-d)
bridge=.0040*gauss(x,0,.009)*gauss(z,.916,.015)
tip=.0065*gauss(x,0,.009)*gauss(z,.902,.0065)
sockets=-.0026*(gauss(x,-.017,.009)+gauss(x,.017,.009))*gauss(z,.919,.006)
mouth=.0021*gauss(x,0,.014)*gauss(z,.883,.004)
chin=.0015*gauss(x,0,.014)*gauss(z,.869,.006)
raw=mask*(d-depth+bridge+tip+sockets+mouth+chin)
delta=raw.copy()
for _ in range(2):delta=.75*delta+.25*mask*(A@delta)
delta=np.clip(delta,-.006,.009)
assert np.count_nonzero(abs(delta)>1e-6)>30, "face mask too small"
m.calc_loop_triangles();tri=np.array([tuple(t.vertices) for t in m.loop_triangles])
def normals(q):return np.cross(q[tri[:,1]]-q[tri[:,0]],q[tri[:,2]]-q[tri[:,0]])
n0=normals(a);valid=np.linalg.norm(n0,axis=1)>h*h*1e-12
# Backtracking avoids introducing severe local folds.
for scale in (1,.75,.5,.25,.125):
    b=a.copy();b[:,1]-=h*delta*scale
    n1=normals(b)
    flip=int(((n0[valid]*n1[valid]).sum(1)<0).sum())
    if flip==0:break
assert flip==0,"facial correction would invert faces"
assert np.max(np.abs(b-a))>1e-5
scene=bpy.context.scene;scene.render.engine="CYCLES";scene.cycles.samples=16
scene.render.resolution_x=480;scene.render.resolution_y=480;scene.render.resolution_percentage=100
scene.view_settings.view_transform="Standard"
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type="ORTHO";cam.data.ortho_scale=h*.205;scene.camera=cam
c=Vector((cx,(lo[1]+hi[1])/2,lo[2]+h*.91))
sheet=Image.new("RGB",(1920,1000),"white");draw=ImageDraw.Draw(sheet)
for stage in range(2):
    for vert,q in zip(m.vertices,b if stage else a):vert.co=inv@Vector(q)
    m.update()
    for j,ang in enumerate((0,45,90,315)):
        r=math.radians(ang);cam.location=c+Vector((math.sin(r)*h*3,-math.cos(r)*h*3,0))
        cam.rotation_euler=(c-cam.location).to_track_quat("-Z","Y").to_euler()
        path=p/f"face_{stage}_{ang}.png";scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        sheet.paste(Image.open(path).convert("RGB"),(480*j,500*stage));draw.text((480*j+5,500*stage+483),f"{'AFTER' if stage else 'BEFORE'} {ang}",fill="black")
sheet.save(p/"facial_relief_comparison.png")
bpy.data.objects.remove(cam,do_unlink=True)
assert np.array_equal(uv_before,np.array([tuple(l.uv) for l in uv.data]))
for o in scene.objects:o.select_set(False)
for o in [body]+hair:o.select_set(True)
bpy.context.view_layer.objects.active=body
bpy.ops.export_scene.gltf(filepath=str(p/"xuanyuan_facial_relief_candidate.glb"),export_format="GLB",use_selection=True)
bpy.ops.wm.save_as_mainfile(filepath=str(p/"xuanyuan_facial_relief_candidate.blend"))
report={"source_run":46132709831,"changed_vertices":int((np.linalg.norm(b-a,axis=1)>1e-6).sum()),"max_displacement_m":float(np.max(np.abs(b-a))),"new_inverted_faces":flip,"hair_objects_preserved":len(hair),"uv_unchanged":True,"backtracking_scale":scale,"limitations":["existing face texture and fused fringe remain","eyes and lips not independent geometry","not final avatar"],"final_avatar":False}
(p/"facial_relief_report.json").write_text(json.dumps(report,indent=2));print(report,flush=True);os._exit(0)
