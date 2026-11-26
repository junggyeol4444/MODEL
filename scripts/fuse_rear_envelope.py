import bpy, numpy as np, pathlib, json, math, os, sys
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from PIL import Image, ImageDraw
p=pathlib.Path("output").resolve()
bpy.ops.wm.open_mainfile(filepath=str(p/"base/xuanyuan_multiview.blend"))
o=next(o for o in bpy.context.scene.objects if o.type=="MESH")
m=o.data
a=np.array([tuple(v.co) for v in m.vertices])
lo=a.min(0); hi=a.max(0); h=hi[2]-lo[2]
assert h>max(hi[:2]-lo[:2])
before=set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=str(p/"rear/0/mesh.glb"))
objs=[x for x in bpy.data.objects if x not in before and x.type=="MESH"]
assert objs
b=[]; faces=[]
for x in objs:
    offset=len(b)
    b.extend(tuple(x.matrix_world@v.co) for v in x.data.vertices)
    faces.extend(tuple(offset+i for i in f.vertices) for f in x.data.polygons)
b=np.array(b)
# Candidate is reconstructed from rear view. Search upright axis permutations
# and horizontal reflection using silhouette agreement, not assumed importer axes.
from scipy.spatial import cKDTree
target=(a[:,[0,2]]-lo[[0,2]])/(hi-lo)[[0,2]]
tree=cKDTree(target)
best=None
import itertools
for perm in itertools.permutations(range(3)):
    c=b[:,perm].copy()
    if np.ptp(c[:,2])<max(np.ptp(c[:,0]),np.ptp(c[:,1])): continue
    for sx,sz in itertools.product([-1,1],repeat=2):
        d=c*np.array([sx,1,sz])
        q=(d-d.min(0))/np.maximum(np.ptp(d,axis=0),1e-9)
        uv=q[:,[0,2]]
        score=float(tree.query(uv)[0].mean()+cKDTree(uv).query(target)[0].mean())
        if best is None or score<best[0]: best=(score,perm,sx,sz,d)
assert best is not None
score,perm,sx,sz,d=best
# Preserve the candidate depth scale relative to height, align center.
d=(d-(d.min(0)+d.max(0))/2)*h/np.ptp(d[:,2])+(lo+hi)/2
# Use the rear envelope; depth sign does not change which outer envelope is sampled.
bv=BVHTree.FromPolygons([Vector(v) for v in d],faces)
orig=BVHTree.FromPolygons([Vector(v) for v in a],[tuple(f.vertices) for f in m.polygons])
delta=np.zeros(len(a))
for i,v in enumerate(a):
    start=Vector((v[0],max(hi[1],d[:,1].max())+h,v[2]))
    r=bv.ray_cast(start,Vector((0,-1,0)))[0]
    t=orig.ray_cast(start,Vector((0,-1,0)))[0]
    if r is None or t is None: continue
    visibility=max(0,1-abs(t.y-v[1])/(.025*h))
    normal=max(0,float(m.vertices[i].normal.y))
    delta[i]=np.clip(r.y-t.y,-.025*h,.08*h)*visibility*normal
# Smooth displacements on existing topology, retaining UV and front geometry.
neighbors=[[] for _ in a]
for e in m.edges:
    i,j=e.vertices;neighbors[i].append(j);neighbors[j].append(i)
active=np.abs(delta)>1e-9
for _ in range(3):
    delta=np.array([.6*delta[i]+.4*np.mean(delta[n]) if active[i] and n else delta[i] for i,n in enumerate(neighbors)])
assert np.max(np.abs(delta))>h*1e-5
for x in objs: bpy.data.objects.remove(x,do_unlink=True)
scene=bpy.context.scene
for x in scene.objects: x.select_set(False)
o.select_set(True);bpy.context.view_layer.objects.active=o
scene.render.engine="CYCLES";scene.cycles.samples=4
scene.render.resolution_x=320;scene.render.resolution_y=480;scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new("Review");scene.world.use_nodes=True
scene.world.node_tree.nodes["Background"].inputs[0].default_value=(.4,.4,.4,1)
scene.view_settings.view_transform="Standard"
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type="ORTHO";cam.data.ortho_scale=h*1.15;scene.camera=cam
center=Vector((lo+hi)/2)
sheet=Image.new("RGB",(320*4,1000),"white");draw=ImageDraw.Draw(sheet)
for stage in range(2):
    if stage:
        for i,v in enumerate(m.vertices): v.co.y=float(a[i,1]+delta[i])
        m.update()
    for i,angle in enumerate((0,90,180,270)):
        r=math.radians(angle);cam.location=center+Vector((math.sin(r)*h*3,-math.cos(r)*h*3,0))
        cam.rotation_euler=(center-cam.location).to_track_quat("-Z","Y").to_euler()
        scene.render.filepath=str(p/f"stage{stage}_{angle}.png")
        bpy.ops.render.render(write_still=True)
        sheet.paste(Image.open(scene.render.filepath).convert("RGB"),(i*320,stage*500))
        draw.text((i*320+5,stage*500+480),f"{'AFTER' if stage else 'BEFORE'} {angle}",fill="black")
sheet.save(p/"geometry_comparison.png")
bpy.data.objects.remove(cam,do_unlink=True)
bpy.ops.export_scene.gltf(filepath=str(p/"xuanyuan_geometry_candidate.glb"),export_format="GLB")
bpy.ops.wm.save_as_mainfile(filepath=str(p/"xuanyuan_geometry_candidate.blend"))
report={"changed_vertices":int(np.count_nonzero(delta)),"max_displacement_relative_height":float(np.max(np.abs(delta))/h),"alignment_score":score,"axis_permutation":perm,"final_avatar":False,"method":"rear envelope deformation; UV preserved; orientation and quality require visual review"}
(p/"geometry_report.json").write_text(json.dumps(report,indent=2))
print(report,flush=True);sys.stdout.flush();os._exit(0)
