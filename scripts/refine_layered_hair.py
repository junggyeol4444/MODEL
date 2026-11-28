"""Layered long-hair revision traced from the original dedicated rear hair view."""
import bpy, math, os, json, pathlib, numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from mathutils import Vector
from PIL import Image, ImageDraw
import bmesh
p=pathlib.Path("output").resolve()
bpy.ops.wm.open_mainfile(filepath=str(p/"base/xuanyuan_hair_candidate.blend"))
core=bpy.data.objects["Hair_Long_Independent"]
body=next(o for o in bpy.context.scene.objects if o.type=="MESH" and o!=core)
a=np.array([tuple(body.matrix_world@v.co) for v in body.data.vertices])
lo=a.min(0);hi=a.max(0);h=hi[2]-lo[2];cx=(lo[0]+hi[0])/2
prev=np.array([tuple(v.co) for v in core.data.vertices])
assert len(prev)==9600
rings=prev.reshape(100,96,3)
# Keep reference-derived root and side depth; smooth discontinuous ray hits.
smooth=rings.copy()
smooth[:,:,1]=gaussian_filter1d(gaussian_filter1d(rings[:,:,1],2.6,axis=0),1.3,axis=1,mode="wrap")
maxy=gaussian_filter1d(smooth[:,:,1].max(1),2)
miny=gaussian_filter1d(smooth[:,:,1].min(1),2)
zs=smooth[:,:,2].mean(1)
# Narrow the under-layer in the lower half so the individually traced tips
# define the silhouette, rather than the previous single pointed mass.
for i in range(100):
    t=i/99
    k=1-.30*np.clip((t-.32)/.68,0,1)
    smooth[i,:,0]=cx+(smooth[i,:,0]-cx)*k
    smooth[i,:,2]+=h*.045*np.clip((t-.65)/.35,0,1)**2
for v,co in zip(core.data.vertices,smooth.reshape(-1,3)): v.co=co
core.data.update();core.name="Hair_Long_Underlayer"
# Traces on original dedicated back-hair panel (450x870 review crop).
# Each tuple is x, y, full strand width in that reference. Curves cross
# at different depth layers rather than being painted onto a flat surface.
traces=[
("Crown_Center",[(245,74,20),(245,113,67),(246,170,70),(250,219,43),(248,261,4)],.015),
("Sweep_L",[(225,79,14),(190,125,47),(177,187,47),(187,255,35),(215,336,30),(253,431,25),(289,535,18),(306,615,1)],.016),
("Sweep_R",[(264,79,12),(298,122,48),(314,185,47),(300,265,38),(275,339,30),(241,425,25),(208,508,16),(190,575,1)],.021),
("Long_L_Cross",[(163,157,17),(156,238,42),(170,332,42),(201,424,44),(235,528,37),(266,634,30),(263,735,23),(230,802,12),(204,833,1)],.023),
("Long_R_Cross",[(328,156,16),(336,240,41),(318,331,43),(293,425,43),(258,535,36),(235,631,29),(256,745,20),(286,811,10),(324,835,1)],.017),
("Outer_L",[(146,188,13),(139,283,23),(148,367,31),(134,461,24),(121,553,17),(133,642,12),(160,690,1)],.008),
("Outer_R",[(343,189,13),(348,284,25),(333,375,31),(355,461,23),(368,550,16),(354,641,10),(335,684,1)],.008),
("Long_L_Inner",[(182,255,21),(178,349,40),(174,449,36),(181,550,29),(201,658,23),(227,731,12),(245,775,1)],.010),
("Long_R_Inner",[(307,250,20),(312,354,37),(312,455,37),(301,550,31),(284,652,23),(261,729,12),(245,776,1)],.011),
("Loose_L",[(152,278,9),(139,338,10),(111,365,7),(90,378,1)],.012),
("Loose_R",[(341,276,9),(357,338,10),(385,365,7),(408,379,1)],.012),
("Fine_L",[(171,370,10),(139,450,9),(114,531,5),(114,572,4),(132,602,1)],.009),
("Fine_R",[(328,365,10),(358,449,9),(386,530,5),(384,573,4),(369,604,1)],.009)]
# Create continuous lofted closed locks with a flattened, asymmetric cross
# section and curved tip. No primitive objects are used.
mat=core.data.materials[0]
locks=[]
xscale=.00082*h/1.8
# Reference panel is a head-only enlarged design; map its vertical span
# to the full-body rear hair trace from the preceding verified model.
z_top=lo[2]+h*.989;z_bottom=lo[2]+h*.513
def zmap(y):return z_top-(y-60)/(835-60)*(z_top-z_bottom)
def depth_at(x,z):
    # Rear envelope in the prior candidate, with a smooth depth falloff
    # toward the lateral margins.
    rear=float(np.interp(z,zs[::-1],maxy[::-1]))
    front=float(np.interp(z,zs[::-1],miny[::-1]))
    return rear-(rear-front)*.26*min(1.6,abs(x-cx)/(.065*h))**1.5
for name,pts,layer in traces:
    pts=np.array(pts,float);t=np.linspace(0,1,56)
    yy=pts[:,1]
    f=PchipInterpolator(yy,pts[:,[0,2]],axis=0)
    py=yy[0]+(yy[-1]-yy[0])*t
    xw=f(py)
    xx=cx+(xw[:,0]-248)*xscale
    zz=zmap(py)
    widths=np.maximum(xw[:,1]*xscale,.0005)
    # Arc tangent allows the lock width to follow the actual sweep.
    dx=np.gradient(xx);dz=np.gradient(zz)
    nn=np.hypot(dx,dz);nx=-dz/nn;nz=dx/nn
    n=12;verts=[];faces=[];uv=[]
    for i,(x,z,w) in enumerate(zip(xx,zz,widths)):
        center_y=depth_at(x,z)+layer*h*.40
        # Root embeds into under-layer; free end curls outward.
        center_y-=h*.011*max(0,1-t[i]*6)
        center_y+=h*.008*t[i]**3
        thickness=max(.0005,w*.24)
        for j in range(n):
            th=2*math.pi*j/n
            lateral=w*.5*math.cos(th)
            # Sculpted central crest on the outward face of each lock.
            back=math.sin(th)
            y=center_y+thickness*.5*back*(1+.23*max(0,back))
            verts.append((float(x+nx[i]*lateral),float(y),float(z+nz[i]*lateral)))
            uv.append((j/n,1-(py[i]-60)/(835-60)))
    for i in range(len(t)-1):
        for j in range(n):
            k=(j+1)%n;faces.append((i*n+j,(i+1)*n+j,(i+1)*n+k,i*n+k))
    faces.extend([tuple(range(n-1,-1,-1)),tuple((len(t)-1)*n+j for j in range(n))])
    m=bpy.data.meshes.new("Hair_"+name+"_Surface");m.from_pydata(verts,[],faces);m.update()
    bm=bmesh.new();bm.from_mesh(m);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(m);bm.free()
    uvlay=m.uv_layers.new(name="HairFlow")
    for face in m.polygons:
        face.use_smooth=True
        ids=[m.loops[l].vertex_index for l in face.loop_indices]
        seam=any(k%n==0 for k in ids) and any(k%n==n-1 for k in ids)
        for l,k in zip(face.loop_indices,ids):
            u,v=uv[k];uvlay.data[l].uv=(1 if seam and u==0 else u,v)
    m.materials.append(mat)
    obj=bpy.data.objects.new("Hair_"+name,m);bpy.context.collection.objects.link(obj);locks.append(obj)
# Render old and new with identical cameras and lighting.
scene=bpy.context.scene;scene.render.engine="CYCLES";scene.cycles.samples=8
scene.render.resolution_x=420;scene.render.resolution_y=620;scene.render.resolution_percentage=100
scene.view_settings.view_transform="Standard"
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type="ORTHO";cam.data.ortho_scale=h*1.15;scene.camera=cam
c=Vector((lo+hi)/2);sheet=Image.new("RGB",(420*4,650*2),"white");draw=ImageDraw.Draw(sheet)
for stage in range(2):
    for v,co in zip(core.data.vertices,prev if stage==0 else smooth.reshape(-1,3)):v.co=co
    core.data.update()
    for obj in locks:obj.hide_render=not bool(stage)
    for j,ang in enumerate((0,90,180,270)):
        r=math.radians(ang);cam.location=c+Vector((math.sin(r)*h*3,-math.cos(r)*h*3,0));cam.rotation_euler=(c-cam.location).to_track_quat("-Z","Y").to_euler()
        path=p/f"layered_review_{stage}_{ang}.png";scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        sheet.paste(Image.open(path).convert("RGB"),(420*j,650*stage));draw.text((420*j+6,650*stage+625),f"{'LAYERED CANDIDATE' if stage else 'PREVIOUS'} {ang}",fill="black")
sheet.save(p/"layered_hair_comparison.png")
bpy.data.objects.remove(cam,do_unlink=True)
for o in scene.objects:o.select_set(False)
for o in [body,core]+locks:o.select_set(True)
bpy.context.view_layer.objects.active=core
bpy.ops.export_scene.gltf(filepath=str(p/"xuanyuan_layered_hair_candidate.glb"),export_format="GLB",use_selection=True)
bpy.ops.wm.save_as_mainfile(filepath=str(p/"xuanyuan_layered_hair_candidate.blend"))
bad=0
for o in locks:
    bm=bmesh.new();bm.from_mesh(o.data);bad+=sum(not e.is_manifold for e in bm.edges);bm.free()
report={"source_run":46111630780,"locks":[o.name for o in locks],"lock_count":len(locks),"lock_vertices":sum(len(o.data.vertices) for o in locks),"nonmanifold_lock_edges":bad,"reference":"traced dedicated rear hair panel of original reference","limitations":["original painted hair underneath remains","manual collision and weight checks still needed","not final avatar"],"final_avatar":False}
(p/"layered_hair_report.json").write_text(json.dumps(report,indent=2));print(report,flush=True);os._exit(0)
