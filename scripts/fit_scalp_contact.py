import bpy, numpy as np, pathlib, math, json
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
p=pathlib.Path("output").resolve()
bpy.ops.wm.open_mainfile(filepath=str(p/"base/xuanyuan_root_candidate.blend"))
core=bpy.data.objects["Hair_Long_Underlayer"]
body=next(o for o in bpy.context.scene.objects if o.type=="MESH" and not o.name.startswith("Hair_"))
a=np.array([tuple(body.matrix_world@v.co) for v in body.data.vertices]);lo=a.min(0);hi=a.max(0);h=hi[2]-lo[2]
bv=BVHTree.FromPolygons([Vector(v) for v in a],[tuple(f.vertices) for f in body.data.polygons])
hair=[o for o in bpy.context.scene.objects if o.type=="MESH" and o.name.startswith("Hair_")]
old={o.name:np.array([tuple(v.co) for v in o.data.vertices]) for o in hair}
new={}
# Attach the inner root surface to the actual scalp, not just its centroid.
for o in hair:
    n=96 if o==core else 12
    r=old[o.name].reshape(-1,n,3).copy()
    original=r.copy()
    for i in range(len(r)):
        t=i/(len(r)-1)
        if t>.32:break
        w=(1-t/.32)**2
        for j in range(n):
            x,y,z=r[i,j]
            hit=bv.ray_cast(Vector((x,hi[1]+h,z)),Vector((0,-1,0)))[0]
            if hit is None:continue
            # Move floating roots inward while retaining the sculpted outer crest.
            target=hit.y-h*.004+h*.008*max(0,math.sin(2*math.pi*j/n))
            r[i,j,1]+=min(0,target-y)*w
        if i==0:
            center=r[i].mean(0)
            r[i,:,0]=center[0]+(r[i,:,0]-center[0])*.75
    r[:,:,1]=gaussian_filter1d(r[:,:,1],.65,axis=0)
    new[o.name]=r.reshape(-1,3)
scene=bpy.context.scene;scene.render.engine="CYCLES";scene.cycles.samples=12
scene.render.resolution_x=480;scene.render.resolution_y=480;scene.render.resolution_percentage=100
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type="ORTHO";cam.data.ortho_scale=h*.37;scene.camera=cam
c=Vector(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,hi[2]-h*.14))
sheet=Image.new("RGB",(1920,1000),"white");draw=ImageDraw.Draw(sheet)
for stage in range(2):
    for o in hair:
        for v,co in zip(o.data.vertices,(new if stage else old)[o.name]):v.co=co
        o.data.update()
    for j,ang in enumerate((0,90,180,270)):
        r=math.radians(ang);cam.location=c+Vector((math.sin(r)*h*3,-math.cos(r)*h*3,0));cam.rotation_euler=(c-cam.location).to_track_quat("-Z","Y").to_euler()
        path=p/f"root_{stage}_{ang}.png";scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        sheet.paste(Image.open(path).convert("RGB"),(480*j,500*stage));draw.text((480*j+5,500*stage+483),f"{'AFTER' if stage else 'BEFORE'} {ang}",fill="black")
sheet.save(p/"scalp_fit_comparison.png")
bpy.data.objects.remove(cam,do_unlink=True)
for o in scene.objects:o.select_set(False)
for o in [body]+hair:o.select_set(True)
bpy.context.view_layer.objects.active=core
bpy.ops.export_scene.gltf(filepath=str(p/"xuanyuan_scalp_fit_candidate.glb"),export_format="GLB",use_selection=True)
bpy.ops.wm.save_as_mainfile(filepath=str(p/"xuanyuan_scalp_fit_candidate.blend"))
report={"source_run":46118625905,"changed_vertices":sum(int(np.count_nonzero(np.linalg.norm(new[k]-old[k],axis=1)>1e-6)) for k in old),"hair_objects":len(hair),"final_avatar":False}
(p/"scalp_fit_report.json").write_text(json.dumps(report));print(report,flush=True)
