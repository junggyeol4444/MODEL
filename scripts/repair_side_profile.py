import bpy, numpy as np, pathlib, json, math, os, sys
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from mathutils import Vector
from PIL import Image, ImageDraw
p=pathlib.Path("output").resolve()
# Side silhouette sampled from recovered original 1448x1086 PNG, not the repository WebP.
# SHA256 6ec1e94481db80e88577f89f9ae40b217689568f1021c38561d5ed191aa27029
# Coordinates in the 1408x1056 review frame, height 657 px. Face points left.
rows=np.array([
[.033,386,473.6],[.067,426.9,467.7],[.1,431.7,465.8],[.133,429.8,467.7],
[.167,377.3,539.7],[.2,379.2,534.8],[.233,381.2,529.9],[.267,382.1,525.1],
[.3,384.1,519.3],[.333,386,514.4],[.367,387,509.5],[.4,389,504.7],
[.433,389.9,498.8],[.467,391.9,494],[.5,393.8,489.1],[.533,394.8,513.4],
[.567,396.7,522.2],[.6,398.7,524.1],[.633,398.7,530.9],[.667,399.6,533.8],
[.7,398.7,529],[.733,397.7,524.1],[.767,396.7,518.3],[.8,406.5,514.4],
[.833,424,506.6],[.867,429.8,500.8],[.9,412.3,488.1],[.933,392.8,489.1],[.967,402.6,479.4]])
def load(path):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    o=next(o for o in bpy.context.scene.objects if o.type=="MESH")
    o.data.calc_loop_triangles()
    return o,np.array([tuple(v.co) for v in o.data.vertices]),np.array([tuple(t.vertices) for t in o.data.loop_triangles])
o,old,tri_old=load(p/"candidate/xuanyuan_geometry_candidate.blend")
o,a,tri=load(p/"base/xuanyuan_multiview.blend")
assert a.shape==old.shape and np.array_equal(tri,tri_old), "candidate topology mismatch"
lo=a.min(0);hi=a.max(0);h=hi[2]-lo[2]
z=(a[:,2]-lo[2])/h
# Preserve UVs and topology. Reject the visually damaged pointwise rear warp.
# Fit a continuous positive-Jacobian affine depth map at each height.
g=np.linspace(.025,.975,77)
bounds=[]
for t in g:
    ids=np.abs(z-t)<.018
    if ids.sum()<15: ids=np.abs(z-t)<.035
    if ids.sum()<3: ids=np.argsort(np.abs(z-t))[:20]
    bounds.append(np.quantile(a[ids,1],[.015,.985]))
bounds=np.array(bounds)
front,back=gaussian_filter1d(bounds,1.5,axis=0).T
ref=PchipInterpolator(rows[:,0],rows[:,1:],axis=0)(np.clip(g,rows[0,0],rows[-1,0]))
ref=gaussian_filter1d(ref,1.5,axis=0)
target=(ref-450)/657*h
# Align reference center using torso, not an arbitrary importer depth offset.
offset=np.median(((front+back)/2-target.mean(1))[(g>.35)&(g<.52)])
target+=offset
scale=np.clip((target[:,1]-target[:,0])/np.maximum(back-front,.025*h),.7,1.45)
# Apply 70% of the reference correction; fade at top/bottom to avoid boot/crown spikes.
weight=.7*np.clip((g-.12)/.08,0,1)*np.clip((.99-g)/.06,0,1)
s=1+weight*(scale-1)
center=(front+back)/2
shift=np.clip(target.mean(1)-center,-.055*h,.055*h)*weight
b=a.copy()
b[:,1]=(a[:,1]-np.interp(z,g,center))*np.interp(z,g,s)+np.interp(z,g,center+shift)
assert np.isfinite(b).all()
def normals(v):
    return np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
n=normals(a);valid=np.linalg.norm(n,axis=1)>1e-12
def metrics(v):
    q=normals(v); ratio=np.linalg.norm(q[valid],axis=1)/np.linalg.norm(n[valid],axis=1)
    return {"normal_reversals":int(((q[valid]*n[valid]).sum(1)<0).sum()),"area_ratio_below_0_1":int((ratio<.1).sum()),"changed_vertices":int((np.linalg.norm(v-a,axis=1)>1e-8).sum())}
old_metrics=metrics(old);new_metrics=metrics(b)
assert new_metrics["normal_reversals"]<=old_metrics["normal_reversals"]
assert new_metrics["changed_vertices"]>0
scene=bpy.context.scene
scene.render.engine="CYCLES";scene.cycles.samples=8
scene.render.resolution_x=420;scene.render.resolution_y=620;scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new("Side review");scene.world.use_nodes=True
scene.world.node_tree.nodes["Background"].inputs[0].default_value=(.4,.4,.4,1)
scene.view_settings.view_transform="Standard"
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type="ORTHO";cam.data.ortho_scale=h*1.15;scene.camera=cam
c=Vector((lo+hi)/2); sheet=Image.new("RGB",(420*4,650*2),"white");draw=ImageDraw.Draw(sheet)
for i,coords in enumerate([old,b]):
    for v,co in zip(o.data.vertices,coords): v.co=co
    o.data.update()
    for j,ang in enumerate((0,90,180,270)):
        r=math.radians(ang);cam.location=c+Vector((math.sin(r)*h*3,-math.cos(r)*h*3,0));cam.rotation_euler=(c-cam.location).to_track_quat("-Z","Y").to_euler()
        scene.render.filepath=str(p/f"side_review_{i}_{ang}.png");bpy.ops.render.render(write_still=True)
        sheet.paste(Image.open(scene.render.filepath).convert("RGB"),(420*j,650*i));draw.text((420*j+6,650*i+625),f"{'PREVIOUS' if i==0 else 'REPAIRED'} {ang}",fill="black")
sheet.save(p/"side_repair_comparison.png")
bpy.data.objects.remove(cam,do_unlink=True)
bpy.ops.export_scene.gltf(filepath=str(p/"xuanyuan_side_repair.glb"),export_format="GLB")
bpy.ops.wm.save_as_mainfile(filepath=str(p/"xuanyuan_side_repair.blend"))
report={"previous":old_metrics,"repaired":new_metrics,"reference_sha256":"6ec1e94481db80e88577f89f9ae40b217689568f1021c38561d5ed191aa27029","max_depth_shift_over_height":float(np.max(np.abs(b[:,1]-a[:,1]))/h),"final_avatar":False}
(p/"side_repair_report.json").write_text(json.dumps(report,indent=2));print(report,flush=True);os._exit(0)
