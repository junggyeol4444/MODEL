"""Reference-contour guided independent long hair surface. Candidate, not final."""
import bpy, math, numpy as np, pathlib, json, os
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageDraw
p=pathlib.Path("output").resolve()
bpy.ops.wm.open_mainfile(filepath=str(p/"base/xuanyuan_side_repair.blend"))
body=next(o for o in bpy.context.scene.objects if o.type=="MESH")
a=np.array([tuple(body.matrix_world@v.co) for v in body.data.vertices])
lo=a.min(0); hi=a.max(0); h=hi[2]-lo[2]; cx=(lo[0]+hi[0])/2
assert h>1 and h>max(hi[:2]-lo[:2])
# Manually traced from full-size original 6441cdb1... reference, using the
# 1408x1056 review frame (all values normalized by body height 657px).
# y_px, left/right edges in rear view, outer depth in side view.
# These outline points are not the body silhouette.
trace=np.array([
[35,682,714,470],[53,666,732,484],[78,661,738,491],
[105,672,730,498],[135,666,735,511],[165,663,736,518],
[195,665,736,525],[225,673,732,530],[255,675,729,528],
[280,679,724,525],[305,686,716,518],[325,691,710,512],
[338,698,703,497]],float)
interp=PchipInterpolator(trace[:,0],trace[:,1:],axis=0)
bv=BVHTree.FromPolygons([Vector(v) for v in a],[tuple(f.vertices) for f in body.data.polygons])
def surface(x,z):
    q=bv.ray_cast(Vector((x,hi[1]+h,z)),Vector((0,-1,0)))[0]
    return q.y if q is not None else None
# Align depth at root to existing head, then follow traced side outline.
rootz=lo[2]+h*(1-(53-28)/657)
rootback=surface(cx,rootz)
assert rootback is not None
depth_offset=rootback-(484-437)*h/657
nr=100; ns=96
vertices=[]; uv=[]; faces=[]
for i in range(nr):
    t=i/(nr-1); py=35+(338-35)*t
    left,right,side=interp(py)
    z=lo[2]+h*(1-(py-28)/657)
    width=(right-left)*h/657
    center=cx+((left+right)/2-700)*h/657
    # Open root is embedded in the original scalp; below the shoulders
    # the inner surface clears the original torso. Thickness tapers at tip.
    rear_target=depth_offset+(side-437)*h/657
    fade=min(1,max(0,(py-90)/90))
    for j in range(ns):
        theta=2*math.pi*j/ns
        # The cross-section is a single continuous sculpted surface.
        # Smooth longitudinal ridges and asymmetric tapered ends follow hair flow.
        u=math.cos(theta); back=max(0,math.sin(theta))
        flow=theta+.34*math.sin(t*4.5)+.19*t
        ridge=(.65*math.cos(11*flow)+.35*math.cos(19*flow+1.4))*h*.0025
        x=center+width*.5*u*(1+.035*math.sin(9*flow))
        local_z=z
        if t>.69:
            local_z+=h*.011*((t-.69)/.31)**2*(.6*math.sin(5*theta)+.4*math.sin(9*theta+.5))
        hit=surface(x,local_z)
        if hit is None: hit=surface(cx,local_z)
        if hit is None: hit=rootback
        inside=hit+h*(-.018*(1-fade)+.006*fade)
        outer=max(rear_target,inside+h*(.027*(1-t)+.009))
        # Oval section with thinner lateral margins; volume is not a flat card.
        y=inside+(outer-inside)*(.5+.5*math.sin(theta))
        y+=ridge*back*(math.sin(math.pi*t)**.6)
        vertices.append((float(x),float(y),float(local_z)))
        uv.append((j/ns,1-t))
for i in range(nr-1):
    for j in range(ns):
        k=(j+1)%ns
        faces.append((i*ns+j,(i+1)*ns+j,(i+1)*ns+k,i*ns+k))
faces.extend([tuple(range(ns-1,-1,-1)),tuple((nr-1)*ns+j for j in range(ns))])
mesh=bpy.data.meshes.new("Xuanyuan_Hair_Long_Surface")
mesh.from_pydata(vertices,[],faces);mesh.update()
hair=bpy.data.objects.new("Hair_Long_Independent",mesh);bpy.context.collection.objects.link(hair)
# Recalculate outward normals and smooth across the continuous surface.
import bmesh
bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
for f in mesh.polygons:f.use_smooth=True
layer=mesh.uv_layers.new(name="HairFlow")
for f in mesh.polygons:
    ids=[mesh.loops[l].vertex_index for l in f.loop_indices]
    seam=any(k%ns==0 for k in ids) and any(k%ns==ns-1 for k in ids)
    for l,k in zip(f.loop_indices,ids):
        u,v=uv[k]
        if seam and u==0:u=1.
        layer.data[l].uv=(u,v)
# Base color sampled from repository rear reference copy (not claimed original).
# Longitudinal highlight variation is built in UV space so the asset exports.
ref=Image.open("references/xuanyuan_back.webp").convert("RGB")
arr=np.asarray(ref)/255.; rh,rw=arr.shape[:2]
patch=arr[int(rh*.10):int(rh*.38),int(rw*.45):int(rw*.55)]
dark=patch[(patch.max(2)<.40)&(patch.min(2)>.035)]
root=np.median(dark,axis=0) if len(dark)>10 else np.array([.13,.14,.12])
tip=np.array([47,63,51])/255.
sz=512
U,V=np.meshgrid(np.linspace(0,1,sz),np.linspace(0,1,sz))
flow=2*np.pi*U+.34*np.sin(V*4.5)+.19*V
streak=.022*np.cos(11*flow)+.012*np.cos(19*flow+1.4)+.009*np.cos(53*flow+2*V)
f=np.clip((V-.35)/.65,0,1)[...,None]
rgb=np.clip(root*(1-f)+tip*f+streak[...,None],0,1)
tex=p/"hair_flow_texture.png"; Image.fromarray(np.uint8(rgb*255)).save(tex)
mat=bpy.data.materials.new("Hair_Reference_Black_Green");mat.use_nodes=True
nodes=mat.node_tree.nodes;links=mat.node_tree.links
bs=nodes.get("Principled BSDF");bs.inputs["Roughness"].default_value=.48
im=nodes.new("ShaderNodeTexImage");im.image=bpy.data.images.load(str(tex));im.image.pack()
links.new(im.outputs["Color"],bs.inputs["Base Color"]);mesh.materials.append(mat)
# Preserve the original mesh unchanged. The painted hair underneath remains
# a separate unresolved texture-cleanup task.
scene=bpy.context.scene; scene.render.engine="CYCLES";scene.cycles.samples=8
scene.render.resolution_x=420;scene.render.resolution_y=620;scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new("Hair review");scene.world.use_nodes=True
scene.world.node_tree.nodes["Background"].inputs[0].default_value=(.4,.4,.4,1)
scene.view_settings.view_transform="Standard"
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type="ORTHO";cam.data.ortho_scale=h*1.15;scene.camera=cam
c=Vector((lo+hi)/2)
sheet=Image.new("RGB",(420*4,650*2),"white"); draw=ImageDraw.Draw(sheet)
for stage in range(2):
    hair.hide_render=not bool(stage)
    for j,ang in enumerate((0,90,180,270)):
        r=math.radians(ang);cam.location=c+Vector((math.sin(r)*h*3,-math.cos(r)*h*3,0));cam.rotation_euler=(c-cam.location).to_track_quat("-Z","Y").to_euler()
        scene.render.filepath=str(p/f"hair_review_{stage}_{ang}.png");bpy.ops.render.render(write_still=True)
        sheet.paste(Image.open(scene.render.filepath).convert("RGB"),(420*j,650*stage));draw.text((420*j+6,650*stage+625),f"{'HAIR CANDIDATE' if stage else 'PREVIOUS'} {ang}",fill="black")
sheet.save(p/"hair_comparison.png")
hair.hide_render=False
bpy.data.objects.remove(cam,do_unlink=True)
for o in scene.objects:o.select_set(False)
body.select_set(True);hair.select_set(True);bpy.context.view_layer.objects.active=hair
bpy.ops.export_scene.gltf(filepath=str(p/"xuanyuan_hair_candidate.glb"),export_format="GLB",use_selection=True)
bpy.ops.wm.save_as_mainfile(filepath=str(p/"xuanyuan_hair_candidate.blend"))
bm=bmesh.new();bm.from_mesh(mesh)
report={"source_run":46001231698,"independent_hair_object":hair.name,"hair_vertices":len(mesh.vertices),"hair_faces":len(mesh.polygons),"nonmanifold_edges":sum(not e.is_manifold for e in bm.edges),"body_vertices_unchanged":len(body.data.vertices)==len(a),"reference":"original side and rear traced outline; compressed rear color sample","limitations":["painted underlying hair not removed","individual layered strands and rigging not validated","not final avatar"],"final_avatar":False}
bm.free();(p/"hair_report.json").write_text(json.dumps(report,indent=2));print(report,flush=True);os._exit(0)
