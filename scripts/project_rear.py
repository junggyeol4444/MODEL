import bpy, pathlib, math, json, sys, os
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from PIL import Image, ImageDraw
p=pathlib.Path("output").resolve()
bpy.ops.wm.open_mainfile(filepath=str(p/"xuanyuan_projected.blend"))
scene=bpy.context.scene
mesh=[o for o in scene.objects if o.type=="MESH"]
assert len(mesh)==1, "Expected one reconstruction"
o=mesh[0]
bpy.context.view_layer.objects.active=o
o.select_set(True)
bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
bpy.context.view_layer.update()
m=o.data
coords=[v.co.copy() for v in m.vertices]
mn=Vector(tuple(min(v[i] for v in coords) for i in range(3)))
mx=Vector(tuple(max(v[i] for v in coords) for i in range(3)))
assert mx.z-mn.z>mx.x-mn.x and mx.z-mn.z>mx.y-mn.y
original_uv=m.uv_layers.active.name
uv=m.uv_layers.new(name="RearReference")
attr=m.color_attributes.new(name="RearVisibility",type="FLOAT_COLOR",domain="POINT")
tree=BVHTree.FromPolygons(coords,[tuple(poly.vertices) for poly in m.polygons])
def smooth(a,b,t):
    x=max(0,min(1,(t-a)/(b-a)))
    return x*x*(3-2*x)
visible=0
for v in m.vertices:
    # First hit from the front prevents projecting onto hidden back surfaces.
    pos=v.co
    hit,_,_,_=tree.ray_cast(Vector((pos.x,mx.y+1,pos.z)),Vector((0,-1,0)))
    depth=abs(hit.y-pos.y) if hit is not None else 1
    weight=smooth(.15,.75,v.normal.y)*(1-smooth(.004,.035,depth))
    if weight>.25: visible+=1
    attr.data[v.index].color=(weight,weight,weight,1)
for poly in m.polygons:
    for li in poly.loop_indices:
        v=coords[m.loops[li].vertex_index]
        uv.data[li].uv=((mx.x-v.x)/(mx.x-mn.x),(v.z-mn.z)/(mx.z-mn.z))
m.uv_layers.active=m.uv_layers[original_uv]
m.uv_layers[original_uv].active_render=True
mat=m.materials[0]
nodes=mat.node_tree.nodes
links=mat.node_tree.links
base=next(n for n in nodes if n.type=="TEX_IMAGE")
bs=nodes.get("Principled BSDF")
uv0=nodes.new("ShaderNodeUVMap");uv0.uv_map=original_uv
links.new(uv0.outputs[0],base.inputs["Vector"])
uv1=nodes.new("ShaderNodeUVMap");uv1.uv_map=uv.name
ref=nodes.new("ShaderNodeTexImage")
ref.image=bpy.data.images.load(str(p/"rear_reference.png"))
ref.extension="CLIP"
links.new(uv1.outputs[0],ref.inputs["Vector"])
color=nodes.new("ShaderNodeVertexColor");color.layer_name=attr.name
mul=nodes.new("ShaderNodeMath");mul.operation="MULTIPLY"
links.new(color.outputs["Color"],mul.inputs[0]);links.new(ref.outputs["Alpha"],mul.inputs[1])
mix=nodes.new("ShaderNodeMixRGB");mix.blend_type="MIX"
links.new(mul.outputs[0],mix.inputs[0]);links.new(base.outputs["Color"],mix.inputs[1]);links.new(ref.outputs["Color"],mix.inputs[2])
em=nodes.new("ShaderNodeEmission");links.new(mix.outputs[0],em.inputs["Color"])
output=next(n for n in nodes if n.type=="OUTPUT_MATERIAL");links.new(em.outputs[0],output.inputs["Surface"])
baked=bpy.data.images.new("ProjectedTexture",width=1024,height=1024,alpha=True)
target=nodes.new("ShaderNodeTexImage");target.image=baked
for n in nodes:n.select=False
target.select=True;nodes.active=target
scene.render.engine="CYCLES";scene.cycles.device="CPU";scene.cycles.samples=8
scene.render.bake.margin=8
bpy.ops.object.bake(type="EMIT")
baked.filepath_raw=str(p/"rear_projected_texture.png");baked.file_format="PNG";baked.save();baked.pack()
# Bake the result into a normal exportable single-texture material.
new=bpy.data.materials.new("Projected reference plus reconstructed texture");new.use_nodes=True
nt=new.node_tree
tex=nt.nodes.new("ShaderNodeTexImage");tex.image=baked
u=nt.nodes.new("ShaderNodeUVMap");u.uv_map=original_uv
nt.links.new(u.outputs[0],tex.inputs["Vector"])
nt.links.new(tex.outputs["Color"],nt.nodes.get("Principled BSDF").inputs["Base Color"])
nt.nodes.get("Principled BSDF").inputs["Roughness"].default_value=.8
m.materials.clear();m.materials.append(new)
m.uv_layers.remove(m.uv_layers["RearReference"])
m.uv_layers.active=m.uv_layers[original_uv]
m.color_attributes.remove(m.color_attributes["RearVisibility"])
bpy.ops.export_scene.gltf(filepath=str(p/"xuanyuan_multiview.glb"),export_format="GLB")
bpy.ops.wm.save_as_mainfile(filepath=str(p/"xuanyuan_multiview.blend"))
# Render actual model from four directions, using unlit texture for inspection.
bs2=nt.nodes.get("Principled BSDF")
em2=nt.nodes.new("ShaderNodeEmission");nt.links.new(tex.outputs["Color"],em2.inputs["Color"])
nt.links.new(em2.outputs[0],nt.nodes.get("Material Output").inputs["Surface"])
center=(mn+mx)/2
size=mx.z-mn.z
scene.render.resolution_x=420;scene.render.resolution_y=640;scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new("Neutral background");scene.world.use_nodes=True
scene.world.node_tree.nodes["Background"].inputs[0].default_value=(.35,.35,.35,1)
scene.view_settings.view_transform="Standard"
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type="ORTHO";cam.data.ortho_scale=size*1.1;scene.camera=cam
sheet=Image.new("RGB",(420*5,675),"white")
image=Image.open(p/"xuanyuan_back.webp").convert("RGB");image.thumbnail((420,640))
sheet.paste(image,((420-image.width)//2,(640-image.height)//2));d=ImageDraw.Draw(sheet);d.text((20,650),"REFERENCE",fill="black")
for i,angle in enumerate((0,90,180,270)):
    rad=math.radians(angle)
    cam.location=center+Vector((math.sin(rad)*size*3,-math.cos(rad)*size*3,0))
    cam.rotation_euler=(center-cam.location).to_track_quat("-Z","Y").to_euler()
    scene.render.filepath=str(p/f"projected_view_{i}.png")
    bpy.ops.render.render(write_still=True)
    sheet.paste(Image.open(scene.render.filepath).convert("RGB"),((i+1)*420,0));d.text(((i+1)*420+20,650),f"MODEL {angle}",fill="black")
sheet.save(p/"rear_comparison.png")
report={"vertices":len(m.vertices),"faces":len(m.polygons),"projected_rear_vertices":visible,"uv_baked":True,"rigged":False,"reference":"142x320 crop from user back reference","geometry_changes":"texture projection only; rear shape evaluated separately"}
(p/"rear_report.json").write_text(json.dumps(report,indent=2))
print("PROJECTION COMPLETE",report,flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0)
