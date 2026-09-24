extends Node3D
## 3D 战场渲染层（数据驱动，可无状态重放任意一帧）。
##
## 坐标映射：仿真平面 -> 战场中心平移到原点，y 轴为高度。
## 单位/墓碑按帧数据摆放；播放时在相邻两帧间插值让移动平滑；
## 特效（尾迹/激光/爆炸）只跟随"进入的那一帧"的事件，
## 因此拖动进度条、倒放都不会残留错误状态。

const COL_BLUE := Color(0.30, 0.58, 1.0)
const COL_RED := Color(1.0, 0.42, 0.34)
const COL_GROUND := Color(0.10, 0.12, 0.17)
const COL_BORDER := Color(0.30, 0.52, 0.95)
const COL_GRAVE := Color(0.16, 0.17, 0.21)

const TRAIL_STEPS := 48            # 尾迹覆盖的步数
const TRAIL_MAX_POINTS := 24       # 每条尾迹最多采样点数（自动抽稀）
const TRAIL_REBUILD_MIN_GAP := 0.03 # 尾迹重建最小间隔（秒），高速播放时限流
const LASER_MAX_AGE := 0.35        # 命中激光淡出时长（秒）
const FX_JUMP_GAP := 2             # 帧间隔超过该值视为拖动跳转，不显示特效
const LASER_POOL := 40
const LASER_LIGHTS := 6
const BOOM_POOL := 8

const OrbitCamera := preload("res://orbit_camera.gd")

var frames: Array = []
var world := Vector2(1000.0, 600.0)
var frame_idx := -1
var playing := false

var _progress := 0.0     # 相邻两帧之间的插值进度 [0,1)
var _prev_frame := -1

var _unit_root := {}     # id -> Node3D（球体 + 炮管 + HP 环）
var _unit_ring := {}     # id -> StandardMaterial3D（HP 环材质）
var _grave := {}         # id -> MeshInstance3D（阵亡标记）
var _id_index := {}      # id -> 帧内下标（单位排列整局稳定，由 recorder 保证）
var _unit_ids: Array = []
var _arena_nodes: Array = []

var _trails_imm: ImmediateMesh
var _trails_mat: StandardMaterial3D
var _trail_timer := 99.0
var _lasers: Array = []   # {node, mat, light, col, active}
var _laser_age := 99.0
var _booms: Array = []    # CPUParticles3D 池


func _ready() -> void:
	_build_environment()
	var cam := OrbitCamera.new()
	cam.target = Vector3.ZERO
	add_child(cam)
	_build_fx()


func setup(frames_: Array, world_: Vector2) -> void:
	_clear_arena()
	_clear_fx()
	if _trails_imm != null:
		_trails_imm.clear_surfaces()
	frames = frames_
	world = world_
	frame_idx = -1
	_prev_frame = -1
	_progress = 0.0
	_build_arena()
	_build_units(frames[0]["units"])


func set_playing(p: bool) -> void:
	if p and not playing:
		_prev_frame = frame_idx
	playing = p


func set_progress(f: float) -> void:
	_progress = clampf(f, 0.0, 1.0)


func set_frame(i: int, jump: bool = false) -> void:
	if frames.is_empty() or i == frame_idx:
		return
	var prev := frame_idx
	_prev_frame = i if jump else prev
	frame_idx = i
	_laser_age = 0.0
	_clear_fx()
	if not jump and absi(i - prev) <= FX_JUMP_GAP:
		for e in frames[i].get("events", []):
			var t := str(e.get("type", ""))
			if t == "hit":
				_spawn_laser(e)
			elif t == "destroyed":
				_spawn_boom(e)
	_trail_timer = 0.0
	_rebuild_trails()


func _process(delta: float) -> void:
	_trail_timer += delta
	_laser_age += delta
	_update_lasers()
	if frame_idx < 0 or frames.is_empty():
		return
	if not playing or _progress <= 0.0 or _prev_frame < 0 or _prev_frame == frame_idx:
		_apply_frame(frame_idx)
	else:
		_apply_interp(_prev_frame, frame_idx, _progress)


# ------------------------------------------------------------------
# 场景搭建
# ------------------------------------------------------------------

func _build_environment() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.045, 0.055, 0.085)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.6, 0.66, 0.78)
	env.ambient_light_energy = 0.6
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.glow_enabled = true
	env.glow_intensity = 0.5
	env.glow_bloom = 0.05
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-52.0, -35.0, 0.0)
	sun.light_energy = 1.15
	sun.shadow_enabled = true
	add_child(sun)


func _build_arena() -> void:
	var ground := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(world.x, world.y)
	ground.mesh = pm
	var gmat := StandardMaterial3D.new()
	gmat.albedo_color = COL_GROUND
	gmat.roughness = 0.95
	ground.material_override = gmat
	add_child(ground)
	_arena_nodes.append(ground)

	# 四条自发光边界，勾出战场轮廓
	var bmat := StandardMaterial3D.new()
	bmat.albedo_color = Color(0.08, 0.10, 0.15)
	bmat.emission_enabled = true
	bmat.emission = COL_BORDER
	var hx := world.x * 0.5
	var hz := world.y * 0.5
	var specs := [
		[Vector3(world.x + 8.0, 2.4, 2.0), Vector3(0, 1.2, -hz - 1.0)],
		[Vector3(world.x + 8.0, 2.4, 2.0), Vector3(0, 1.2, hz + 1.0)],
		[Vector3(2.0, 2.4, world.y + 8.0), Vector3(-hx - 1.0, 1.2, 0)],
		[Vector3(2.0, 2.4, world.y + 8.0), Vector3(hx + 1.0, 1.2, 0)],
	]
	for s in specs:
		var mi := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = s[0]
		mi.mesh = bm
		mi.material_override = bmat
		mi.position = s[1]
		add_child(mi)
		_arena_nodes.append(mi)


func _build_units(first: Array) -> void:
	var sphere := SphereMesh.new()
	sphere.radius = 6.0
	sphere.height = 12.0
	var barrel := BoxMesh.new()
	barrel.size = Vector3(15.0, 2.4, 2.4)
	var ring := TorusMesh.new()
	ring.inner_radius = 8.5
	ring.outer_radius = 10.5
	var grave_mesh := CylinderMesh.new()
	grave_mesh.height = 1.4
	grave_mesh.top_radius = 4.2
	grave_mesh.bottom_radius = 5.4
	var grave_mat := StandardMaterial3D.new()
	grave_mat.albedo_color = COL_GRAVE
	grave_mat.roughness = 1.0
	var team_mats := [_unit_material(COL_BLUE), _unit_material(COL_RED)]

	for i in first.size():
		var u: Dictionary = first[i]
		var id := str(u["id"])
		_id_index[id] = i
		_unit_ids.append(id)
		var mat: StandardMaterial3D = team_mats[int(u["team"])]

		var root := Node3D.new()
		var body := MeshInstance3D.new()
		body.mesh = sphere
		body.material_override = mat
		body.position = Vector3(0, 7.2, 0)
		root.add_child(body)
		var gun := MeshInstance3D.new()
		gun.mesh = barrel
		gun.material_override = mat
		gun.position = Vector3(9.5, 7.2, 0)
		root.add_child(gun)
		var r := MeshInstance3D.new()
		r.mesh = ring
		var rmat := StandardMaterial3D.new()
		rmat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		r.material_override = rmat
		r.position = Vector3(0, 0.7, 0)
		root.add_child(r)
		_unit_ring[id] = rmat
		add_child(root)
		_unit_root[id] = root

		var g := MeshInstance3D.new()
		g.mesh = grave_mesh
		g.material_override = grave_mat
		g.visible = false
		add_child(g)
		_grave[id] = g


func _unit_material(col: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = col.darkened(0.25)
	m.metallic = 0.25
	m.roughness = 0.55
	m.emission_enabled = true
	m.emission = col * 0.35
	return m


func _clear_arena() -> void:
	for id in _unit_root:
		_unit_root[id].queue_free()
	for id in _grave:
		_grave[id].queue_free()
	for n in _arena_nodes:
		n.queue_free()
	_unit_root.clear()
	_unit_ring.clear()
	_grave.clear()
	_id_index.clear()
	_unit_ids.clear()
	_arena_nodes.clear()


# ------------------------------------------------------------------
# 逐帧摆位
# ------------------------------------------------------------------

func _apply_frame(i: int) -> void:
	var units: Array = frames[i]["units"]
	for id in _unit_ids:
		var u: Dictionary = units[_id_index[id]]
		var root: Node3D = _unit_root[id]
		var grave: MeshInstance3D = _grave[id]
		if bool(u["alive"]):
			root.visible = true
			root.position = _to3(float(u["x"]), float(u["y"]))
			root.rotation.y = -float(u["heading"])
			grave.visible = false
			_set_ring(id, float(u["hp"]) / maxf(1.0, float(u["max_hp"])))
		else:
			root.visible = false
			grave.visible = true
			grave.position = _to3(float(u["x"]), float(u["y"])) + Vector3(0, 0.7, 0)


func _apply_interp(a_idx: int, b_idx: int, t: float) -> void:
	var ua: Array = frames[a_idx]["units"]
	var ub: Array = frames[b_idx]["units"]
	for id in _unit_ids:
		var k: int = _id_index[id]
		var u0: Dictionary = ua[k]
		var u1: Dictionary = ub[k]
		var root: Node3D = _unit_root[id]
		var grave: MeshInstance3D = _grave[id]
		if not bool(u1["alive"]):
			root.visible = false
			grave.visible = true
			grave.position = _to3(float(u1["x"]), float(u1["y"])) + Vector3(0, 0.7, 0)
			continue
		grave.visible = false
		root.visible = true
		var p0 := _to3(float(u0["x"]), float(u0["y"]))
		var p1 := _to3(float(u1["x"]), float(u1["y"]))
		if bool(u0["alive"]):
			root.position = p0.lerp(p1, t)
			root.rotation.y = -lerp_angle(float(u0["heading"]), float(u1["heading"]), t)
		else:
			root.position = p1
			root.rotation.y = -float(u1["heading"])
		_set_ring(id, float(u1["hp"]) / maxf(1.0, float(u1["max_hp"])))


func _set_ring(id: String, ratio: float) -> void:
	var m: StandardMaterial3D = _unit_ring[id]
	m.albedo_color = Color(0.9, 0.2, 0.2).lerp(Color(0.25, 0.9, 0.35), clampf(ratio, 0.0, 1.0))


func _to3(x: float, y: float) -> Vector3:
	return Vector3(x - world.x * 0.5, 0.0, y - world.y * 0.5)


# ------------------------------------------------------------------
# 特效：尾迹 / 命中激光 / 击毁爆散
# ------------------------------------------------------------------

func _build_fx() -> void:
	# 尾迹：一个 ImmediateMesh 承载全部单位的渐隐折线，仅在换帧时重建
	_trails_imm = ImmediateMesh.new()
	var tm := MeshInstance3D.new()
	tm.mesh = _trails_imm
	tm.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_trails_mat = StandardMaterial3D.new()
	_trails_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_trails_mat.vertex_color_use_as_albedo = true
	_trails_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	tm.material_override = _trails_mat
	add_child(tm)

	# 命中激光：细长自发光方柱（靠泛光 bloom 出激光感），附带少量点光源
	var box := BoxMesh.new()
	box.size = Vector3(1.0, 1.0, 1.0)
	for i in LASER_POOL:
		var mi := MeshInstance3D.new()
		mi.mesh = box
		mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		var m := StandardMaterial3D.new()
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		m.albedo_color = Color(1, 1, 1, 0)
		m.emission_enabled = true
		mi.material_override = m
		mi.visible = false
		add_child(mi)
		var light: OmniLight3D = null
		if i < LASER_LIGHTS:
			light = OmniLight3D.new()
			light.omni_range = 60.0
			light.visible = false
			add_child(light)
		_lasers.append({"node": mi, "mat": m, "light": light, "col": Color(1, 1, 1), "active": false})

	# 击毁爆散：一次性 CPU 粒子池（兼容所有渲染器）
	for i in BOOM_POOL:
		var p := CPUParticles3D.new()
		p.emitting = false
		p.one_shot = true
		p.amount = 26
		p.lifetime = 0.7
		p.explosiveness = 1.0
		p.direction = Vector3(0, 1, 0)
		p.spread = 180.0
		p.gravity = Vector3(0, -32, 0)
		p.initial_velocity_min = 9.0
		p.initial_velocity_max = 26.0
		p.damping_min = 2.0
		p.damping_max = 6.0
		p.scale_amount_min = 0.6
		p.scale_amount_max = 1.5
		p.mesh = _boom_mesh()
		p.color_ramp = _boom_ramp()
		add_child(p)
		_booms.append(p)


func _boom_mesh() -> SphereMesh:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.vertex_color_use_as_albedo = true
	m.albedo_color = Color(1, 0.6, 0.3)
	var s := SphereMesh.new()
	s.radius = 0.9
	s.height = 1.8
	s.material = m
	return s


func _boom_ramp() -> Gradient:
	var g := Gradient.new()
	g.colors = PackedColorArray([
		Color(1.0, 0.8, 0.4, 1.0),
		Color(1.0, 0.35, 0.1, 0.9),
		Color(0.3, 0.05, 0.02, 0.0),
	])
	return g


func _clear_fx() -> void:
	for slot in _lasers:
		slot["active"] = false
		slot["node"].visible = false
		if slot["light"] != null:
			slot["light"].visible = false


func _spawn_laser(e: Dictionary) -> void:
	var a := _find_unit_pos(str(e.get("attacker", "")))
	var b := _find_unit_pos(str(e.get("target", "")))
	if not a.is_finite() or not b.is_finite():
		return
	var dist := a.distance_to(b)
	if dist < 1.0:
		return
	var col := COL_BLUE if _team_of(str(e.get("attacker", ""))) == 0 else COL_RED
	for slot in _lasers:
		if slot["active"]:
			continue
		var node: MeshInstance3D = slot["node"]
		var mid := (a + b) * 0.5 + Vector3(0, 9.0, 0)
		node.look_at_from_position(mid, b + Vector3(0, 9.0, 0), Vector3.UP)
		node.scale = Vector3(1.5, 1.5, dist)
		node.visible = true
		slot["col"] = col
		slot["active"] = true
		var m: StandardMaterial3D = slot["mat"]
		m.albedo_color = Color(col.r, col.g, col.b, 1.0)
		m.emission = col * 2.5
		var light = slot["light"]
		if light != null:
			light.light_color = col
			light.position = mid
			light.visible = true
		return


func _update_lasers() -> void:
	var k := 0.85  # 暂停时常显，便于定位观察
	if playing:
		k = 1.0 - clampf(_laser_age / LASER_MAX_AGE, 0.0, 1.0)
	for slot in _lasers:
		if not slot["active"]:
			continue
		if k <= 0.0:
			slot["active"] = false
			slot["node"].visible = false
			if slot["light"] != null:
				slot["light"].visible = false
			continue
		var col: Color = slot["col"]
		var m: StandardMaterial3D = slot["mat"]
		m.albedo_color = Color(col.r, col.g, col.b, 0.9 * k)
		m.emission = col * (2.5 * k + 0.1)
		if slot["light"] != null:
			slot["light"].light_energy = 2.5 * k


func _spawn_boom(e: Dictionary) -> void:
	var p := _find_unit_pos(str(e.get("unit", "")))
	if not p.is_finite():
		return
	for b in _booms:
		if b.emitting:
			continue
		b.position = p + Vector3(0, 4.0, 0)
		b.restart()
		return


func _find_unit_pos(id: String) -> Vector3:
	if frame_idx < 0 or not _id_index.has(id):
		return Vector3.INF
	var idx: int = _id_index[id]
	var u: Dictionary = frames[frame_idx]["units"][idx]
	return _to3(float(u["x"]), float(u["y"]))


func _team_of(id: String) -> int:
	return 0 if id.begins_with("B") else 1


func _rebuild_trails() -> void:
	if _trails_imm == null or frames.is_empty() or frame_idx < 0:
		return
	_trails_imm.clear_surfaces()
	var units: Array = frames[frame_idx]["units"]
	var start := maxi(0, frame_idx - TRAIL_STEPS + 1)
	var span := frame_idx - start + 1
	var stride := maxi(1, int(ceil(float(span) / float(TRAIL_MAX_POINTS))))
	for id in _unit_ids:
		var k: int = _id_index[id]
		var u: Dictionary = units[k]
		if not bool(u["alive"]):
			continue
		var col := COL_BLUE if int(u["team"]) == 0 else COL_RED
		# 先采样到数组, 不足 2 点则跳过: LINE_STRIP 少于 2 顶点时
		# Vulkan 每帧每条带报 "Too few vertices"(frame 0 时 span=1 必现)
		var verts := PackedVector3Array()
		var cols := PackedColorArray()
		var kk := start
		while kk <= frame_idx:
			var pu: Dictionary = frames[kk]["units"][k]
			var t := float(kk - start) / float(maxi(1, span - 1))
			cols.append(Color(col.r, col.g, col.b, 0.04 + 0.5 * t * t))
			verts.append(_to3(float(pu["x"]), float(pu["y"])) + Vector3(0, 1.4, 0))
			kk += stride
		if verts.size() < 2:
			continue
		_trails_imm.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
		for vi in verts.size():
			_trails_imm.surface_set_color(cols[vi])
			_trails_imm.surface_add_vertex(verts[vi])
		_trails_imm.surface_end()
