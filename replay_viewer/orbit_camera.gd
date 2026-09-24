extends Node3D
## 轨道相机：左键拖拽旋转视角，滚轮缩放，右键拖拽平移。
## 镜头始终看向 target（战场中心），缩放带平滑过渡。

const PITCH_MIN := -1.45
const PITCH_MAX := -0.08
const DIST_MIN := 140.0
const DIST_MAX := 2600.0

var target := Vector3.ZERO
var yaw := 0.0
var pitch := -0.85
var distance := 950.0
var _target_distance := 950.0

var _yaw_node: Node3D
var _pitch_node: Node3D
var _cam: Camera3D


func _ready() -> void:
	_yaw_node = Node3D.new()
	add_child(_yaw_node)
	_pitch_node = Node3D.new()
	_yaw_node.add_child(_pitch_node)
	_cam = Camera3D.new()
	_cam.fov = 55.0
	_cam.near = 0.5
	_cam.far = 10000.0
	_pitch_node.add_child(_cam)
	_cam.position = Vector3(0, 0, distance)
	_cam.current = true
	_apply()


func _apply() -> void:
	position = target
	_yaw_node.rotation.y = yaw
	_pitch_node.rotation.x = pitch
	_cam.position = Vector3(0, 0, distance)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.pressed and mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			_target_distance = maxf(DIST_MIN, _target_distance * 0.9)
		elif mb.pressed and mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_target_distance = minf(DIST_MAX, _target_distance * 1.1)
	elif event is InputEventMouseMotion:
		var mm := event as InputEventMouseMotion
		if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
			yaw -= mm.relative.x * 0.005
			pitch = clampf(pitch - mm.relative.y * 0.005, PITCH_MIN, PITCH_MAX)
			_apply()
		elif Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
			# 沿镜头水平面平移观察点
			var fwd := -_yaw_node.global_transform.basis.z
			var right := _yaw_node.global_transform.basis.x
			var pan := (-mm.relative.x * right + mm.relative.y * fwd) * distance * 0.0016
			pan.y = 0.0
			target += pan
			_apply()


func _process(delta: float) -> void:
	if absf(distance - _target_distance) > 0.5:
		distance = lerpf(distance, _target_distance, minf(1.0, delta * 12.0))
		_cam.position = Vector3(0, 0, distance)
