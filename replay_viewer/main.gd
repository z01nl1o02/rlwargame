extends Control
## 战争模拟回放查看器（Godot 4 · 3D 版）
##
## 加载 backend/war_sim/recorder.py 生成的 battle_*.json，逐帧回放整场战斗。
## 回放是纯数据驱动：任意帧可直接摆放，暂停/倒放/拖进度条天然支持。
##
## 用法：
##   godot --path replay_viewer                     # 启动后弹文件选择框
##   godot --path replay_viewer -- <battle.json>    # 直接打开指定回放
##   也可把 JSON 文件直接拖进窗口
## 快捷键：空格 播放/暂停；←/→ 单步；R 切换正/倒放；Home/End 跳开头/结尾
## 鼠标：左键拖拽旋转，滚轮缩放，右键拖拽平移

const BattleWorld := preload("res://battle_world.gd")
const StatsPanel := preload("res://stats_panel.gd")

const TOP_H := 42.0
const BOTTOM_H := 60.0
const STATS_W := 330.0
const PAD := 8.0
const SPEEDS := [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
const BASE_STEPS_PER_SEC := 20.0

var frames: Array = []
var meta: Dictionary = {}
var world := Vector2(1000.0, 600.0)
var frame_count := 0

var frame_idx := 0
var playing := false
var direction := 1
var speed_idx := 2
var _accum := 0.0

var battle_world
var stats_panel
var file_label: Label
var info_label: Label
var open_btn: Button
var start_btn: Button
var play_btn: Button
var dir_btn: Button
var speed_btn: Button
var end_btn: Button
var slider: HSlider
var time_label: Label
var dialog: FileDialog


func _ready() -> void:
	theme = _make_theme()
	mouse_filter = Control.MOUSE_FILTER_IGNORE

	battle_world = BattleWorld.new()
	add_child(battle_world)

	stats_panel = StatsPanel.new()
	stats_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	stats_panel.clip_contents = true
	add_child(stats_panel)

	file_label = Label.new()
	file_label.text = "未加载回放"
	file_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	file_label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	add_child(file_label)

	info_label = Label.new()
	info_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	info_label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	add_child(info_label)

	open_btn = _mk_button("打开回放…", 114.0)
	open_btn.pressed.connect(_open_dialog)

	start_btn = _mk_button("开头", 64.0)
	start_btn.pressed.connect(_jump_start)
	play_btn = _mk_button("播放", 84.0)
	play_btn.pressed.connect(_toggle_play)
	dir_btn = _mk_button("方向:正放", 116.0)
	dir_btn.pressed.connect(_toggle_direction)
	speed_btn = _mk_button("速度 1x", 104.0)
	speed_btn.pressed.connect(_cycle_speed)
	end_btn = _mk_button("结尾", 64.0)
	end_btn.pressed.connect(_jump_end)

	slider = HSlider.new()
	slider.min_value = 0
	slider.max_value = 0
	slider.step = 1
	slider.focus_mode = Control.FOCUS_NONE
	slider.editable = false
	slider.value_changed.connect(_on_slider_changed)
	add_child(slider)

	time_label = Label.new()
	time_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	time_label.add_theme_font_size_override("font_size", 13)
	add_child(time_label)

	dialog = FileDialog.new()
	dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	dialog.access = FileDialog.ACCESS_FILESYSTEM
	dialog.filters = PackedStringArray(["*.json ; 回放 JSON"])
	dialog.title = "选择回放文件 battle_*.json"
	dialog.file_selected.connect(_load_replay)
	add_child(dialog)

	get_window().files_dropped.connect(_on_files_dropped)

	_update_buttons()
	_layout()

	var args := OS.get_cmdline_user_args()
	if args.size() > 0 and not str(args[0]).is_empty():
		_load_replay(str(args[0]))
	else:
		_open_dialog()


func _make_theme() -> Theme:
	var th := Theme.new()
	var font := SystemFont.new()
	font.font_names = PackedStringArray([
		"Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei",
		"Microsoft YaHei", "PingFang SC", "sans-serif",
	])
	th.default_font = font
	th.default_font_size = 14
	return th


func _mk_button(text: String, width: float) -> Button:
	var b := Button.new()
	b.text = text
	b.focus_mode = Control.FOCUS_NONE
	add_child(b)
	return b


func _notification(what: int) -> void:
	if what == NOTIFICATION_RESIZED:
		_layout()



func _layout() -> void:
	if stats_panel == null:
		return  # _ready 完成前根 Control 就可能收到 RESIZED 通知，此时子控件尚未创建
	var s := size
	if s.x <= 1.0 or s.y <= 1.0:
		return
	stats_panel.position = Vector2(s.x - STATS_W, TOP_H)
	stats_panel.size = Vector2(STATS_W - PAD, maxf(50.0, s.y - TOP_H - BOTTOM_H))
	file_label.position = Vector2(PAD, 6.0)
	file_label.size = Vector2(s.x - STATS_W - s.x * 0.08, TOP_H - 12.0)
	info_label.position = Vector2(s.x * 0.42 + PAD, 6.0)
	info_label.size = Vector2(maxf(100.0, s.x - STATS_W - s.x * 0.42 - PAD * 2.0), TOP_H - 12.0)
	open_btn.position = Vector2(s.x - 122.0, 5.0)
	open_btn.size = Vector2(114.0, 32.0)

	var by := s.y - BOTTOM_H
	var x := PAD
	var widths := [[start_btn, 64.0], [play_btn, 84.0], [dir_btn, 116.0], [speed_btn, 104.0], [end_btn, 64.0]]
	for w in widths:
		var btn: Button = w[0]
		btn.position = Vector2(x, by + 13.0)
		btn.size = Vector2(w[1], 34.0)
		x += w[1] + 6.0
	var label_w := 268.0
	var slider_w := maxf(80.0, s.x - x - label_w - PAD * 2.0)
	slider.position = Vector2(x, by + 18.0)
	slider.size = Vector2(slider_w, 24.0)
	time_label.position = Vector2(s.x - label_w - PAD, by + 13.0)
	time_label.size = Vector2(label_w, 34.0)


# ------------------------------------------------------------------
# 回放推进
# ------------------------------------------------------------------

func _process(delta: float) -> void:
	if playing and frame_count > 1:
		_accum += delta * BASE_STEPS_PER_SEC * SPEEDS[speed_idx]
		while _accum >= 1.0:
			_accum -= 1.0
			var nxt := frame_idx + direction
			if nxt < 0 or nxt >= frame_count:
				_accum = 0.0
				_set_playing(false)
				break
			_set_frame(nxt)
	battle_world.set_progress(_accum)


func _set_frame(i: int, jump: bool = false) -> void:
	if frame_count == 0:
		return
	frame_idx = clampi(i, 0, frame_count - 1)
	battle_world.set_frame(frame_idx, jump)
	stats_panel.set_cursor(frame_idx)
	slider.set_value_no_signal(float(frame_idx))
	_update_time_label()


func _set_playing(p: bool) -> void:
	playing = p
	if not p:
		_accum = 0.0
	battle_world.set_playing(p)
	_update_buttons()


func _toggle_play() -> void:
	if frame_count == 0:
		return
	if not playing:
		if direction == 1 and frame_idx >= frame_count - 1:
			_set_frame(0, true)  # 结尾再按播放 -> 从头开始
		elif direction == -1 and frame_idx <= 0:
			_set_frame(frame_count - 1, true)  # 开头再按倒放 -> 从结尾开始，与正放对称
	_set_playing(not playing)


func _toggle_direction() -> void:
	direction = -direction
	_update_buttons()
	_update_time_label()


func _cycle_speed() -> void:
	speed_idx = (speed_idx + 1) % SPEEDS.size()
	_update_buttons()


func _jump_start() -> void:
	_set_frame(0, true)


func _jump_end() -> void:
	_set_frame(frame_count - 1, true)


func _on_slider_changed(v: float) -> void:
	_set_frame(int(v), true)


func _unhandled_key_input(event: InputEvent) -> void:
	var k := event as InputEventKey
	if k == null or not k.pressed or k.echo:
		return
	match k.keycode:
		KEY_SPACE:
			_toggle_play()
		KEY_LEFT:
			_set_frame(frame_idx - 1)
		KEY_RIGHT:
			_set_frame(frame_idx + 1)
		KEY_R:
			_toggle_direction()
		KEY_HOME:
			_jump_start()
		KEY_END:
			_jump_end()


func _update_buttons() -> void:
	play_btn.text = "暂停" if playing else "播放"
	dir_btn.text = "方向:正放" if direction == 1 else "方向:倒放"
	speed_btn.text = "速度 " + _fmt_speed(SPEEDS[speed_idx]) + "x"


func _fmt_speed(v: float) -> String:
	if absf(v - roundf(v)) < 0.01:
		return str(int(v))
	return str(v)


func _update_time_label() -> void:
	if frame_count == 0:
		time_label.text = ""
		return
	var fr: Dictionary = frames[frame_idx]
	var st: Dictionary = fr["stats"]
	var mark := "  [倒放]" if direction == -1 else ""
	time_label.text = "%d / %d 步  t=%.0fs  蓝 %d vs 红 %d%s" % [
		frame_idx, frame_count - 1, float(fr.get("time", 0.0)),
		int(st.get("blue_alive", 0)), int(st.get("red_alive", 0)), mark,
	]


# ------------------------------------------------------------------
# 回放加载
# ------------------------------------------------------------------

func _load_replay(path: String) -> void:
	var fa := FileAccess.open(path, FileAccess.READ)
	if fa == null:
		_show_error("无法打开文件: %s" % path)
		return
	var doc = JSON.parse_string(fa.get_as_text())
	if typeof(doc) != TYPE_DICTIONARY:
		_show_error("JSON 解析失败: %s" % path)
		return
	var m = doc.get("meta")
	var fr = doc.get("frames")
	if typeof(m) != TYPE_DICTIONARY or typeof(fr) != TYPE_ARRAY or fr.is_empty():
		_show_error("不是有效的回放文件(缺少 meta/frames): %s" % path.get_file())
		return
	if int(m.get("format_version", 0)) != 1:
		_show_error("不支持的 format_version: %s" % str(m.get("format_version")))
		return
	var cfg: Dictionary = m.get("config", {})
	meta = m
	frames = fr
	frame_count = frames.size()
	world = Vector2(float(cfg.get("world_w", 1000.0)), float(cfg.get("world_h", 600.0)))
	var per_side := int(cfg.get("n_units_per_side", 50))

	playing = false
	direction = 1
	speed_idx = 2
	_accum = 0.0
	frame_idx = 0

	battle_world.setup(frames, world)
	stats_panel.setup(frames, per_side)

	slider.max_value = frame_count - 1
	slider.editable = true
	file_label.text = "%s   (%d 帧)" % [path.get_file(), frame_count]
	info_label.remove_theme_color_override("font_color")
	info_label.text = _summary_text()
	_set_frame(0, true)
	_update_buttons()
	_update_time_label()


func _summary_text() -> String:
	var res: Dictionary = meta.get("result", {})
	# Dictionary.get() 返回 Variant，Godot 4 中 := 推断会触发 INFERENCE_ON_VARIANT 错误，须显式声明类型
	var winner: String = str({"blue": "蓝方", "red": "红方", "draw": "平局"}.get(str(res.get("winner", "draw")), "平局"))
	var src: String = str({"run_sim": "离线模拟", "server": "网页对局"}.get(str(meta.get("source", "")), str(meta.get("source", "?"))))
	var ended: String = str({"episode_end": "战斗结束", "manual_reset": "手动重置", "script_exit": "步数上限"}.get(str(meta.get("ended_by", "")), str(meta.get("ended_by", "?"))))
	return "seed=%s · %s · %s: %s · 蓝 %d / 红 %d 存活" % [
		str(meta.get("seed", "?")), src, ended, winner,
		int(res.get("blue_alive", 0)), int(res.get("red_alive", 0)),
	]


func _show_error(msg: String) -> void:
	push_warning(msg)
	info_label.add_theme_color_override("font_color", Color(1.0, 0.5, 0.42))
	info_label.text = "加载失败: " + msg


func _open_dialog() -> void:
	var replays := ProjectSettings.globalize_path("res://../backend/replays")
	if replays != "" and DirAccess.dir_exists_absolute(replays):
		dialog.current_dir = replays
	dialog.popup_centered(Vector2i(960, 640))


func _on_files_dropped(files: PackedStringArray) -> void:
	if files.size() > 0:
		_load_replay(files[0])
