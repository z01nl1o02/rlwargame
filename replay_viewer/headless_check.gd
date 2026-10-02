extends SceneTree
## 回放查看器无头回归测试（不需要显示器/显卡）。
##
## 用法:
##   godot --headless --path replay_viewer -s res://headless_check.gd
##
## 覆盖: 场景启动、错误路径(文件缺失/坏 JSON/错误 format_version)、
## 两份真实回放的加载与全帧遍历(含事件特效生成)、跳转/正倒放环绕/
## 变速/滑杆/快捷键、播放暂停推进。全部通过打印 ALL PASS 并以 0 退出。
##
## 已知 headless 假象: 启动阶段 "Window 0 spawned at invalid position"
## 来自 FileDialog.popup_centered 在无显示服务器的 64x64 假窗口上居中,
## 真实窗口环境不受影响, 不计入失败。

var failures: PackedStringArray = []


func _initialize() -> void:
	_run()


func _check(cond: bool, what: String) -> void:
	if not cond:
		failures.append(what)
		print("  FAIL: ", what)
	else:
		print("  ok: ", what)


func _latest_replays() -> PackedStringArray:
	var dir_path := ProjectSettings.globalize_path("res://../backend/replays")
	var out: PackedStringArray = []
	var dir := DirAccess.open(dir_path)
	if dir == null:
		return out
	var names: PackedStringArray = []
	for f in dir.get_files():
		if f.begins_with("battle_") and f.ends_with(".json"):
			names.append(dir_path.path_join(f))
	names.sort()
	out = names
	return out


func _run() -> void:
	print("== boot ==")
	var packed := load("res://main.tscn")
	_check(packed != null, "main.tscn loads")
	var main: Node = packed.instantiate()
	root.add_child(main)
	await process_frame
	await process_frame
	_check(main.file_label != null, "labels created (no _layout crash)")
	_check(main.dialog != null, "dialog created")

	print("== error paths ==")
	main._load_replay("/nonexistent/x.json")
	_check(str(main.info_label.text).begins_with("加载失败"), "missing file -> error label")

	var bad := FileAccess.open("/tmp/rv_bad.json", FileAccess.WRITE)
	bad.store_string("{\"meta\": {}, \"frames\": []}")
	bad.close()
	main._load_replay("/tmp/rv_bad.json")
	_check(str(main.info_label.text).begins_with("加载失败"), "empty frames -> error label")

	bad = FileAccess.open("/tmp/rv_badver.json", FileAccess.WRITE)
	bad.store_string("{\"meta\": {\"format_version\": 99}, \"frames\": [{}]}")
	bad.close()
	main._load_replay("/tmp/rv_badver.json")
	_check(str(main.info_label.text).begins_with("加载失败"), "bad format_version -> error label")

	for p in ["/tmp/rv_bad.json", "/tmp/rv_badver.json"]:
		DirAccess.remove_absolute(p)

	var replays := _latest_replays()
	if replays.is_empty():
		failures.append("backend/replays 下没有 battle_*.json, 无法做真实回放回归")
		print("  FAIL: no replay files found")
	else:
		_test_replay(main, replays[0])
		if replays.size() > 1:
			print("== second replay: ", replays[1].get_file(), " ==")
			main._load_replay(replays[1])
			_check(main.frame_count > 1, "frames loaded: %d" % main.frame_count)
			main._jump_end()
			await process_frame
			_check(main.battle_world.frame_idx == main.frame_count - 1, "last frame applied")
			_check(main.battle_world._unit_ids.size() > 0, "unit ids built")

	if failures.is_empty():
		print("\nALL PASS")
		quit(0)
	else:
		print("\n%d FAILURES" % failures.size())
		quit(1)


func _test_replay(main: Node, path: String) -> void:
	print("== replay: ", path.get_file(), " ==")
	main._load_replay(path)
	_check(main.frame_count > 1, "frames loaded: %d" % main.frame_count)
	_check(main.battle_world._unit_root.size() > 0, "unit nodes built: %d" % main.battle_world._unit_root.size())
	_check(main.battle_world._unit_ring.size() == main.battle_world._unit_root.size(), "hp-ring per unit")
	_check(main.battle_world._grave.size() == main.battle_world._unit_root.size(), "grave per unit")
	_check(main.slider.max_value == main.frame_count - 1, "slider range set")
	_check(str(main.file_label.text).contains("帧"), "file label updated")
	_check(not str(main.info_label.text).begins_with("加载失败"), "summary shown: " + str(main.info_label.text))

	# 逐帧步进覆盖所有帧（起点已在帧 0），走过全部事件特效生成路径
	for i in range(1, main.frame_count):
		main._set_frame(i, false)
	_check(main.frame_idx == main.frame_count - 1, "stepped to last frame")
	_check(main.battle_world.frame_idx == main.frame_count - 1, "world frame synced")

	print("== interactions ==")
	main._set_frame(main.frame_count - 1, true)
	main._toggle_play()  # 结尾播放 -> 应回到开头并播放
	_check(main.frame_idx == 0, "play at end wraps to start")
	_check(main.playing, "playing on")
	main._toggle_play()
	_check(not main.playing, "paused after toggle")
	main._toggle_direction()
	_check(main.direction == -1, "direction reversed")
	# 开头 + 倒放 -> 对称环绕到结尾开始
	main._toggle_play()
	_check(main.playing, "playing on")
	_check(main.frame_idx == main.frame_count - 1, "play at start in reverse wraps to end")
	var mid: int = main.frame_count / 2
	main._on_slider_changed(mid)
	for j in 5:
		await process_frame
	_check(main.frame_idx < mid, "backward playback advanced (idx=%d)" % main.frame_idx)
	main._toggle_play()
	_check(not main.playing, "paused")
	# 速度切换一轮
	var s0: int = main.speed_idx
	main._cycle_speed()
	_check(main.speed_idx == (s0 + 1) % main.SPEEDS.size(), "speed cycles")
	# 滑杆跳转
	main._on_slider_changed(mid)
	_check(main.frame_idx == mid, "slider scrub")
	# 键盘快捷键
	var ev := InputEventKey.new()
	ev.keycode = KEY_RIGHT
	ev.pressed = true
	main._unhandled_key_input(ev)
	_check(main.frame_idx == mid + 1, "KEY_RIGHT steps")
	ev = InputEventKey.new()
	ev.keycode = KEY_HOME
	ev.pressed = true
	main._unhandled_key_input(ev)
	_check(main.frame_idx == 0, "KEY_HOME jumps to start")
	ev = InputEventKey.new()
	ev.keycode = KEY_END
	ev.pressed = true
	main._unhandled_key_input(ev)
	_check(main.frame_idx == main.frame_count - 1, "KEY_END jumps to end")
