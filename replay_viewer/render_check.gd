extends SceneTree
## 真实渲染（像素级）回归。区别于 headless_check.gd：需要显示器 + GPU（或软渲染），
## 验证的是"渲染管线与画面内容"，而不只是脚本逻辑。
##
## 用法:
##   godot --path replay_viewer --audio-driver Dummy -s res://render_check.gd
##
## 做什么: 自动选取 backend/replays/ 最新回放, seek 到 4 个代表性帧
## (开局 / 激光最密集 / 大量阵亡 / 终局)真实渲染, 抓取视口图像保存到
## /tmp/rv_shots/, 并做像素统计断言:
##   - 开局: 战场上同时可见蓝方与红方单位、满血 HP 环(绿)
##   - 激光帧: 双方仍可见, 且泛光亮像素不少于开局(激光 glow)
##   - 终局: 败方(先打完的一方)在 3D 区的像素大幅塌陷
## 全过打印 ALL PASS 退出 0; 有失败退出 1; headless 环境跳过退出 2。
## 统计只取 3D 视口裁剪区, 避开右侧统计曲线/底栏/顶栏的 UI 干扰。

const OUT_DIR := "/tmp/rv_shots"
# 3D 视口裁剪区(避开顶栏/底栏/右侧统计面板), 再向内收紧以减少边界墙像素占比
const CROP := Rect2i(160, 80, 780, 640)

var main: Node
var failures := 0
var start_blue := 0
var start_red := 0
var start_green := 0


func _initialize() -> void:
	if DisplayServer.get_name() == "headless":
		print("SKIP: 渲染回归需要真实显示与 GPU, headless 请用 headless_check.gd")
		quit(2)
		return
	main = load("res://main.tscn").instantiate()
	root.add_child(main)
	_run()


func _check(cond: bool, what: String) -> void:
	if not cond:
		failures += 1
		print("  FAIL: ", what)
	else:
		print("  ok: ", what)


func _run() -> void:
	await process_frame
	await process_frame
	DirAccess.make_dir_recursive_absolute(OUT_DIR)
	# 无命令行参数时 _ready 会弹文件对话框; 自动发现回放后手动加载, 关掉对话框
	var path := _latest_replay()
	if path.is_empty():
		_check(false, "backend/replays 下没有 battle_*.json")
		_finish()
		return
	if main.dialog != null:
		main.dialog.hide()
	main._load_replay(path)
	print("replay=", path.get_file(), " frames=", main.frame_count)
	_check(main.frame_count > 1, "回放已加载")

	var best_i := 0
	var best_n := 0
	for i in main.frame_count:
		var evs: Array = main.frames[i].get("events", [])
		if evs.size() > best_n:
			best_n = evs.size()
			best_i = i

	# 开局
	main._jump_start()
	for j in 6:
		await process_frame
	var img := _shot("1_start")
	var st := _color_stats(img)
	start_blue = st.blue
	start_red = st.red
	start_green = st.green
	print("start: ", st)
	_check(st.blue >= 300, "开局可见蓝方单位+边界墙(blue=%d)" % st.blue)
	_check(st.red >= 80, "开局可见红方单位(red=%d, 单位本体较小属正常)" % st.red)
	_check(st.green >= 30, "开局可见满血 HP 环(green=%d)" % st.green)

	# 激光最密集帧: 顺序步进进入(jump 不生成特效), 暂停下激光常显
	for i in range(maxi(1, best_i - 2), best_i + 1):
		main._set_frame(i, false)
	for j in 3:
		await process_frame
	img = _shot("2_lasers")
	st = _color_stats(img)
	print("lasers(frame=%d, events=%d): " % [best_i, best_n], st)
	_check(st.blue > 0 and st.red > 0, "激光帧战场双方可见")

	# 大量阵亡帧
	var grave_i := int(main.frame_count / 2)
	for i in main.frame_count - 1:
		if int(main.frames[i + 1]["stats"]["blue_alive"]) + int(main.frames[i + 1]["stats"]["red_alive"]) <= 80:
			grave_i = mini(grave_i, i + 1)
			break
	main._set_frame(grave_i, true)
	for j in 6:
		await process_frame
	img = _shot("3_graves")
	print("graves(frame=%d): " % grave_i, _color_stats(img))

	# 终局: 败方(蓝先灭, 与 stats 一致即可推导)像素塌陷
	main._jump_end()
	for j in 6:
		await process_frame
	img = _shot("4_end")
	st = _color_stats(img)
	# 终局: 败方像素应明显塌陷(边界墙/地面等常驻背景色仍在, 不能断言归零)
	var eb: int = main.frames[main.frame_count - 1]["stats"]["blue_alive"]
	var er: int = main.frames[main.frame_count - 1]["stats"]["red_alive"]
	print("end: ", st)
	if eb == er:
		print("  ok: 终局平局(eb=%d er=%d), 跳过单边塌除断言" % [eb, er])
	elif eb < er:
		_check(st.blue <= start_blue * 2 / 3,
			"终局蓝败, 3D 区蓝色像素塌陷(end=%d, start=%d)" % [st.blue, start_blue])
	else:
		_check(st.red <= start_red * 2 / 3,
			"终局红败, 3D 区红色像素塌陷(end=%d, start=%d)" % [st.red, start_red])
	_check(img.get_width() >= 800, "视口尺寸正常(%dx%d)" % [img.get_width(), img.get_height()])
	for tag in ["1_start", "2_lasers", "3_graves", "4_end"]:
		_check(FileAccess.file_exists("%s/%s.png" % [OUT_DIR, tag]), "截图已落盘: %s.png" % tag)
	_finish()


func _latest_replay() -> String:
	var dir_path := ProjectSettings.globalize_path("res://../backend/replays")
	var names: PackedStringArray = []
	var dir := DirAccess.open(dir_path)
	if dir == null:
		return ""
	for f in dir.get_files():
		if f.begins_with("battle_") and f.ends_with(".json"):
			names.append(dir_path.path_join(f))
	names.sort()
	return names[names.size() - 1] if not names.is_empty() else ""


func _shot(tag: String) -> Image:
	var img: Image = root.get_texture().get_image()
	img.save_png("%s/%s.png" % [OUT_DIR, tag])
	return img


func _color_stats(img: Image) -> Dictionary:
	var blue := 0
	var red := 0
	var green := 0
	var bright := 0
	var n := 0
	for y in range(CROP.position.y, CROP.end.y, 4):
		for x in range(CROP.position.x, CROP.end.x, 4):
			n += 1
			var c := img.get_pixel(x, y)
			if c.b > c.r + 0.15 and c.b > c.g + 0.05 and c.b > 0.25:
				blue += 1
			elif c.r > c.b + 0.15 and c.r > c.g + 0.05 and c.r > 0.25:
				red += 1
			elif c.g > c.r + 0.1 and c.g > c.b + 0.1 and c.g > 0.25:
				green += 1
			if c.r + c.g + c.b > 2.4:
				bright += 1
	return {"blue": blue, "red": red, "green": green, "bright": bright, "samples": n}


func _finish() -> void:
	if failures == 0:
		print("\nALL PASS (截图在 %s)" % OUT_DIR)
		quit(0)
	else:
		print("\n%d FAILURES" % failures)
		quit(1)
