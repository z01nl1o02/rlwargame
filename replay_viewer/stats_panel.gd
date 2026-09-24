extends Control
## 右侧统计曲线（2D 叠加）：存活单位 / 累计伤害 / 火炮命中率。
## 数据在 setup() 一次性抽好，绘制只读数组；游标随当前步移动。
## 暂停、倒放、拖进度条都只是移动游标，无任何残留状态。

const COL_BLUE := Color(0.30, 0.58, 1.0)
const COL_RED := Color(1.0, 0.42, 0.34)
const COL_PANEL := Color(0.07, 0.085, 0.12, 0.92)
const COL_AXIS := Color(0.28, 0.33, 0.44)
const COL_TEXT := Color(0.78, 0.82, 0.90)
const COL_DIM := Color(0.5, 0.55, 0.65)
const COL_CURSOR := Color(1, 1, 1, 0.5)

var _n := 0
var _cursor := 0
var _per_side := 50
var blue_alive := PackedFloat32Array()
var red_alive := PackedFloat32Array()
var blue_dmg := PackedFloat32Array()
var red_dmg := PackedFloat32Array()
var blue_eff := PackedFloat32Array()
var red_eff := PackedFloat32Array()
var _max_dmg := 1.0


func setup(frames: Array, per_side: int) -> void:
	_n = frames.size()
	_per_side = per_side
	blue_alive.resize(_n)
	red_alive.resize(_n)
	blue_dmg.resize(_n)
	red_dmg.resize(_n)
	blue_eff.resize(_n)
	red_eff.resize(_n)
	var md := 1.0
	for i in _n:
		var st: Dictionary = frames[i]["stats"]
		blue_alive[i] = float(st["blue_alive"])
		red_alive[i] = float(st["red_alive"])
		blue_dmg[i] = float(st["blue_damage"])
		red_dmg[i] = float(st["red_damage"])
		blue_eff[i] = float(st["blue_fire_efficiency"])
		red_eff[i] = float(st["red_fire_efficiency"])
		md = maxf(md, maxf(blue_dmg[i], red_dmg[i]))
	_max_dmg = md * 1.06
	set_cursor(0)


func set_cursor(i: int) -> void:
	if _n == 0:
		return
	_cursor = clampi(i, 0, _n - 1)
	queue_redraw()


func _notification(what: int) -> void:
	if what == NOTIFICATION_RESIZED:
		queue_redraw()


func _draw() -> void:
	var font := get_theme_default_font()
	if font == null:
		return
	draw_rect(Rect2(Vector2.ZERO, size), COL_PANEL)
	var pad := 6.0
	var gap := 8.0
	var chart_h := (size.y - pad * 2.0 - gap * 2.0) / 3.0
	_draw_chart(Rect2(Vector2(pad, pad), Vector2(size.x - pad * 2.0, chart_h)),
		"存活单位", blue_alive, red_alive, float(_per_side), "%d", 1.0, font)
	_draw_chart(Rect2(Vector2(pad, pad + chart_h + gap), Vector2(size.x - pad * 2.0, chart_h)),
		"累计伤害", blue_dmg, red_dmg, _max_dmg, "%.0f", 1.0, font)
	_draw_chart(Rect2(Vector2(pad, pad + (chart_h + gap) * 2.0), Vector2(size.x - pad * 2.0, chart_h)),
		"命中率", blue_eff, red_eff, 1.0, "%d%%", 100.0, font)


func _draw_chart(rect: Rect2, title: String, a: PackedFloat32Array, b: PackedFloat32Array,
		ymax: float, vfmt: String, vscale: float, font: Font) -> void:
	var fs := 13
	var title_y := rect.position.y + 5.0 + fs
	draw_string(font, rect.position + Vector2(8.0, 5.0 + fs), title, HORIZONTAL_ALIGNMENT_LEFT, -1, fs, COL_TEXT)
	draw_rect(rect, COL_AXIS, false, 1.0)
	if _n < 2 or a.size() < 2:
		return

	# 当前值（右上角，随游标实时变化）
	var va := _fmt(a[_cursor], vfmt, vscale)
	var vb := _fmt(b[_cursor], vfmt, vscale)
	draw_string(font, Vector2(rect.end.x - 158.0, title_y), "蓝 " + va, HORIZONTAL_ALIGNMENT_RIGHT, 74.0, fs, COL_BLUE)
	draw_string(font, Vector2(rect.end.x - 78.0, title_y), "红 " + vb, HORIZONTAL_ALIGNMENT_RIGHT, 74.0, fs, COL_RED)

	# 绘图区（超采样时按列抽稀）
	var plot := Rect2(rect.position + Vector2(10.0, fs + 22.0), rect.size - Vector2(20.0, fs + 42.0))
	if plot.size.x < 10.0 or plot.size.y < 10.0:
		return
	var stride := maxi(1, int(float(_n) / (plot.size.x * 2.0))) # step size to resample a/b fed in
	var pa := PackedVector2Array()
	var pb := PackedVector2Array()
	var i := 0
	while i < _n:
		var t := float(i) / float(_n - 1)
		var x := plot.position.x + plot.size.x * t
		pa.append(Vector2(x, plot.position.y + plot.size.y * (1.0 - clampf(a[i] / ymax, 0.0, 1.0))))
		pb.append(Vector2(x, plot.position.y + plot.size.y * (1.0 - clampf(b[i] / ymax, 0.0, 1.0))))
		i += stride
	if (_n - 1) % stride != 0:
		var tx := plot.position.x + plot.size.x
		pa.append(Vector2(tx, plot.position.y + plot.size.y * (1.0 - clampf(a[_n - 1] / ymax, 0.0, 1.0))))
		pb.append(Vector2(tx, plot.position.y + plot.size.y * (1.0 - clampf(b[_n - 1] / ymax, 0.0, 1.0))))
	draw_polyline(pa, COL_BLUE, 1.5, true)
	draw_polyline(pb, COL_RED, 1.5, true)

	# 当前步游标 + 数值圆点
	var cx := plot.position.x + plot.size.x * float(_cursor) / float(_n - 1)
	draw_line(Vector2(cx, plot.position.y), Vector2(cx, plot.position.y + plot.size.y), COL_CURSOR, 1.0)
	draw_circle(Vector2(cx, plot.position.y + plot.size.y * (1.0 - clampf(a[_cursor] / ymax, 0.0, 1.0))), 3.0, COL_BLUE)
	draw_circle(Vector2(cx, plot.position.y + plot.size.y * (1.0 - clampf(b[_cursor] / ymax, 0.0, 1.0))), 3.0, COL_RED)

	# 轴标注
	var axis_y := plot.position.y + plot.size.y + 13.0
	draw_string(font, Vector2(plot.position.x, axis_y), "0", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, COL_DIM)
	draw_string(font, Vector2(plot.position.x, axis_y), "%d 步" % (_n - 1), HORIZONTAL_ALIGNMENT_RIGHT, plot.size.x, 11, COL_DIM)
	draw_string(font, Vector2(plot.position.x, plot.position.y - 3.0), _fmt(ymax, vfmt, vscale), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, COL_DIM)


func _fmt(v: float, vfmt: String, vscale: float) -> String:
	return vfmt % [int(round(v * vscale))]
