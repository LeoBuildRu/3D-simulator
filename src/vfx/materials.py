# -*- coding: utf-8 -*-
"""
Шейдеры абстрактного мира (GLSL 330).

Все материалы пишут цвет в ПРЕДУМНОЖЕННОМ виде (rgb уже умножен на альфу), а
смешивание — ONE / ONE_MINUS_SRC_ALPHA: одним режимом получаются и
полупрозрачные голограммы (альфа = покрытие), и чистое свечение (альфа ≈ 0,
цвет просто добавляется). Цвета — в HDR: всё ярче 1 подхватит bloom.

Униформы, общие для всей сцены (ставит компоновщик на корень):
    u_time  — секунды с начала сцены,
    u_view  — (ширина, высота) буфера в пикселях.
"""

from __future__ import annotations

from panda3d.core import Shader

_HEADER = "#version 330\n"

#: Общий «голографический» свет: френель по кромке, сканлинии по мировой
#: высоте, координатная сетка, мерцание, растворение с горящим краем.
HOLO_LIB = r"""
uniform float u_time;
uniform mat4 p3d_ViewMatrixInverse;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i), n100 = hash13(i + vec3(1,0,0));
    float n010 = hash13(i + vec3(0,1,0)), n110 = hash13(i + vec3(1,1,0));
    float n001 = hash13(i + vec3(0,0,1)), n101 = hash13(i + vec3(1,0,1));
    float n011 = hash13(i + vec3(0,1,1)), n111 = hash13(i + vec3(1,1,1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

// Радуга «карта высот»: синий -> голубой -> зелёный -> жёлтый -> красный.
vec3 rainbow(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c = clamp(vec3(
        1.5 - abs(4.0 * t - 3.0),
        1.5 - abs(4.0 * t - 2.0),
        1.5 - abs(4.0 * t - 1.0)), 0.0, 1.0);
    return c * c * 1.15 + vec3(0.02, 0.03, 0.06);
}

vec3 camera_pos() { return p3d_ViewMatrixInverse[3].xyz; }

// Основа голограммы. Возвращает предумноженный цвет.
vec4 holo_shade(vec3 world, vec3 n, vec3 base, float alpha,
                float fresnel_k, float scan_k, float grid_k, float flicker_k) {
    vec3 v = normalize(camera_pos() - world);
    n = normalize(n);
    float ndv = abs(dot(n, v));
    float fres = pow(1.0 - ndv, 2.2);
    float lambert = 0.35 + 0.65 * abs(dot(n, normalize(vec3(0.3, -0.4, 0.85))));
    // бегущие сканлинии по мировой высоте
    float scan = 0.5 + 0.5 * sin(world.z * 90.0 - u_time * 6.0);
    scan = pow(scan, 8.0);
    float band = smoothstep(0.0, 0.08, fract(world.z * 0.35 - u_time * 0.12))
               * (1.0 - smoothstep(0.08, 0.2, fract(world.z * 0.35 - u_time * 0.12)));
    // координатная сетка 0.5 м с антиалиасингом по производным
    vec2 g = abs(fract(world.xy * 2.0 - 0.5) - 0.5) / fwidth(world.xy * 2.0);
    float grid = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
    float flick = 1.0 - flicker_k * (0.5 + 0.5 * sin(u_time * 37.0 + world.y * 3.0))
                                 * step(0.93, vnoise(vec3(u_time * 3.0, world.y * 0.2, 0.0)));
    vec3 col = base * lambert * 0.55
             + base * fres * fresnel_k * 2.2
             + base * scan * scan_k * 0.8
             + base * band * scan_k * 1.2
             + base * grid * grid_k * 1.5;
    col *= flick;
    float a = clamp(alpha * (0.35 + 0.65 * fres + 0.4 * grid * grid_k), 0.0, 1.0);
    return vec4(col * alpha, a);
}

// Растворение: 0 — нет, 1 — полностью; горящий край шириной w.
vec2 dissolve(vec3 world, float amount, float w) {
    float n = vnoise(world * 3.5) * 0.7 + vnoise(world * 11.0) * 0.3;
    float cut = n - (amount * (1.0 + w) - w);
    return vec2(step(0.0, cut), 1.0 - smoothstep(0.0, w, cut));
}
"""

# --------------------------------------------------------------------------- #
# Голографическая поверхность (кузова, наполнитель, каркасные тела)
# --------------------------------------------------------------------------- #

HOLO_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
out vec3 v_world;
out vec3 v_normal;
void main() {
    vec4 w = p3d_ModelMatrix * p3d_Vertex;
    v_world = w.xyz;
    v_normal = mat3(p3d_ModelMatrix) * p3d_Normal;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

HOLO_FRAG = _HEADER + HOLO_LIB + r"""
uniform vec4 u_color;          // rgb — цвет, a — непрозрачность
uniform vec4 u_style;          // fresnel, scan, grid, flicker
uniform vec2 u_hrange;         // высоты для радуги (min, max)
uniform float u_rainbow;       // 0 — u_color, 1 — радуга по высоте
uniform float u_dissolve;      // 0..1
uniform vec4 u_reveal;         // xyz — нормаль плоскости, w — положение; всё,
                               // что «за» плоскостью, скрыто (горящий шов)
uniform vec4 u_highlight;      // xyz — центр, w — радиус подсветки
uniform vec3 u_dwave;          // фронт PBR по глубине (ставит компоновщик на корень)
uniform vec3 u_camFwd;         // направление взгляда камеры (мир)
uniform float u_waveCut;       // 1 — голограмма уходит вместе с проявлением PBR
in vec3 v_world;
in vec3 v_normal;
out vec4 o;
void main() {
    float side = dot(v_world, u_reveal.xyz) - u_reveal.w;
    if (side > 0.0) discard;
    float wseam = 0.0;
    if (u_waveCut > 0.5 && u_dwave.z > 0.5) {
        // та же глубина вдоль взгляда, что и у фронта в сведении
        float passed = u_dwave.x - dot(v_world - camera_pos(), u_camFwd);
        if (passed > u_dwave.y * 0.5) discard;
        wseam = exp(-pow(passed / max(u_dwave.y * 0.12, 1e-3), 2.0));
    }
    float seam = 1.0 - smoothstep(0.0, 0.12, -side);
    vec2 d = dissolve(v_world, u_dissolve, 0.08);
    if (d.x < 0.5) discard;
    float h = (v_world.z - u_hrange.x) / max(1e-4, u_hrange.y - u_hrange.x);
    vec3 base = mix(u_color.rgb, rainbow(h), u_rainbow);
    vec4 c = holo_shade(v_world, v_normal, base, u_color.a,
                        u_style.x, u_style.y, u_style.z, u_style.w);
    float hl = u_highlight.w > 0.0
             ? 1.0 - smoothstep(0.0, u_highlight.w, distance(v_world, u_highlight.xyz)) : 0.0;
    c.rgb += base * (seam * 4.0 + d.y * 3.0 + hl * 1.5) * u_color.a;
    c.rgb += vec3(0.4, 0.9, 1.0) * wseam * 2.5;
    c.a = clamp(c.a + wseam * 0.4, 0.0, 1.0);
    c.a = clamp(c.a + seam * 0.5 * u_color.a, 0.0, 1.0);
    o = c;
}
"""

# --------------------------------------------------------------------------- #
# Поверхность по картам высот с морфингом этапов (наполнитель)
# --------------------------------------------------------------------------- #

HEIGHTFIELD_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform sampler2D u_hA;        // r — высота, g — есть ли данные
uniform sampler2D u_hB;
uniform float u_front;         // фронт перехода A -> B по оси u_axis, 0..1
uniform float u_frontW;        // ширина фронта (доля)
uniform float u_axis;          // 0 — вдоль X сетки, 1 — вдоль Y
uniform vec3 u_grid;           // x0, y0, шаг сетки
uniform float u_rise;          // 0..1 — «вырастание» из пола (для появления)
uniform float u_floor;
in vec4 p3d_Vertex;            // xy — индекс ячейки
out vec3 v_world;
out vec3 v_local;
out vec3 v_normal;
out float v_valid;
out float v_front;

vec2 sample_h(ivec2 c, float k) {
    vec2 a = texelFetch(u_hA, c, 0).rg;
    vec2 b = texelFetch(u_hB, c, 0).rg;
    return mix(a, b, k);
}

void main() {
    ivec2 size = textureSize(u_hA, 0);
    ivec2 c = ivec2(p3d_Vertex.xy);
    vec2 t = vec2(c) / vec2(size - 1);
    float along = u_axis > 0.5 ? t.y : t.x;
    float k = smoothstep(u_front - u_frontW, u_front, along);
    k = 1.0 - k;                     // до фронта — уже B
    vec2 hv = sample_h(c, k);
    float hx0 = sample_h(clamp(c - ivec2(1,0), ivec2(0), size - 1), k).r;
    float hx1 = sample_h(clamp(c + ivec2(1,0), ivec2(0), size - 1), k).r;
    float hy0 = sample_h(clamp(c - ivec2(0,1), ivec2(0), size - 1), k).r;
    float hy1 = sample_h(clamp(c + ivec2(0,1), ivec2(0), size - 1), k).r;
    float z = mix(u_floor, hv.r, u_rise);
    vec3 local = vec3(u_grid.x + float(c.x) * u_grid.z,
                      u_grid.y + float(c.y) * u_grid.z, z);
    vec3 n = normalize(vec3((hx0 - hx1) * u_rise, (hy0 - hy1) * u_rise, 2.0 * u_grid.z));
    vec4 w = p3d_ModelMatrix * vec4(local, 1.0);
    v_world = w.xyz;
    v_local = local;
    v_normal = mat3(p3d_ModelMatrix) * n;
    v_valid = hv.g;
    v_front = 1.0 - smoothstep(0.0, u_frontW * 0.6, abs(along - u_front));
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(local, 1.0);
}
"""

HEIGHTFIELD_FRAG = _HEADER + HOLO_LIB + r"""
uniform vec4 u_color;
uniform vec4 u_style;
uniform vec2 u_hrange;
uniform float u_rainbow;
uniform float u_dissolve;
uniform vec4 u_reveal;
uniform float u_contours;      // изолинии высоты
uniform vec4 u_box;            // xmin, xmax, ymin, ymax наполнителя (локально)
uniform float u_cut;           // 0..1 — булева обрезка всего, что снаружи u_box
in vec3 v_world;
in vec3 v_local;
in vec3 v_normal;
in float v_valid;
in float v_front;
out vec4 o;
void main() {
    if (v_valid < 0.5) discard;
    if (dot(v_world, u_reveal.xyz) - u_reveal.w > 0.0) discard;
    vec2 d = dissolve(v_world, u_dissolve, 0.08);
    if (d.x < 0.5) discard;
    float outside = max(max(u_box.x - v_local.x, v_local.x - u_box.y),
                        max(u_box.z - v_local.y, v_local.y - u_box.w));
    float seam = 0.0;
    if (u_cut > 0.0) {
        float n = vnoise(v_world * 6.0);
        if (outside > 0.0 && u_cut * 1.25 - n * 0.25 > min(outside * 2.0, 1.0)) discard;
        seam = exp(-outside * outside * 900.0) * (1.0 - smoothstep(0.85, 1.0, u_cut)) * u_cut;
    }
    float h = (v_world.z - u_hrange.x) / max(1e-4, u_hrange.y - u_hrange.x);
    vec3 base = mix(u_color.rgb, rainbow(h), u_rainbow);
    vec4 c = holo_shade(v_world, v_normal, base, u_color.a,
                        u_style.x, u_style.y, u_style.z, u_style.w);
    float iso = abs(fract(h * 14.0) - 0.5) / fwidth(h * 14.0);
    c.rgb += base * (1.0 - clamp(iso, 0.0, 1.0)) * u_contours * u_color.a * 1.6;
    c.rgb += vec3(0.6, 0.9, 1.0) * v_front * 3.0 * u_color.a;
    c.rgb += base * d.y * 3.0 * u_color.a;
    c.rgb += vec3(1.0, 0.55, 0.2) * seam * 8.0;
    c.a = clamp(c.a + seam, 0.0, 1.0);
    o = c;
}
"""

# --------------------------------------------------------------------------- #
# Снимок: фото на меше глубины, без освещения
# --------------------------------------------------------------------------- #

PHOTO_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 v_uv;
out vec3 v_world;
void main() {
    v_uv = p3d_MultiTexCoord0;
    v_world = (p3d_ModelMatrix * p3d_Vertex).xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

PHOTO_FRAG = _HEADER + HOLO_LIB + r"""
uniform sampler2D u_photo;
uniform float u_alpha;
uniform float u_holo;          // 0 — фото, 1 — голографический тон
uniform float u_dissolve;
uniform vec4 u_scan;           // xyz — центр волны, w — радиус (сканирование)
uniform vec2 u_hrange;
in vec2 v_uv;
in vec3 v_world;
out vec4 o;
void main() {
    vec2 d = dissolve(v_world, u_dissolve, 0.1);
    if (d.x < 0.5) discard;
    vec3 photo = texture(u_photo, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
    photo = pow(photo, vec3(2.2));   // в линейное — компоновщик вернёт гамму
    float lum = dot(photo, vec3(0.299, 0.587, 0.114));
    float h = (v_world.z - u_hrange.x) / max(1e-4, u_hrange.y - u_hrange.x);
    vec3 holo = mix(vec3(0.05, 0.35, 0.6), rainbow(h), 0.55) * (0.25 + 1.4 * lum);
    vec3 col = mix(photo, holo, u_holo);
    float ring = 0.0;
    if (u_scan.w > 0.0) {
        float r = distance(v_world, u_scan.xyz) - u_scan.w;
        ring = exp(-r * r * 40.0);
    }
    col += vec3(0.4, 0.85, 1.0) * (ring * 2.5 + d.y * 3.0);
    o = vec4(col * u_alpha, u_alpha);
}
"""

# --------------------------------------------------------------------------- #
# Облако точек лидара
# --------------------------------------------------------------------------- #

POINTS_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform vec2 u_view;
uniform vec3 u_origin;         // сенсор (в системе узла)
uniform float u_throw;         // прогресс развёртки 0..1 (+ хвост)
uniform float u_fly;           // доля развёртки на полёт одной точки
uniform float u_size;          // размер точки, пикс. на 1 м… (см. ниже)
uniform vec4 u_alphas;         // фон, машина, общий, вспышка
uniform float u_rainbow;       // 0 — интенсивность, 1 — радуга по высоте
uniform vec2 u_hrange;
uniform vec4 u_tint;           // подсветка машины (rgb, сила)
uniform float u_collapse;      // 0..1 — стягивание фона в «пыль»
uniform float u_useColor;      // 1 — цвет точки из вершины (rgb), а не по интенсивности
uniform float u_maxPx;         // потолок размера точки, пиксели
in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec4 i_data;                // x — развёртка, y — метка, z — случайное, w — высота
out vec4 v_color;
out float v_head;

vec3 rainbow(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c = clamp(vec3(1.5 - abs(4.0*t - 3.0), 1.5 - abs(4.0*t - 2.0),
                        1.5 - abs(4.0*t - 1.0)), 0.0, 1.0);
    return c * c * 1.15 + vec3(0.02, 0.03, 0.06);
}

void main() {
    float start = i_data.x * (1.0 - u_fly);
    float p = clamp((u_throw - start) / max(u_fly, 1e-4), 0.0, 1.0);
    float travel = 1.0 - pow(1.0 - p, 3.0);
    vec3 pos = mix(u_origin, p3d_Vertex.xyz, travel);
    float truck = i_data.y;
    // фон «сдувает»: точки уходят вверх и в стороны, тая
    float blow = u_collapse * (1.0 - truck);
    pos += blow * vec3(sin(i_data.z * 40.0), cos(i_data.z * 53.0), 1.5 + i_data.z)
         * (0.6 + 2.2 * blow) * blow;
    vec4 clip = p3d_ModelViewProjectionMatrix * vec4(pos, 1.0);
    gl_Position = clip;
    float vis = step(0.0001, p) * mix(u_alphas.x, u_alphas.y, truck) * u_alphas.z;
    vis *= 1.0 - blow;
    // у самой камеры (мачта, столбы) точки гаснут, а не раздуваются в пятна
    vis *= smoothstep(1.5, 4.0, clip.w);
    v_head = (1.0 - smoothstep(0.85, 1.0, p)) * step(0.0001, p);
    float land = exp(-pow((u_throw - start - u_fly) * 30.0, 2.0)) * step(u_fly, u_throw - start + u_fly);
    float h = (i_data.w - u_hrange.x) / max(1e-4, u_hrange.y - u_hrange.x);
    vec3 base = mix(vec3(0.25, 0.75, 1.0) * (0.35 + 1.2 * p3d_Color.r), rainbow(h), u_rainbow);
    base = mix(base, p3d_Color.rgb, u_useColor);
    base = mix(base, u_tint.rgb, u_tint.a * truck);
    vec3 col = base + vec3(0.7, 0.95, 1.0) * (v_head * 0.9 + land * u_alphas.w * 3.0);
    v_color = vec4(col, 1.0) * vis * (1.0 - 0.45 * v_head);
    float px = u_size * u_view.y / max(clip.w, 0.05) * (1.0 + 0.3 * v_head + 1.5 * land * u_alphas.w);
    gl_PointSize = clamp(px, 1.0, u_maxPx) * step(0.001, vis);
}
"""

POINTS_FRAG = _HEADER + r"""
in vec4 v_color;
in float v_head;
out vec4 o;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(q, q);
    if (r2 > 1.0) discard;
    float core = exp(-r2 * 3.0);
    // точка — наполовину покрытие, наполовину свечение
    o = vec4(v_color.rgb * core, v_color.a * core * 0.55);
}
"""

# --------------------------------------------------------------------------- #
# Толстые линии (квады, расширяемые в экранном пространстве)
# --------------------------------------------------------------------------- #

LINES_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform vec2 u_view;
uniform float u_width;         // пиксели
in vec4 p3d_Vertex;            // этот конец
in vec3 i_other;               // другой конец отрезка
in vec4 i_line;                // x — сторона (-1/1), y — конец (0/1),
                               // z — длина до этой точки, w — длина всей линии
in vec4 p3d_Color;
out vec4 v_color;
out float v_dist;
out float v_total;
out float v_side;
void main() {
    vec4 a = p3d_ModelViewProjectionMatrix * vec4(p3d_Vertex.xyz, 1.0);
    vec4 b = p3d_ModelViewProjectionMatrix * vec4(i_other, 1.0);
    float aw = max(a.w, 0.02), bw = max(b.w, 0.02);
    vec2 sa = a.xy / aw * u_view, sb = b.xy / bw * u_view;
    vec2 dir = sb - sa;
    float len = length(dir);
    dir = len > 1e-5 ? dir / len : vec2(1.0, 0.0);
    if (i_line.y > 0.5) dir = -dir;      // у второго конца «другой» — начало
    vec2 nrm = vec2(-dir.y, dir.x);
    vec2 off = nrm * i_line.x * u_width / u_view;
    gl_Position = vec4(a.xy + off * aw, a.z, aw);
    v_color = p3d_Color;
    v_dist = i_line.z;
    v_total = i_line.w;
    v_side = i_line.x;
}
"""

LINES_FRAG = _HEADER + r"""
uniform float u_draw;          // 0..1 — сколько линии уже «прочерчено»
uniform float u_alpha;
uniform float u_glow;
uniform float u_dash;          // 0 — сплошная, иначе шаг штриха в метрах
uniform float u_time;
in vec4 v_color;
in float v_dist;
in float v_total;
in float v_side;
out vec4 o;
void main() {
    float drawn = u_draw * v_total;
    if (v_dist > drawn) discard;
    if (u_dash > 0.0 && fract(v_dist / u_dash - u_time * 0.5) > 0.55) discard;
    float edge = 1.0 - abs(v_side);
    float soft = smoothstep(0.0, 0.6, edge);
    float head = exp(-pow((drawn - v_dist) * 6.0, 2.0)) * step(u_draw, 0.999);
    vec3 col = v_color.rgb * (1.0 + u_glow * pow(edge, 3.0)) + vec3(1.0) * head * 5.0;
    float a = soft * v_color.a * u_alpha;
    o = vec4(col * a, a * 0.6);
}
"""

# --------------------------------------------------------------------------- #
# Искры/частицы (вся траектория в вершинном шейдере)
# --------------------------------------------------------------------------- #

SPARKS_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform vec2 u_view;
uniform float u_time;          // секунды от запуска этого облака искр
uniform vec3 u_gravity;
uniform float u_drag;
uniform float u_size;
uniform float u_alpha;
in vec4 p3d_Vertex;            // точка рождения
in vec3 i_vel;
in vec4 i_life;                // x — задержка, y — жизнь, z — случайное, w — оттенок
out vec4 v_color;
vec3 hue(float h) {
    return clamp(abs(fract(h + vec3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
}
void main() {
    float t = u_time - i_life.x;
    float life = clamp(t / i_life.y, 0.0, 1.0);
    float alive = step(0.0, t) * step(t, i_life.y);
    float k = u_drag > 0.0 ? (1.0 - exp(-u_drag * max(t, 0.0))) / u_drag : max(t, 0.0);
    vec3 pos = p3d_Vertex.xyz + i_vel * k + 0.5 * u_gravity * t * t;
    pos += vec3(sin(t * 7.0 + i_life.z * 30.0), cos(t * 6.0 + i_life.z * 20.0), 0.0) * 0.03 * t;
    vec4 clip = p3d_ModelViewProjectionMatrix * vec4(pos, 1.0);
    gl_Position = clip;
    float tw = 0.6 + 0.4 * sin(t * 40.0 + i_life.z * 100.0);
    float fade = (1.0 - life) * smoothstep(0.0, 0.08, life);
    vec3 c = mix(vec3(1.0), hue(i_life.w), 0.75) * (2.5 + 3.0 * (1.0 - life));
    v_color = vec4(c, 1.0) * fade * tw * alive * u_alpha;
    gl_PointSize = clamp(u_size * u_view.y / max(clip.w, 0.05) * (1.2 - life), 1.0, 20.0) * alive;
}
"""

SPARKS_FRAG = _HEADER + r"""
in vec4 v_color;
out vec4 o;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float r = length(q);
    float star = exp(-r * r * 6.0) + 0.6 * exp(-abs(q.x) * 14.0) * exp(-abs(q.y) * 2.5)
                                   + 0.6 * exp(-abs(q.y) * 14.0) * exp(-abs(q.x) * 2.5);
    if (star < 0.01) discard;
    o = vec4(v_color.rgb * star, v_color.a * star * 0.2);
}
"""

# --------------------------------------------------------------------------- #
# Пол абстрактного мира: бесконечная сетка, тающая к горизонту
# --------------------------------------------------------------------------- #

GRID_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
out vec3 v_world;
void main() {
    v_world = (p3d_ModelMatrix * p3d_Vertex).xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

GRID_FRAG = _HEADER + HOLO_LIB + r"""
uniform float u_alpha;
uniform vec4 u_pulse;          // xyz — центр, w — радиус кольца
uniform vec3 u_center;         // центр «сцены» — от него тает сетка
in vec3 v_world;
out vec4 o;
float lines(vec2 p, float scale) {
    vec2 g = abs(fract(p * scale - 0.5) - 0.5) / fwidth(p * scale);
    return 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
}
void main() {
    float dist = distance(v_world.xy, u_center.xy);
    float fade = exp(-dist * 0.12);
    float minor = lines(v_world.xy, 1.0) * 0.14;
    float major = lines(v_world.xy, 0.2) * 0.5;
    float ring = 0.0;
    if (u_pulse.w > 0.0) {
        float r = distance(v_world.xy, u_pulse.xy) - u_pulse.w;
        ring = exp(-r * r * 6.0);
    }
    vec3 col = vec3(0.08, 0.35, 0.75) * (minor + major) * fade
             + vec3(0.3, 0.8, 1.0) * ring * fade * 2.0;
    float a = u_alpha;
    o = vec4(col * a, (minor + major) * fade * 0.25 * a);
}
"""

# --------------------------------------------------------------------------- #
# Текст (поле расстояний шрифта) с голографическим свечением
# --------------------------------------------------------------------------- #

TEXT_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 v_uv;
out vec3 v_local;
void main() {
    v_uv = p3d_MultiTexCoord0;
    v_local = p3d_Vertex.xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

TEXT_FRAG = _HEADER + r"""
uniform sampler2D p3d_Texture0;
uniform vec4 u_color;
uniform float u_reveal;        // локальный X, до которого текст проявлен
uniform float u_glow;
uniform float u_time;
in vec2 v_uv;
in vec3 v_local;
out vec4 o;
void main() {
    if (v_local.x > u_reveal) discard;
    float d = texture(p3d_Texture0, v_uv).a;      // 0.5 — контур глифа
    float w = fwidth(d) * 0.75;
    float fill = smoothstep(0.5 - w, 0.5 + w, d);
    float glow = smoothstep(0.15, 0.5, d) * (1.0 - fill);
    float cursor = exp(-pow((u_reveal - v_local.x) * 8.0, 2.0));
    float scan = 0.85 + 0.15 * sin(v_local.z * 60.0 - u_time * 8.0);
    vec3 col = u_color.rgb * (fill * 1.6 * scan + glow * u_glow) + vec3(1.0) * cursor * fill * 3.0;
    float a = (fill + glow * 0.3) * u_color.a;
    o = vec4(col * u_color.a, a * 0.9);
}
"""

# --------------------------------------------------------------------------- #
# Постобработка: bloom (dual Kawase) и сведение
# --------------------------------------------------------------------------- #

QUAD_VERT = _HEADER + r"""
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 v_uv;
void main() {
    v_uv = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

BLOOM_DOWN_FRAG = _HEADER + r"""
uniform sampler2D u_src;
uniform float u_threshold;     // > 0 только у первого шага: отсечка яркости
in vec2 v_uv;
out vec4 o;
vec3 fetch(vec2 uv) {
    vec3 c = texture(u_src, uv).rgb;
    if (u_threshold > 0.0) {
        float l = max(c.r, max(c.g, c.b));
        float k = clamp((l - u_threshold) / max(l, 1e-4), 0.0, 1.0);
        c *= k * k;
    }
    return min(c, vec3(64.0));
}
void main() {
    vec2 px = 1.0 / vec2(textureSize(u_src, 0));
    vec3 s = fetch(v_uv) * 4.0;
    s += fetch(v_uv + vec2(-px.x, -px.y));
    s += fetch(v_uv + vec2( px.x, -px.y));
    s += fetch(v_uv + vec2(-px.x,  px.y));
    s += fetch(v_uv + vec2( px.x,  px.y));
    o = vec4(s / 8.0, 1.0);
}
"""

BLOOM_UP_FRAG = _HEADER + r"""
uniform sampler2D u_src;       // меньший уровень
uniform sampler2D u_base;      // этот уровень цепочки вниз
in vec2 v_uv;
out vec4 o;
void main() {
    vec2 px = 0.5 / vec2(textureSize(u_src, 0));
    vec3 s = texture(u_src, v_uv + vec2(-px.x * 2.0, 0.0)).rgb;
    s += texture(u_src, v_uv + vec2(-px.x, px.y)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(0.0, px.y * 2.0)).rgb;
    s += texture(u_src, v_uv + vec2(px.x, px.y)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(px.x * 2.0, 0.0)).rgb;
    s += texture(u_src, v_uv + vec2(px.x, -px.y)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(0.0, -px.y * 2.0)).rgb;
    s += texture(u_src, v_uv + vec2(-px.x, -px.y)).rgb * 2.0;
    o = vec4(s / 12.0 + texture(u_base, v_uv).rgb, 1.0);
}
"""

COMPOSITE_FRAG = _HEADER + r"""
uniform sampler2D u_scene;     // абстрактный мир, HDR, предумноженный
uniform sampler2D u_depth;     // его глубина
uniform sampler2D u_bloom;
uniform sampler2D u_world;     // готовый кадр RenderPipeline (с гаммой)
uniform sampler2D u_worldDepth;// глубина сцены RP
uniform float u_useWorldDepth;
uniform float u_abstract;      // 1 — абстрактный мир закрывает всё
uniform vec3 u_wave;           // xy — центр волны на экране (uv), z — радиус
uniform float u_waveWidth;
uniform float u_bloomK;
uniform float u_exposure;
uniform float u_aberration;
uniform float u_grain;
uniform float u_vignette;
uniform float u_glitch;
uniform float u_fade;          // затемнение всего кадра (0 — ничего)
uniform float u_time;
uniform vec4 u_bg;             // цвет пустоты (rgb) и сила её свечения
uniform vec2 u_view;
uniform float u_tonemap;       // 0 — только гамма (снимок 1:1), 1 — ACES
uniform vec3 u_dwave;          // фронт PBR по глубине: радиус (м), ширина (м), вкл.
uniform vec2 u_nearFar;        // ближняя/дальняя плоскости камеры
// «плоский» снимок станции поверх всего (вступление)
uniform sampler2D u_intro;
uniform float u_introMix;      // непрозрачность слоя снимка
uniform float u_undistort;     // 0 — как снято (фишай), 1 — как видит 3D-камера
uniform vec4 u_lens;           // f, cx, cy, f' (фокус 3D-камеры), пиксели снимка
uniform vec4 u_lensK;          // k1, k2, ширина, высота снимка
in vec2 v_uv;
out vec4 o;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec3 filmic(vec3 x) {           // ACES (Narkowicz) + гамма; u_tonemap 0 — только гамма
    x = max(x * u_exposure, vec3(0.0));
    vec3 aces = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
    return pow(mix(clamp(x, 0.0, 1.0), aces, u_tonemap), vec3(1.0 / 2.2));
}

// Пиксель экрана -> пиксель снимка: 0 — растянутый снимок как есть,
// 1 — та же точка, что показывает камера-обскура с фокусом f' (с дисторсией).
vec2 intro_uv(vec2 uv) {
    float W = u_lensK.z, H = u_lensK.w;
    float aspect = u_view.x / u_view.y;
    vec2 s = vec2((uv.x - 0.5) * H * aspect + 0.5 * W, (1.0 - uv.y) * H);
    vec2 xu = (s - u_lens.yz) / u_lens.w;
    float r2 = dot(xu, xu);
    vec2 d = u_lens.x * xu * (1.0 + u_lensK.x * r2 + u_lensK.y * r2 * r2) + u_lens.yz;
    vec2 p = mix(s, d, u_undistort);
    return vec2(p.x / W, 1.0 - p.y / H);
}

void main() {
    vec2 uv = v_uv;
    if (u_glitch > 0.0) {
        float row = floor(uv.y * 60.0);
        float g = step(1.0 - u_glitch * 0.35, hash(vec2(row, floor(u_time * 24.0))));
        uv.x += g * (hash(vec2(row, u_time)) - 0.5) * 0.06 * u_glitch;
    }
    vec2 dir = (uv - 0.5);
    float ab = u_aberration * dot(dir, dir);
    vec4 sc = texture(u_scene, uv);
    sc.r = texture(u_scene, uv + dir * ab).r;
    sc.b = texture(u_scene, uv - dir * ab).b;
    vec3 bloom = texture(u_bloom, uv).rgb * u_bloomK;

    // маска абстрактного мира: 1 — пустота, 0 — PBR; волна открывает PBR
    float mask = u_abstract;
    float ring = 0.0;
    if (u_wave.z > 0.0) {
        vec2 d = (v_uv - u_wave.xy) * vec2(u_view.x / u_view.y, 1.0);
        float r = length(d);
        float edge = u_waveWidth;
        mask *= smoothstep(u_wave.z - edge, u_wave.z, r);
        ring = exp(-pow((r - u_wave.z) / max(edge * 0.12, 1e-3), 2.0)) * step(0.001, u_abstract);
    }

    // фронт по глубине: пиксель PBR-мира ближе радиуса — уже открыт
    vec3 dfront = vec3(0.0);
    if (u_dwave.z > 0.5) {
        float dz = texture(u_worldDepth, v_uv).r * 2.0 - 1.0;
        float n = u_nearFar.x, f = u_nearFar.y;
        float lin = 2.0 * n * f / (f + n - dz * (f - n));
        float w = u_dwave.y;
        float k = smoothstep(u_dwave.x - w, u_dwave.x, lin);   // 1 — ещё пустота
        mask *= k;
        float edge = exp(-pow((lin - u_dwave.x) / max(w * 0.12, 1e-3), 2.0));
        // цифровые изолинии глубины сразу за фронтом
        float behind = clamp((u_dwave.x - lin) / (w * 2.5), 0.0, 1.0);
        float iso = 1.0 - smoothstep(0.02, 0.07, abs(fract(lin * 2.0) - 0.5));   // каждые 0.5 м
        float trail = (1.0 - behind) * step(lin, u_dwave.x) * iso;
        dfront = vec3(0.35, 0.85, 1.0) * (edge * 2.2 + trail * 0.35);
    }

    // голограмма за объектом PBR-мира не видна — там, где этот мир открыт
    float coverage = sc.a;
    if (u_useWorldDepth > 0.5 && mask < 0.999) {
        float dv = texture(u_depth, uv).r;
        float dw = texture(u_worldDepth, uv).r;
        if (dv < 0.99999 && dv > dw + 0.00002) {
            float k = 1.0 - mask;
            coverage *= 1.0 - k; sc.rgb *= 1.0 - k; bloom *= 1.0 - 0.75 * k;
        }
    }
    // кромка волны — тонкое радужное кольцо
    vec2 wd = (v_uv - u_wave.xy) * vec2(u_view.x / u_view.y, 1.0);
    float ang = atan(wd.y, wd.x);
    vec3 ringc = 0.55 + 0.45 * cos(6.2831 * (ang / 6.2831 + vec3(0.0, 0.33, 0.67)) + u_time * 2.0);
    bloom += ringc * ring * 0.9;
    sc.rgb += ringc * ring * 0.6 * (1.0 - coverage);

    vec3 world = texture(u_world, v_uv).rgb;
    // пустота: глубокий фон с лёгким виньетированным свечением
    vec2 c = (v_uv - 0.5) * vec2(u_view.x / u_view.y, 1.0);
    vec3 voidc = u_bg.rgb * (1.0 + u_bg.a * exp(-dot(c, c) * 3.0));
    vec3 under = mix(world, filmic(voidc), mask) + dfront * (1.0 - coverage);
    bloom += dfront * 0.6;
    vec3 over = filmic(sc.rgb + bloom);
    // предумноженное наложение в тонмапленном пространстве
    vec3 col = over + under * (1.0 - clamp(coverage, 0.0, 1.0));
    float vig = 1.0 - u_vignette * smoothstep(0.35, 1.05, length(v_uv - 0.5) * 1.35);
    col *= vig;
    col += (hash(v_uv * u_view + fract(u_time) * 100.0) - 0.5) * u_grain;
    if (u_introMix > 0.0) {
        vec2 iu = intro_uv(v_uv);
        float inside = step(0.0, iu.x) * step(iu.x, 1.0) * step(0.0, iu.y) * step(iu.y, 1.0);
        vec3 ph = texture(u_intro, iu).rgb * inside;
        col = mix(col, ph, u_introMix);
    }
    col *= 1.0 - u_fade;
    o = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def make(vert: str, frag: str) -> Shader:
    return Shader.make(Shader.SL_GLSL, vert, frag)


_cache: dict = {}


def shader(name: str) -> Shader:
    """Скомпилированный шейдер по имени (holo, heightfield, photo, points,
    lines, sparks, grid, text, bloom_down, bloom_up, composite)."""
    if name not in _cache:
        pairs = {
            "holo": (HOLO_VERT, HOLO_FRAG),
            "heightfield": (HEIGHTFIELD_VERT, HEIGHTFIELD_FRAG),
            "photo": (PHOTO_VERT, PHOTO_FRAG),
            "points": (POINTS_VERT, POINTS_FRAG),
            "lines": (LINES_VERT, LINES_FRAG),
            "sparks": (SPARKS_VERT, SPARKS_FRAG),
            "grid": (GRID_VERT, GRID_FRAG),
            "text": (TEXT_VERT, TEXT_FRAG),
            "bloom_down": (QUAD_VERT, BLOOM_DOWN_FRAG),
            "bloom_up": (QUAD_VERT, BLOOM_UP_FRAG),
            "composite": (QUAD_VERT, COMPOSITE_FRAG),
        }
        _cache[name] = make(*pairs[name])
    return _cache[name]
