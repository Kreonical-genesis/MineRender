# render_from_zips.py
"""
Render Minecraft model JSONs found inside ZIP resourcepacks.
- Reads all .zip from ./import/
- Extracts each to ./temp_import/<zipstem>/
- Finds models in assets/minecraft/models/**/*.json
- Resolves textures inside the extracted tree (assets/minecraft/textures/**)
- Renders simplified isometric PNG for each model to export/<zipstem>/<model_relative_path>.png

Dependencies:
pip install pillow numpy
"""

import zipfile
import shutil
import json
import os
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw
import math
import sys

# ------------------- DEBUG / cache -------------------
DEBUG = True            # поставь False чтобы отключить подробный debug/спам
TEXTURE_CACHE = {}      # кеш path_str -> PIL.Image
RESOLVE_CACHE = {}      # кеш tex_ref+ziproot -> Path (чтобы не резолвить повторно)
DEBUG_DIR = Path("debug")
if DEBUG:
    DEBUG_DIR.mkdir(exist_ok=True)


BASE_IN = Path("import")
TEMP_ROOT = Path("temp_import")
EXPORT_ROOT = Path("export")
EXPORT_ROOT.mkdir(exist_ok=True)
TEMP_ROOT.mkdir(exist_ok=True)

CANVAS_SIZE = (512, 512)
SCALE = 16
OFFSET = (CANVAS_SIZE[0] // 2, CANVAS_SIZE[1] // 2 + 40)


def compute_uv_box(uv, texture_img):
    """
    Вернёт (u0,v0,u1,v1) в пикселях, безопасно:
     - если UV в диапазоне 0..16, масштабирует по размеру текстуры
     - сортирует координаты (u0<=u1, v0<=v1)
     - обрезает по границам текстуры
     - гарантирует хотя бы 1x1 пиксель
    """
    tw, th = texture_img.size
    if not uv:
        return 0, 0, tw, th

    try:
        u0, v0, u1, v1 = map(float, uv)
    except Exception:
        return 0, 0, tw, th

    # Если все координаты <=16 — это вероятно UV в 0..16 пространстве -> масштабируем
    if max(abs(u0), abs(u1), abs(v0), abs(v1)) <= 16:
        sx = tw / 16.0
        sy = th / 16.0
        u0 *= sx; u1 *= sx; v0 *= sy; v1 *= sy

    # сортируем (на случай reverse UV)
    if u1 < u0:
        u0, u1 = u1, u0
    if v1 < v0:
        v0, v1 = v1, v0

    # clamp по границам текстуры
    u0 = max(0.0, min(float(tw), u0))
    u1 = max(0.0, min(float(tw), u1))
    v0 = max(0.0, min(float(th), v0))
    v1 = max(0.0, min(float(th), v1))

    # если degenerate (0 width/height) — расширим до 1 пикселя по возможности
    if int(u1) - int(u0) < 1:
        if int(u1) < tw:
            u1 = min(tw, u1 + 1.0)
        else:
            u0 = max(0.0, u0 - 1.0)
    if int(v1) - int(v0) < 1:
        if int(v1) < th:
            v1 = min(th, v1 + 1.0)
        else:
            v0 = max(0.0, v0 - 1.0)

    return int(u0), int(v0), int(u1), int(v1)


def paste_texture_on_quad(canvas: Image.Image, texture_img: Image.Image, uv_box, dest_quad, dbg_paths=None):
    """
    Пытается деформировать src(uv_box) -> dest_quad и вставить в canvas.
    Возвращает (applied: bool, method: 'transform'|'fallback'|'skip', reason:str)
    dbg_paths: dict с путями для отладки (может содержать 'before', 'after', 'crop', 'overlay')
    """
    try:
        u0, v0, u1, v1 = uv_box
    except Exception:
        return False, "skip", "invalid_uv"

    tw, th = texture_img.size
    if u0 >= u1 or v0 >= v1 or tw == 0 or th == 0:
        return False, "skip", "empty_uv"

    # crop исходный кусок
    src = texture_img.crop((u0, v0, u1, v1)).convert("RGBA")

    # вычислим целевой bbox
    xs = [int(round(p[0])) for p in dest_quad]
    ys = [int(round(p[1])) for p in dest_quad]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = max(1, maxx - minx)
    h = max(1, maxy - miny)

    # локальные координаты квадрата
    quad_local = []
    for (x, y) in dest_quad:
        quad_local.extend([x - minx, y - miny])

    # Отладочные сохранения
    if DEBUG and dbg_paths:
        try:
            # crop
            crop_path = dbg_paths.get("crop")
            if crop_path:
                src.save(crop_path)
            # overlay quad (на пустой картинке)
            ov = Image.new("RGBA", CANVAS_SIZE, (0,0,0,0))
            draw = ImageDraw.Draw(ov)
            draw.polygon([tuple(p) for p in dest_quad], outline=(255,0,0,255), width=2)
            overlay_path = dbg_paths.get("overlay")
            if overlay_path:
                ov.save(overlay_path)
        except Exception as e:
            print(f"[debug] failed saving crop/overlay: {e}", file=sys.stderr)

    # если bbox очень маленький — простая вставка
    if w < 2 or h < 2:
        resized = src.resize((w, h), Image.NEAREST)
        canvas.paste(resized, (minx, miny), resized)
        return True, "fallback", "small_bbox"

    # Попытка трансформировать (QUAD)
    try:
        transformed = src.transform((w, h), Image.QUAD, tuple(map(float, quad_local)), Image.BICUBIC)
        dst = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dst.paste(transformed, (0, 0), transformed)
        # debug: сохраним canvas BEFORE и AFTER если нужно — делаем это в вызывающем коде
        canvas.alpha_composite(dst, (minx, miny))
        return True, "transform", "ok"
    except Exception as e:
        # Логируем причину падения transform и делаем fallback
        if DEBUG:
            print(f"[paste_transform] transform failed: {e}. Falling back to resize paste.", file=sys.stderr)
        try:
            resized = src.resize((w, h), Image.NEAREST)
            canvas.paste(resized, (minx, miny), resized)
            return True, "fallback", "transform_exception"
        except Exception as e2:
            if DEBUG:
                print(f"[paste_fallback] fallback paste failed: {e2}", file=sys.stderr)
            return False, "skip", "fallback_failed"
    if u1 < u0:
        src = src.transpose(Image.FLIP_LEFT_RIGHT)
        u0, u1 = u1, u0
    if v1 < v0:
        src = src.transpose(Image.FLIP_TOP_BOTTOM)
        v0, v1 = v1, v0


def iso_project(x, y, z, scale=SCALE):
    # Minecraft approx isometric: 30° X, 45° Y
    px = int((x - z) * scale + OFFSET[0])
    py = int((x + z)/2 * scale - y * scale + OFFSET[1])
    return px, py


def rotate_elem_point(point, rotation):
    """
    point: (x, y, z)
    rotation: {"origin": [ox, oy, oz], "axis": "x|y|z", "angle": deg, "rescale": bool}
    Возвращает координаты после rotation
    """
    import math
    x, y, z = point
    if not rotation:
        return x, y, z
    ox, oy, oz = rotation.get("origin", [8,8,8])
    angle = math.radians(rotation.get("angle",0))
    axis = rotation.get("axis","y")
    # translate to origin
    x -= ox; y -= oy; z -= oz
    # rotate
    if axis=="x":
        y, z = y*math.cos(angle)-z*math.sin(angle), y*math.sin(angle)+z*math.cos(angle)
    elif axis=="y":
        x, z = x*math.cos(angle)+z*math.sin(angle), -x*math.sin(angle)+z*math.cos(angle)
    elif axis=="z":
        x, y = x*math.cos(angle)-y*math.sin(angle), x*math.sin(angle)+y*math.cos(angle)
    # translate back
    x += ox; y += oy; z += oz
    return x, y, z

def quad_for_face(elem_from, elem_to, face_name, rotation=None):
    try:
        x1, y1, z1 = elem_from
        x2, y2, z2 = elem_to
    except Exception:
        return None

    v = {
        "000": rotate_elem_point((x1, y1, z1), rotation),
        "001": rotate_elem_point((x1, y1, z2), rotation),
        "010": rotate_elem_point((x1, y2, z1), rotation),
        "011": rotate_elem_point((x1, y2, z2), rotation),
        "100": rotate_elem_point((x2, y1, z1), rotation),
        "101": rotate_elem_point((x2, y1, z2), rotation),
        "110": rotate_elem_point((x2, y2, z1), rotation),
        "111": rotate_elem_point((x2, y2, z2), rotation),
    }

    faces = {
        "north": ["100","101","001","000"],
        "south": ["011","010","110","111"],
        "west":  ["000","001","011","010"],
        "east":  ["110","100","000","010"],
        "up":    ["010","110","100","000"],
        "down":  ["101","111","011","001"]
    }

    keys = faces.get(face_name)
    if not keys:
        return None
    return [v[k] for k in keys]



def find_models(extracted_root: Path):
    """Find model jsons under assets/minecraft/models/**/*.json"""
    models_dir = extracted_root / "assets" / "minecraft" / "models"
    if not models_dir.exists():
        return []
    return list(models_dir.rglob("*.json"))


def resolve_texture_path(tex_ref: str, texture_map: dict, extracted_root: Path):
    """
    Улучшенное и кэшированное разрешение текстуры.
    Возвращает Path или None. Печать только при DEBUG.
    """
    if not tex_ref:
        return None

    cache_key = (tex_ref, str(extracted_root))
    if cache_key in RESOLVE_CACHE:
        return RESOLVE_CACHE[cache_key]

    # Если ссылка на ключ (например "#0"), возьмём значение из texture_map и рекурсивно
    if tex_ref.startswith("#"):
        key = tex_ref[1:]
        mapped = texture_map.get(key)
        if mapped is None:
            if DEBUG:
                print(f"[resolve] texture key '#{key}' not found in textures map", file=sys.stderr)
            RESOLVE_CACHE[cache_key] = None
            return None
        res = resolve_texture_path(mapped, texture_map, extracted_root)
        RESOLVE_CACHE[cache_key] = res
        return res

    val = tex_ref
    if ":" in val:
        val = val.split(":", 1)[1]

    # подготовим кандидатов (в порядке попыток)
    candidates = []
    candidates.append(Path("assets/minecraft/textures") / f"{val}.png")
    candidates.append(Path("assets/minecraft/textures") / val)
    base = Path(val)
    if not base.parts or base.parts[0] not in ("item", "block"):
        candidates.append(Path("assets/minecraft/textures") / "item" / f"{val}.png")
        candidates.append(Path("assets/minecraft/textures") / "block" / f"{val}.png")

    name = val.split("/")[-1]
    candidates.append(Path("assets/minecraft/textures") / f"{name}.png")
    candidates.append(Path("assets/minecraft/textures") / "item" / f"{name}.png")
    candidates.append(Path("assets/minecraft/textures") / "block" / f"{name}.png")

    # проверяем кандидатов
    tried = []
    for rel in candidates:
        p = extracted_root / rel
        tried.append(str(p))
        if p.exists():
            if DEBUG:
                print(f"[resolve] {tex_ref} -> {p}", file=sys.stderr)
            RESOLVE_CACHE[cache_key] = p
            return p

    # fallback: поиск по всему assets/**/textures/**/<name>.*
    for p in extracted_root.glob(f"assets/**/textures/**/{name}.*"):
        if p.suffix.lower() in (".png",):
            if DEBUG:
                print(f"[resolve fallback] {tex_ref} -> {p}", file=sys.stderr)
            RESOLVE_CACHE[cache_key] = p
            return p

    if DEBUG:
        print(f"[resolve FAILED] {tex_ref}, tried {len(tried)} candidates", file=sys.stderr)
    RESOLVE_CACHE[cache_key] = None
    return None


def render_model_to_image(model_json, extracted_root: Path):
    """
    Версия рендера с подробным debug'ом:
     - для каждой грани сохраняет crop, overlay, canvas_before, canvas_after
     - пишет краткий текстовый лог face_log.txt в debug-папку модели
    """
    canvas = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    texmap = model_json.get("textures", {}) or {}
    elements = model_json.get("elements", []) or []

    model_name_meta = model_json.get("__name") or model_json.get("name") or "model"
    safe_model = str(model_name_meta).replace("/", "_")
    zip_name = model_json.get("__source", "unknown_zip")
    dbg_model_dir = DEBUG_DIR / zip_name / safe_model
    if DEBUG:
        dbg_model_dir.mkdir(parents=True, exist_ok=True)
    face_log_lines = []

    # depth sorting
    def depth(e):
        try:
            fx = (e["from"][0] + e["to"][0]) / 2
            fz = (e["from"][2] + e["to"][2]) / 2
            fy = (e["from"][1] + e["to"][1]) / 2
            return (fz, -fy, fx)  # сортируем по z, потом y, потом x
        except Exception:
            return 0

    sorted_elems = sorted(elements, key=depth)

    for elem_idx, elem in enumerate(sorted_elems):
        f = elem.get("from")
        t = elem.get("to")
        if f is None or t is None:
            continue
        faces = elem.get("faces", {}) or {}
        for face_name, face_data in faces.items():
            uv = None
            texref = None
            if isinstance(face_data, dict):
                uv = face_data.get("uv")
                texref = face_data.get("texture")

            # компактный лог
            if DEBUG:
                print(f"[render] {zip_name}/{safe_model} elem#{elem_idx} face={face_name} texref={texref}", file=sys.stderr)

            # resolve texture path
            texpath = None
            if texref:
                texpath = resolve_texture_path(texref, texmap, extracted_root)
            if texpath is None and texmap:
                first = next(iter(texmap.values()))
                texpath = resolve_texture_path(first, texmap, extracted_root)

            if texpath is None:
                line = f"{face_name}: SKIP - texture not resolved (texref={texref})"
                face_log_lines.append(line)
                if DEBUG:
                    print(f"[render] {line}", file=sys.stderr)
                continue

            teximg = load_texture_cached(texpath)
            if teximg is None:
                line = f"{face_name}: SKIP - texture load failed ({texpath})"
                face_log_lines.append(line)
                if DEBUG:
                    print(f"[render] {line}", file=sys.stderr)
                continue


            uv_box = compute_uv_box(uv, teximg)
            quad3d = quad_for_face(f, t, face_name)
            if quad3d is None:
                line = f"{face_name}: SKIP - quad3d none"
                face_log_lines.append(line)
                if DEBUG:
                    print(f"[render] {line}", file=sys.stderr)
                continue
            quad2d = [iso_project(*pt) for pt in quad3d]

            # debug paths
            dbg_base = dbg_model_dir / f"{safe_model}__elem{elem_idx}__{face_name}"
            crop_path = dbg_base.with_suffix(".crop.png")
            overlay_path = dbg_base.with_suffix(".overlay.png")
            canvas_before_path = dbg_base.with_suffix(".before.png")
            canvas_after_path = dbg_base.with_suffix(".after.png")

            # save canvas before
            if DEBUG:
                try:
                    canvas.save(canvas_before_path)
                except Exception as e:
                    print(f"[debug] couldn't save canvas_before: {e}", file=sys.stderr)

            # call paste and get result
            applied, method, reason = paste_texture_on_quad(canvas, teximg, uv_box, quad2d,
                                                           dbg_paths={"crop": crop_path, "overlay": overlay_path})
            # save canvas after
            if DEBUG:
                try:
                    canvas.save(canvas_after_path)
                except Exception as e:
                    print(f"[debug] couldn't save canvas_after: {e}", file=sys.stderr)

            # compute some diagnostics about crop
            try:
                crop_img = Image.open(crop_path)
                # check if crop has non-transparent pixels
                bbox = crop_img.split()[-1].getbbox()  # alpha bbox
                crop_has_content = bbox is not None
            except Exception:
                crop_has_content = False

            # bbox of quad on canvas
            try:
                qminx = min(p[0] for p in quad2d)
                qminy = min(p[1] for p in quad2d)
                qmaxx = max(p[0] for p in quad2d)
                qmaxy = max(p[1] for p in quad2d)
                quad_bbox = (qminx, qminy, qmaxx, qmaxy)
            except Exception:
                quad_bbox = None

            # log line
            line = f"{face_name}: applied={applied} method={method} reason={reason} uv_box={uv_box} crop_content={crop_has_content} quad_bbox={quad_bbox} texpath={texpath}"
            face_log_lines.append(line)
            if DEBUG:
                print(f"[face_debug] {line}", file=sys.stderr)

    # save face_log
    try:
        log_path = dbg_model_dir / "face_log.txt"
        with open(log_path, "w", encoding="utf8") as lf:
            lf.write("\n".join(face_log_lines))
        if DEBUG:
            print(f"[debug] wrote face_log {log_path}", file=sys.stderr)
    except Exception as e:
        if DEBUG:
            print(f"[debug] couldn't write face_log: {e}", file=sys.stderr)

    return canvas

def load_texture_cached(path: Path):
    key = str(path)
    if key in TEXTURE_CACHE:
        return TEXTURE_CACHE[key]
    try:
        img = Image.open(path).convert("RGBA")
        TEXTURE_CACHE[key] = img
        if DEBUG:
            print(f"[load_texture] loaded {path}", file=sys.stderr)
        return img
    except Exception as e:
        TEXTURE_CACHE[key] = None
        print(f"[load_texture] failed to open {path}: {e}", file=sys.stderr)
        return None


def process_zip(zip_path: Path):
    zipstem = zip_path.stem
    target_dir = TEMP_ROOT / zipstem
    if target_dir.exists():
        print(f"Folder {target_dir} already exists — перезапись.")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {target_dir} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
    except Exception as e:
        print(f"Ошибка распаковки {zip_path}: {e}")
        return

    models = find_models(target_dir)
    print(f"Найдено {len(models)} моделей в {zip_path.name}")
    for model_path in models:
        try:
            with model_path.open("r", encoding="utf8") as f:
                model_json = json.load(f)
            # добавим метаданные для дебага
            models_root = target_dir / "assets" / "minecraft" / "models"
            try:
                rel = model_path.relative_to(models_root)
                model_json["__name"] = str(rel).replace(os.sep, "_")
            except Exception:
                model_json["__name"] = model_path.stem
            model_json["__source"] = zipstem

        except Exception as e:
            print(f"Ошибка чтения {model_path}: {e}")
            continue
        try:
            img = render_model_to_image(model_json, target_dir)
            # Формируем путь в export: export/<zipstem>/<path_rel_to_models_dir>.png
            models_root = target_dir / "assets" / "minecraft" / "models"
            try:
                rel = model_path.relative_to(models_root)
            except Exception:
                # в случае непредвиденной структуры — просто использовать имя файла
                rel = Path(model_path.stem + ".json")
            out_dir = EXPORT_ROOT / zipstem / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / (model_path.stem + ".png")
            img.save(out_file)
            print(f"Saved {out_file}")
        except Exception as e:
            print(f"Ошибка рендера модели {model_path}: {e}")

def main():
    zips = list(BASE_IN.glob("*.zip"))
    if not zips:
        print("Не найдено .zip в папке import/. Поместите zip-архивы туда.")
        return
    for z in zips:
        process_zip(z)
    print("Готово. Рендеры в папке export/, распакованные архивы в temp_import/")

if __name__ == "__main__":
    main()
