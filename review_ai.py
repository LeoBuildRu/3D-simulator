#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_ai.py
============
Ручная приёмка AI-кадров датасета. Показывает каждый <кадр>_ai.png, по стрелкам
переключает на исходник <кадр>.png (и обратно), чтобы глазами сверить, что AI
не сдвинул/не переврал геометрию относительно GT. Вердикт с клавиатуры:

  ← / →        переключить показ AI  ↔  исходник (не-AI)
  ↑ / S        показать seg-карту (если есть), отпустить — вернуться
  Enter        ПРИНЯТЬ кадр, перейти к следующему
  Delete/Bksp  ОТКЛОНИТЬ: убрать _ai.png из папки, показать следующий
  U / Ctrl+Z   отменить последнее отклонение (вернуть файл, шаг назад)
  Home/End     первый / последний кадр
  Esc / Q      выйти (прогресс приёмки уже применён к файлам)

Отклонённые по умолчанию НЕ удаляются насовсем, а переносятся в подпапку
`rejected_ai/` рядом (можно вернуть). Флаг --hard-delete — удалять физически.

Запуск:
  python review_ai.py                         # папка по умолчанию из higgsfield_seedream_cli
  python review_ai.py --input-dir <папка>
  python review_ai.py <кадр1>_ai.png ...      # только указанные

Зависимости: tkinter (в составе Python) + Pillow.
"""

import argparse
import sys
from pathlib import Path

try:
    import tkinter as tk
    from PIL import Image, ImageTk
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"Нужны tkinter и Pillow: {e}")

# Значения по умолчанию берём из основного скрипта, чтобы не дублировать пути.
try:
    from higgsfield_seedream_cli import INPUT_DIR, OUTPUT_SUFFIX, SEG_SUFFIX
except Exception:  # noqa: BLE001
    INPUT_DIR = Path(r"D:\IQoko\dataset_segmentation_random")
    OUTPUT_SUFFIX = "_ai"
    SEG_SUFFIX = "_seg.png"

REJECT_DIRNAME = "rejected_ai"


def source_for(ai: Path) -> Path:
    """<stem>_ai.png -> <stem>.png"""
    stem = ai.stem
    if stem.endswith(OUTPUT_SUFFIX):
        stem = stem[: -len(OUTPUT_SUFFIX)]
    return ai.with_name(stem + ai.suffix)


def seg_for(ai: Path) -> Path:
    stem = ai.stem
    if stem.endswith(OUTPUT_SUFFIX):
        stem = stem[: -len(OUTPUT_SUFFIX)]
    return ai.with_name(stem + SEG_SUFFIX)


def discover_ai(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".png" and p.stem.endswith(OUTPUT_SUFFIX)
    )


class Reviewer:
    VIEW_AI, VIEW_SRC, VIEW_SEG = "AI", "ИСХОДНИК", "SEG"

    def __init__(self, root: tk.Tk, ai_files, reject_dir: Path, hard_delete: bool):
        self.root = root
        self.items = list(ai_files)           # оставшиеся к показу (в порядке)
        self.idx = 0
        self.reject_dir = reject_dir
        self.hard_delete = hard_delete
        self.kept = 0
        self.rejected = 0
        self.undo_stack = []                  # [(ai_path, moved_to_or_None, insert_idx)]
        self.view = self.VIEW_AI
        self._pil_cache = {}                  # path -> PIL.Image (full-res)
        self._tk_img = None                   # держим ссылку, иначе GC съест

        root.configure(bg="black")
        root.title("AI review")
        self.canvas = tk.Label(root, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(root, bg="#111", fg="#ddd", anchor="w",
                               font=("Consolas", 11), padx=8, pady=4)
        self.status.pack(fill="x", side="bottom")

        root.bind("<Left>", lambda e: self.toggle_src())
        root.bind("<Right>", lambda e: self.toggle_src())
        root.bind("<Up>", lambda e: self.show_seg(True))
        root.bind("<KeyRelease-Up>", lambda e: self.show_seg(False))
        root.bind("<s>", lambda e: self.show_seg(True))
        root.bind("<KeyRelease-s>", lambda e: self.show_seg(False))
        root.bind("<Return>", lambda e: self.keep())
        root.bind("<KP_Enter>", lambda e: self.keep())
        root.bind("<Delete>", lambda e: self.reject())
        root.bind("<BackSpace>", lambda e: self.reject())
        root.bind("<u>", lambda e: self.undo())
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Home>", lambda e: self.goto(0))
        root.bind("<End>", lambda e: self.goto(len(self.items) - 1))
        root.bind("<Escape>", lambda e: self.quit())
        root.bind("<q>", lambda e: self.quit())
        root.bind("<Configure>", lambda e: self._render())

        self.render()

    # ─── навигация ───
    def current(self):
        if 0 <= self.idx < len(self.items):
            return self.items[self.idx]
        return None

    def goto(self, i):
        if not self.items:
            return
        self.idx = max(0, min(i, len(self.items) - 1))
        self.view = self.VIEW_AI
        self.render()

    def advance(self):
        # остаёмся на том же idx (следующий элемент занял это место при удалении)
        # или сдвигаемся, если просто «принять»
        self.view = self.VIEW_AI
        if self.idx >= len(self.items):
            self.finish()
        else:
            self.render()

    # ─── действия ───
    def keep(self):
        if self.current() is None:
            return
        self.kept += 1
        self.idx += 1
        self.advance()

    def reject(self):
        ai = self.current()
        if ai is None:
            return
        moved_to = None
        try:
            if self.hard_delete:
                ai.unlink()
            else:
                self.reject_dir.mkdir(exist_ok=True)
                moved_to = self.reject_dir / ai.name
                if moved_to.exists():
                    moved_to.unlink()
                ai.replace(moved_to)
        except Exception as e:  # noqa: BLE001
            self._flash(f"НЕ УДАЛОСЬ удалить {ai.name}: {e}")
            return
        self.undo_stack.append((ai, moved_to, self.idx))
        self.items.pop(self.idx)
        self._pil_cache.pop(ai, None)
        self.rejected += 1
        # idx теперь указывает на следующий элемент (или конец)
        self.advance()

    def undo(self):
        if not self.undo_stack:
            self._flash("Отменять нечего")
            return
        ai, moved_to, insert_idx = self.undo_stack.pop()
        try:
            if moved_to is not None and moved_to.exists():
                moved_to.replace(ai)
            elif not ai.exists():
                self._flash(f"Файл удалён физически, вернуть нельзя: {ai.name}")
                return
        except Exception as e:  # noqa: BLE001
            self._flash(f"НЕ УДАЛОСЬ вернуть {ai.name}: {e}")
            return
        self.items.insert(insert_idx, ai)
        self.idx = insert_idx
        self.rejected -= 1
        self.view = self.VIEW_AI
        self.render()

    def toggle_src(self):
        self.view = self.VIEW_SRC if self.view == self.VIEW_AI else self.VIEW_AI
        self.render()

    def show_seg(self, on: bool):
        if on and self.view != self.VIEW_SEG:
            self._view_before_seg = self.view
            self.view = self.VIEW_SEG
            self.render()
        elif not on and self.view == self.VIEW_SEG:
            self.view = getattr(self, "_view_before_seg", self.VIEW_AI)
            self.render()

    # ─── отрисовка ───
    def _path_for_view(self, ai: Path):
        if self.view == self.VIEW_SRC:
            return source_for(ai)
        if self.view == self.VIEW_SEG:
            return seg_for(ai)
        return ai

    def _load(self, path: Path):
        img = self._pil_cache.get(path)
        if img is None:
            img = Image.open(path).convert("RGB")
            self._pil_cache[path] = img
        return img

    def render(self):
        if self.current() is None:
            if not self.items:
                self.finish()
                return
            self.idx = min(self.idx, len(self.items) - 1)
        self._render()
        self._update_status()

    def _render(self):
        ai = self.current()
        if ai is None:
            return
        path = self._path_for_view(ai)
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        if cw <= 1 or ch <= 1:
            self.root.after(30, self._render)
            return
        try:
            img = self._load(path)
        except Exception as e:  # noqa: BLE001
            self.canvas.config(image="", text=f"нет файла:\n{path.name}\n{e}", fg="#f66")
            self._tk_img = None
            return
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        disp = img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))),
                          Image.LANCZOS if scale < 1 else Image.NEAREST)
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.config(image=self._tk_img, text="")

    def _update_status(self, extra=""):
        ai = self.current()
        name = ai.name if ai else "—"
        total = len(self.items)
        pos = (self.idx + 1) if total else 0
        mode = "УДАЛ-НАВСЕГДА" if self.hard_delete else f"в {REJECT_DIRNAME}/"
        self.status.config(
            text=(f"[{pos}/{total}]  показ: {self.view:8}  |  {name}   "
                  f"|  принято {self.kept}  отклонено {self.rejected}  ({mode})   "
                  f"|  ←→ AI/исходник · ↑ seg · Enter принять · Del отклонить · U отменить · Esc выход"
                  + (f"    {extra}" if extra else ""))
        )

    def _flash(self, msg):
        self._update_status(msg)
        self.root.bell()

    def finish(self):
        self.canvas.config(image="", text=(
            f"Готово.\n\nПринято: {self.kept}\nОтклонено: {self.rejected}\n\n"
            + ("Отклонённые удалены физически." if self.hard_delete
               else f"Отклонённые перенесены в {self.reject_dir}")
            + "\n\nEsc — выход"), fg="#ddd", font=("Consolas", 16))
        self._tk_img = None
        self.status.config(text=f"Готово. принято {self.kept}, отклонено {self.rejected}. Esc — выход.")

    def quit(self):
        self.root.destroy()


def run(input_dir: Path, files=None, hard_delete=False) -> int:
    if files:
        ai_files = [f if f.is_absolute() else (Path.cwd() / f) for f in files]
        ai_files = [f for f in ai_files if f.is_file()]
        base = ai_files[0].parent if ai_files else input_dir
    else:
        if not input_dir.is_dir():
            raise SystemExit(f"Папка не найдена: {input_dir}")
        ai_files = discover_ai(input_dir)
        base = input_dir

    if not ai_files:
        print("Нет _ai.png для приёмки.")
        return 0

    print(f"К приёмке: {len(ai_files)} AI-кадров из {base}")
    root = tk.Tk()
    root.geometry("1400x820")
    Reviewer(root, ai_files, base / REJECT_DIRNAME, hard_delete)
    root.mainloop()
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Ручная приёмка AI-кадров датасета.")
    ap.add_argument("files", nargs="*", type=Path, help="Конкретные *_ai.png (иначе — вся папка).")
    ap.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    ap.add_argument("--hard-delete", action="store_true",
                    help=f"Удалять отклонённые физически (иначе — в {REJECT_DIRNAME}/).")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    sys.exit(run(a.input_dir, a.files, a.hard_delete))
