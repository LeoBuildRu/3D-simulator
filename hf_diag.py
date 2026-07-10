#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Диагностика загрузчика изображений в композиторе higgsfield.
Открывает автоматизационный профиль, печатает структуру file-input'ов и
кнопок загрузки, пробует открыть диалог и set_input_files, логирует сеть.
Окно остаётся открытым — посмотри глазами, появилась ли миниатюра."""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

USER_DATA = r"D:\IQoko\hf-chrome-profile"
PROFILE = "Default"
URL = "https://higgsfield.ai/ai/image?model=seedream_v5_lite"
TEST_IMG = r"renders\dataset_segmentation_random\r0001_vol0013.63_random_20260706_104559_373809.png"

DUMP_JS = r"""
() => {
  const info = {inputs: [], uploadButtons: []};
  const chain = (el) => {
    const out = [];
    for (let i=0; i<6 && el; i++, el=el.parentElement) {
      out.push({
        tag: el.tagName,
        role: el.getAttribute && el.getAttribute('role'),
        aria: el.getAttribute && el.getAttribute('aria-label'),
        cls: (el.className||'').toString().slice(0,60),
      });
    }
    return out;
  };
  document.querySelectorAll('input[type=file]').forEach((el,i) => {
    const r = el.getBoundingClientRect();
    info.inputs.push({
      i, accept: el.accept, cls: (el.className||'').slice(0,50),
      visible: el.offsetParent !== null,
      rect: {w: Math.round(r.width), h: Math.round(r.height)},
      html: el.outerHTML.slice(0,160),
      chain: chain(el.parentElement),
    });
  });
  // кандидаты-кнопки загрузки
  const rx = /upload|image|добав|photo|attach|reference|загруз/i;
  [...document.querySelectorAll('button,[role=button],label,div')].forEach(el => {
    const t = (el.getAttribute('aria-label')||'') + ' ' + (el.textContent||'').slice(0,25);
    if (rx.test(t) && el.offsetParent!==null) {
      const r = el.getBoundingClientRect();
      if (r.width>0 && r.width<400 && r.height>0 && r.height<200)
        info.uploadButtons.push({tag: el.tagName, aria: el.getAttribute('aria-label'),
          txt:(el.textContent||'').trim().slice(0,25), cls:(el.className||'').toString().slice(0,40)});
    }
  });
  info.uploadButtons = info.uploadButtons.slice(0,15);
  return info;
}
"""


async def main():
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=USER_DATA, channel="chrome", headless=False,
            args=[f"--profile-directory={PROFILE}"], no_viewport=True)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        errs = []
        page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        net = []
        page.on("response", lambda r: net.append(
            f"{r.request.method} {r.url.split('?')[0]}")
            if "higgsfield" in r.url and r.request.method in ("POST", "PUT") else None)

        await page.goto(URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        # убрать баннер cookies
        try:
            await page.evaluate("""() => {for (const id of ['cookiescript_injected_wrapper','cookiescript_injected']){const e=document.getElementById(id); if(e)e.remove();}}""")
        except Exception:
            pass
        await page.wait_for_timeout(500)

        import json
        info = await page.evaluate(DUMP_JS)
        print("\n===== FILE INPUTS =====")
        print(json.dumps(info["inputs"], ensure_ascii=False, indent=1))
        print("\n===== КАНДИДАТЫ-КНОПКИ ЗАГРУЗКИ =====")
        print(json.dumps(info["uploadButtons"], ensure_ascii=False, indent=1))

        # пробуем открыть file chooser кликом по ancestor label/button каждого инпута
        print("\n===== ПРОБА FILE CHOOSER =====")
        n = await page.locator('input[type=file]').count()
        for i in range(n):
            try:
                anc = page.locator('input[type=file]').nth(i).locator(
                    "xpath=ancestor::*[self::label or self::button][1]")
                async with page.expect_file_chooser(timeout=3000):
                    await anc.click()
                print(f"  input[{i}]: диалог ОТКРЫЛСЯ по клику предка")
            except Exception as e:
                print(f"  input[{i}]: диалог НЕ открылся ({type(e).__name__})")

        # set_input_files на input[0], ждём и смотрим сеть/превью
        print("\n===== SET_INPUT_FILES input[0] =====")
        net.clear()
        try:
            await page.locator('input[type=file]').first.set_input_files(str(Path(TEST_IMG)))
            await page.wait_for_timeout(8000)
            print(f"  сеть higgsfield POST/PUT: {net or 'ничего'}")
            imgs = await page.evaluate(
                "() => [...document.querySelectorAll('img')].filter(i=>/cloudfront|higgs|blob/.test(i.src)).length")
            print(f"  img(cloudfront/higgs/blob) на странице: {imgs}")
        except Exception as e:
            print(f"  set_input_files упал: {e}")

        if errs:
            print("\n===== CONSOLE ERRORS =====")
            for e in errs[-10:]:
                print("  " + e[:160])

        print("\nОкно останется открытым 60 сек — посмотри, появилась ли миниатюра "
              "картинки в композиторе, и попробуй сам перетащить/выбрать файл, "
              "наблюдая, что меняется.")
        await page.wait_for_timeout(60000)
        await ctx.close()


asyncio.run(main())
