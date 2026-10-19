"""End-to-end browser flow: genuine in-domain CSV + synthetic scientific R4 fixture.

The R4 fixture is only used to exercise UI controls; its sentiment/emotion
predictions are placeholders. Actual Work R4 result import is additionally
checked offline against the signed handoff package, separately.
"""
import csv
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]
BASE="http://127.0.0.1:8501"

def wait_for_server(timeout=120):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        try:
            with urlopen(BASE+"/_stcore/health",timeout=3) as r:
                if r.status==200: return
        except (URLError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError("Streamlit server never became healthy")

def sample_csv(dest):
    g=pd.read_csv(ROOT/"data/validation/paired_axis_annotations_733.csv")
    sub=g.loc[g.item_id.isin([9,710,449]),["text"]].copy()
    assert len(sub)==3
    out=pd.concat([sub,pd.DataFrame({"text":["O bolo de chocolate ficou pronto.", ""]})],ignore_index=True)
    out.to_csv(dest,index=False)
    return out

def create_r4_fixture(dest):
    # The fixture uses real keys/gold but synthetic/constant 0.1 model scores.
    env=runpy.run_path(str(ROOT/"studio/test_work_r4.py"))
    dest.write_bytes(env["build"]())
    return dest

def run():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        csv_file=td/"corpus_proprio.csv"
        zip_file=td/"work_r4_synthetic_fixture.zip"
        sample_csv(csv_file)
        create_r4_fixture(zip_file)
        wait_for_server()
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
            page=browser.new_page(accept_downloads=True,viewport={"width":1440,"height":900})
            page.goto(BASE,wait_until="domcontentloaded",timeout=120000)
            expect(page.get_by_text("Axiological Studio").first).to_be_visible(timeout=60000)
            print("E2E_OPEN_HOME")
            page.locator("textarea").first.fill("Sinceramente precisamos ensinar sobre inteligência artificial na educação.")
            page.get_by_role("button",name="Analisar publicação").click()
            expect(page.get_by_text("Orientação · 3 classes")).to_be_visible(timeout=60000)
            print("E2E_NEW_TEXT_PREDICTION")
            page.get_by_role("tab",name="Analisar CSV").click()
            page.locator('input[type="file"]').first.set_input_files(str(csv_file))
            expect(page.get_by_text("5 registros.",exact=False)).to_be_visible(timeout=30000)
            page.get_by_role("button",name="Analisar lote").click()
            expect(page.get_by_role("button",name="Exportar resultados CSV")).to_be_visible(timeout=60000)
            with page.expect_download() as download_info:
                page.get_by_role("button",name="Exportar resultados CSV").click()
            download=download_info.value
            out_csv=td/"classified.csv"
            download.save_as(out_csv)
            records=pd.read_csv(out_csv)
            assert len(records)==5
            assert {"axiological_applicability","axiological_acceptance_3","axiological_intensity_3",
                    "axiological_review_recommended"}.issubset(records.columns)
            print("E2E_OWN_CSV_EXPORT",len(records))
            page.get_by_role("tab",name="Matriz do meu corpus").click()
            expect(page.get_by_text("Total processado")).to_be_visible(timeout=30000)
            expect(page.get_by_text("Contagens de predições por célula")).to_be_visible(timeout=30000)
            print("E2E_OWN_CORPUS_MATRIX")
            page.get_by_role("tab",name="Estudo Beyond Sentiment").click()
            page.locator('input[type="file"]').last.set_input_files(str(zip_file))
            expect(page.get_by_text("Pares válidos")).to_be_visible(timeout=60000)
            expect(page.get_by_text("Consenso nos dois eixos")).to_be_visible(timeout=30000)
            print("E2E_R4_UPLOAD_AND_RECONCILIATION")
            page.get_by_text("Explorar textos e conflitos").click()
            expect(page.get_by_role("button",name="Exportar recorte CSV")).to_be_visible(timeout=30000)
            with page.expect_download() as r4_export:
                page.get_by_role("button",name="Exportar recorte CSV").click()
            r4_out=td/"r4_recorte.csv"
            r4_export.value.save_as(r4_out)
            checked=pd.read_csv(r4_out)
            assert len(checked)==733
            print("E2E_R4_FILTER_EXPORT",len(checked))
            browser.close()
            print("E2E_ALL_PASS")

if __name__=="__main__":
    run()
