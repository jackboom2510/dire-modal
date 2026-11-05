"""Test cho `platforms/site/` — trang pitch deck.

Vì sao có file này
------------------
Một trang giới thiệu là nơi số liệu dễ trôi nhất trong cả repo: HTML trông đúng
ngay cả khi số đã sai ba tháng. Ba lớp kiểm:

1. **Nguồn số.** `index.html` / `deck.js` không được chứa số liệu; mọi số đến từ
   `data.json`, và `data.json` phải khớp TSV / manifest / `demo_train.yaml`.
2. **Không lệch.** `data.json` phải được sinh lại y hệt (`build.py --check`),
   capital: sửa dữ liệu mà quên build là test đỏ.
3. **Trang tự đứng được.** Không CDN, không tài nguyên ngoài, mọi khoá
   `data-bind` phải có dữ liệu (thiếu thì trang hiện "⚠ thiếu dữ liệu", không
   hiện 0).

Chạy::

    python3 -m pytest tests/test_site.py -q
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "platforms" / "site"
SRC = ROOT / "src"
HTML = SITE / "index.html"
JS = SITE / "deck.js"
CSS = SITE / "style.css"
DATA = SITE / "data.json"

sys.path.insert(0, str(SRC))

#: Số liệu ĐẶC TRƯNG — không được xuất hiện như literal trong HTML/CSS/JS.
#: (Số cấu trúc như `width: 3px` thì không sao; đây là các con số *có nghĩa*.)
FORBIDDEN_LITERALS = [
    "99.9", "99,9", "98.7", "98,67", "97.9", "97,91",
    "76.6", "76,6", "125015131", "125011230", "3901", "3 901",
    "5.74", "1.94", "0.013", "$0.013",
]


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. data.json khớp nguồn trong repo
# ---------------------------------------------------------------------------
def test_data_json_exists_and_parses(data):
    assert data["schema"] == 1
    assert data["generated_by"] == "platforms/site/build.py"
    for key in ("paper", "pipeline", "demo", "measured", "status", "roadmap", "labels"):
        assert key in data, key


def test_paper_numbers_match_the_tsv(data):
    """Không có đường nào để bảng trên web lệch bảng trong paper_table2.tsv."""
    from dire_modal import eval as E  # noqa: PLC0415

    table = E.load_paper_table()
    for det in data["paper"]["detectors"]:
        for gen in data["paper"]["generators"]:
            assert det["acc"][gen] == table[det["id"]][gen]["acc"], (det["id"], gen, "acc")
            assert det["ap"][gen] == table[det["id"]][gen]["ap"], (det["id"], gen, "ap")
        # trung bình phải là trung bình 8 ô, không phải chép hàng `avg`
        mean = sum(det["acc"][g] for g in data["paper"]["generators"]) / 8
        assert det["mean_acc"] == pytest.approx(mean, abs=0.005)
        assert abs(det["mean_acc"] - det["avg_row"]["acc"]) <= 0.05


def test_paper_has_eight_columns_and_four_rows(data):
    assert len(data["paper"]["generators"]) == 8
    assert [d["label"] for d in data["paper"]["detectors"]] == ["ADM", "PNDM", "iDDPM", "StyleGAN"]


def test_pipeline_counts_come_from_the_manifest(data):
    from dire_modal.manifest import resolve_many  # noqa: PLC0415

    ents = resolve_many("e2")
    tars = [e for e in ents if e.remote.endswith(".tar.gz")]
    fakes = [e for e in tars if not e.remote.endswith("/real.tar.gz")]
    assert data["pipeline"]["items"] == len(ents)
    assert data["pipeline"]["tars"] == len(tars) == 16
    assert data["pipeline"]["pairs"] == len(fakes) == 15
    assert data["pipeline"]["containers"] == 15 * 4 == 60
    assert data["pipeline"]["detectors_count"] == len(data["paper"]["detectors"])


def test_demo_numbers_come_from_demo_train(data):
    """Chi phí trên trang phải là chi phí mà `demo_train` thật sự ước lượng."""
    from dire_modal import demo_train as DT  # noqa: PLC0415

    cfg = DT.load_config()
    est = DT.estimate(images=int(cfg["images"]), epochs=int(cfg["epochs"]),
                      batch=int(cfg["batch_size"]),
                      sec_per_step=float(cfg["sec_per_step"]),
                      usd_per_gpu_hour=float(cfg["usd_per_gpu_hour"]),
                      val_images=int(cfg["val_images"]))
    assert data["demo"]["steps"] == est["train_steps"]
    assert data["demo"]["usd"] == est["usd"]
    assert data["demo"]["seed"] == cfg["seed"] and data["demo"]["n_out"] == cfg["n_out"]
    # Chưa đo thì phải nói chưa đo.
    assert data["demo"]["measured"] is False


def test_every_number_carries_a_source(data):
    assert "ICCV" in data["paper"]["source"]
    assert "manifest" in data["pipeline"]["source"]
    assert "demo_train" in data["demo"]["source"]
    for key in ("baidu_kib_s", "ls_rounding", "fc_weight"):
        assert data["measured"][key]["source"].strip(), key


def test_labels_cover_all_three_kinds_of_number(data):
    """Ba loại nhãn = ba loại số. Không có loại thứ tư."""
    assert "paper" in data["labels"]["paper"].lower()
    assert "CHƯA" in data["labels"]["paper"]
    assert "paper" in data["labels"]["demo"]          # nhãn demo nói rõ nó KHÔNG phải paper


# ---------------------------------------------------------------------------
# 2. data.json không được cũ
# ---------------------------------------------------------------------------
def test_data_json_is_up_to_date():
    """Sửa TSV/manifest/demo_train.yaml mà quên `build.py` ⇒ test này đỏ."""
    r = subprocess.run([sys.executable, str(SITE / "build.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_build_is_deterministic():
    a = subprocess.run([sys.executable, str(SITE / "build.py"), "--stdout"],
                       capture_output=True, text=True, cwd=ROOT)
    assert a.returncode == 0, a.stderr
    b = subprocess.run([sys.executable, str(SITE / "build.py"), "--stdout"],
                       capture_output=True, text=True, cwd=ROOT)
    assert a.stdout == b.stdout == DATA.read_text(encoding="utf-8")


def test_build_only_uses_stdlib_plus_repo_code():
    tree_src = (SITE / "build.py").read_text(encoding="utf-8")
    imports = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", tree_src, re.M))
    assert not ({"numpy", "torch", "PIL", "flask", "jinja2"} & imports), imports


# ---------------------------------------------------------------------------
# 3. HTML/CSS/JS: không số liệu, không phụ thuộc ngoài
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["index.html", "deck.js", "style.css"])
def test_no_data_number_is_hardcoded(name):
    text = (SITE / name).read_text(encoding="utf-8")
    hits = [lit for lit in FORBIDDEN_LITERALS if lit in text]
    assert not hits, (
        f"{name} chứa số liệu viết tay {hits}. Số phải đến từ data.json "
        "(do build.py sinh ra) — xem docstring của build.py."
    )


def test_no_quantity_is_typed_into_the_markup():
    """Bắt MỌI cặp "số + đơn vị" trong HTML, không chỉ danh sách đen ở trên.

    Danh sách đen chỉ bắt được số đã biết; test này bắt cả số *tương lai* ai đó
    gõ vào markup (kiểu "400 epoch", "2 000 ảnh", "99.9 %").
    """
    html = HTML.read_text(encoding="utf-8")
    pat = r"\d+(?:[.,]\d+)*\s*(?:%|KiB|MiB|GiB|MB|GB|byte|epoch|step|slide)"
    hits = [h for h in re.findall(pat, html) if h.replace(" ", "") != "100%"]
    # "tải đủ 100 %" là cách nói, không phải số đo — ngoại lệ duy nhất.
    assert not hits, (
        f"index.html có số+đơn vị viết tay {hits}. Số phải do build.py sinh ra "
        "và bind bằng [data-bind]."
    )


def test_demo_scale_is_claimed_correctly(data):
    """So sánh quy mô demo vs tác giả phải lấy từ dữ liệu, không gõ tay."""
    adr = (ROOT / "docs" / "adr-0003-demo-training.md").read_text(encoding="utf-8")
    assert f"| {data['demo']['paper_epochs']} + earlystop 5 |" in adr, (
        "`paper_epochs` trên trang không khớp bảng trong ADR-0003"
    )
    assert data["demo"]["paper_epochs"] > data["demo"]["epochs"]
    # "images_epochs" là chuỗi hiển thị ⇒ phải khớp hai trường số bên cạnh nó.
    assert data["demo"]["images_epochs"] == f"{data['demo']['images']} × {data['demo']['epochs']}"


def test_every_data_bind_has_a_value(data):
    """Khoá trỏ sai ⇒ trang hiện '⚠ thiếu dữ liệu'. Test bắt trước khi mở trình duyệt."""
    keys = sorted(set(re.findall(r'data-bind="([^"]+)"', HTML.read_text(encoding="utf-8"))))
    assert len(keys) >= 20, f"quá ít khoá bind ({len(keys)}) — trang đang bỏ số ra ngoài?"

    def pick(obj, path):
        for part in path.split("."):
            if not isinstance(obj, dict) or part not in obj:
                return None
            obj = obj[part]
        return obj

    missing = [k for k in keys if pick(data, k) is None]
    assert not missing, f"khoá không có trong data.json: {missing}"


def test_no_external_resources():
    """Trang phải chạy offline: không CDN, không font Google, không analytics.

    `<a href="https://…">` thì được — đó là link người dùng *bấm*, không phải
    tài nguyên trình duyệt tự tải. Thứ bị cấm là `src`/`<link href>` trỏ ra
    ngoài, vì mất mạng là trang hỏng.
    """
    html = HTML.read_text(encoding="utf-8")
    for tag in re.findall(r"<(?:script|link|img|iframe|source)[^>]*>", html):
        assert "http" not in tag.lower(), f"thẻ tải tài nguyên ngoài: {tag}"
    for pat in ("fonts.googleapis", "google-analytics", "unpkg", "jsdelivr", "cdn."):
        assert pat not in html, f"index.html phụ thuộc ngoài: {pat}"


def test_css_imports_nothing():
    css = CSS.read_text(encoding="utf-8")
    assert "@import" not in css, "CSS không được import từ ngoài"
    assert "url(http" not in css, "CSS không được trỏ ra ngoài"


def test_palette_is_shared_with_the_video(monkeypatch):
    """Trang web và video manim phải dùng CÙNG một bảng màu.

    Hai nơi tự chọn màu riêng thì sớm muộn cũng lệch, và lúc đó không ai biết
    cái nào là chuẩn. Ở đây `style.py` là chuẩn; CSS phải chứa đúng các mã màu.
    """
    style = (ROOT / "tools" / "manim" / "style.py").read_text(encoding="utf-8")
    colors = dict(re.findall(r'"(\w+)":\s*"(#[0-9A-Fa-f]{6})"', style))
    assert colors, "không đọc được bảng màu từ tools/manim/style.py"
    css = CSS.read_text(encoding="utf-8").lower()
    for name in ("real", "fake", "map", "net", "ok", "warn"):
        assert colors[name].lower() in css, f"{name} ({colors[name]}) không có trong CSS"


# ---------------------------------------------------------------------------
# 4. Cấu trúc deck
# ---------------------------------------------------------------------------
def test_slides_are_ordered_and_complete():
    html = HTML.read_text(encoding="utf-8")
    names = re.findall(r'data-slide="([\w-]+)"', html)
    assert names[:2] == ["cover", "problem"]
    assert names[-1] == "end"
    assert len(names) >= 10, names
    assert len(set(names)) == len(names), "tên slide trùng"
    # Mỗi slide phải có tiêu đề (h1/h2) trừ slide bìa đã có h1.
    sections = re.findall(r"<section[^>]*data-slide=[^>]*>(.*?)</section>", html, re.S)
    assert len(sections) == len(names)
    for name, body in zip(names, sections):
        assert re.search(r"<h1|<h2", body), f"slide {name} không có tiêu đề"


def test_status_and_roadmap_are_not_rosy():
    """Trang không được tự khen: phải có mục 'chưa có' và một slide nói giới hạn."""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    states = {s["state"] for s in data["status"]["items"]}
    assert {"run", "todo"} <= states, states
    assert data["roadmap"], "roadmap rỗng"
    html = HTML.read_text(encoding="utf-8")
    assert 'data-slide="limits"' in html
    assert "không" in html.lower()


def test_js_binds_and_never_prints_zero_for_missing_data():
    js = JS.read_text(encoding="utf-8")
    assert "thiếu dữ liệu" in js, "thiếu khoá ⇒ phải báo, không được hiện 0"
    assert "không đọc được data.json" in js, "fetch hỏng ⇒ phải báo, không để 'đang tải…'"
    assert "data-bind" in js
    # `innerHTML` chỉ được dùng để XOÁ khung (`= ""`). Gán chuỗi sinh từ dữ liệu
    # vào innerHTML là đường để một file trong repo chèn được HTML vào trang.
    bad = re.findall(r"innerHTML\s*=\s*([^;\n]+)", js)
    assert all(v.strip().startswith('""') for v in bad), bad


def test_html_is_vietnamese_and_accessible():
    html = HTML.read_text(encoding="utf-8")
    assert 'lang="vi"' in html
    assert "aria-live" in html
    assert html.count("aria-label") >= 2
    # Có đường dự phòng khi JS hỏng: nội dung chữ nằm ngay trong HTML.
    assert "prefers-reduced-motion" in CSS.read_text(encoding="utf-8")
    assert "@media print" in CSS.read_text(encoding="utf-8")


def test_readme_documents_how_to_serve():
    readme = (SITE / "README.md").read_text(encoding="utf-8")
    assert "http.server" in readme
    assert "build.py" in readme
    assert "data.json" in readme


# ---------------------------------------------------------------------------
# 5. Trang chạy được thật (nếu máy có node + jsdom thì kiểm sâu hơn)
# ---------------------------------------------------------------------------
NODE = shutil.which("node")


@pytest.mark.skipif(not NODE, reason="máy không có node")
def test_page_renders_in_a_dom_if_node_and_jsdom_exist(tmp_path):
    """Tuỳ chọn: nếu có `jsdom` thì dựng DOM thật và kiểm số đã được đổ vào.

    Không bắt buộc (CI không cài node_modules), nhưng khi chạy được thì đây là
    phép kiểm mạnh nhất: nó bắt lỗi cú pháp JS, sai tên khoá, sai id phần tử.
    """
    probe = subprocess.run([NODE, "-e", "require.resolve('jsdom')"],
                           capture_output=True, text=True,
                           cwd=str(Path.home()))
    if probe.returncode != 0:
        pytest.skip("không có jsdom (npm i jsdom)")

    script = tmp_path / "check.js"
    script.write_text(
        "const fs=require('fs'),path=require('path');"
        "const {JSDOM,VirtualConsole}=require('jsdom');"
        f"const S='{SITE}';"
        "const html=fs.readFileSync(path.join(S,'index.html'),'utf8');"
        "const js=fs.readFileSync(path.join(S,'deck.js'),'utf8');"
        "const data=fs.readFileSync(path.join(S,'data.json'),'utf8');"
        "const errs=[];const vc=new VirtualConsole();"
        "vc.on('jsdomError',e=>errs.push(e.message));vc.on('error',m=>errs.push(m));"
        "vc.on('warn',m=>errs.push(m));"
        "const dom=new JSDOM(html,{runScripts:'outside-only',url:'http://localhost/',virtualConsole:vc});"
        "const w=dom.window;"
        "w.fetch=async()=>({ok:true,status:200,json:async()=>JSON.parse(data)});"
        "w.eval(js);"
        "setTimeout(()=>{"
        "const miss=[...w.document.querySelectorAll('[data-bind]')]"
        ".filter(e=>/thiếu dữ liệu/.test(e.textContent)).map(e=>e.dataset.bind);"
        "const rows=w.document.querySelectorAll('#paper-table tbody tr').length;"
        "const out={miss,rows,active:w.document.querySelectorAll('[data-active]').length,"
        "total:w.document.querySelector('#total').textContent,errs};"
        "console.log(JSON.stringify(out));"
        "},50);",
        encoding="utf-8",
    )
    r = subprocess.run([NODE, str(script)], capture_output=True, text=True,
                       cwd=str(Path.home()), timeout=120)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert payload["miss"] == []
    assert payload["rows"] == 8
    assert payload["active"] == 1
    assert payload["errs"] == []
