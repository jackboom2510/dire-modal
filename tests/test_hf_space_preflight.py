"""`tools/hf_space_preflight.py` — lỗi môi trường, không phải lỗi code.

Sự cố thật (Windows, Git Bash + miniforge), `hf repo create` chết bằng::

    File "...\\httpx\\_config.py", line 35, in create_ssl_context
        ctx = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
    FileNotFoundError: [Errno 2] No such file or directory

Không tên tệp, không tên biến. Test ở đây khoá ba điều:

1. biến CA trỏ tệp không tồn tại ⇒ báo ĐỎ và **nêu tên biến**;
2. đường dẫn kiểu MSYS (`/mingw64/ssl/...`) được nhận diện là nguyên nhân riêng,
   vì cách sửa khác với "tệp bị xoá";
3. token **không bao giờ** xuất hiện trong output — kể cả khi nó có trong env.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import hf_space_preflight as pf  # noqa: E402


def test_a_missing_ca_file_is_the_root_cause_and_says_which_variable(tmp_path):
    """Thông điệp phải gọi được tên biến — traceback của `ssl` thì không."""
    env = {"SSL_CERT_FILE": str(tmp_path / "không-có.pem")}
    report = pf.build_report("me/dire-demo", env=env, exists=Path.exists,
                             which=lambda _n: "/usr/bin/git", home=tmp_path)

    assert report["blocked"] is True
    assert [c["var"] for c in report["ca_broken"]] == ["SSL_CERT_FILE"]

    text = pf.render(report)
    assert "SSL_CERT_FILE" in text
    assert "FileNotFoundError" in text
    assert "unset SSL_CERT_FILE" in text
    assert "Remove-Item Env:SSL_CERT_FILE" in text


def test_msys_style_path_is_its_own_cause(tmp_path):
    """`/mingw64/ssl/certs/ca-bundle.crt` không phải "tệp bị xoá" mà là "sai kiểu
    đường dẫn": Python bản Windows đọc `/mingw64/...` thành `C:\\mingw64\\...`."""
    report = pf.build_report("me/dire-demo", env={"SSL_CERT_FILE": "/mingw64/ssl/certs/ca-bundle.crt"},
                             exists=lambda _p: False, which=lambda _n: "/usr/bin/git", home=tmp_path)
    broken = report["ca_broken"]
    assert len(broken) == 1 and broken[0]["msys"] is True
    assert "MSYS" in broken[0]["reason"]
    assert "MSYS" in pf.render(report)


def test_a_good_bundle_is_not_a_problem(tmp_path):
    ca = tmp_path / "cacert.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    report = pf.build_report("me/dire-demo", env={"SSL_CERT_FILE": str(ca)},
                             exists=Path.exists, which=lambda _n: "/usr/bin/git", home=tmp_path)
    assert report["blocked"] is False and report["ca_broken"] == []
    assert "VÌ SAO" not in pf.render(report)


def test_no_ca_variables_is_the_normal_case(tmp_path):
    report = pf.build_report("me/dire-demo", env={}, exists=Path.exists,
                             which=lambda _n: "/usr/bin/git", home=tmp_path)
    assert report["blocked"] is False
    assert "không đặt biến nào" in pf.render(report)


def test_the_token_is_never_printed_even_when_present(tmp_path, monkeypatch):
    """Token trong output là token trong log, trong ảnh chụp màn hình, trong chat."""
    secret = "hf_" + "x" * 34
    monkeypatch.setenv("HF_TOKEN", secret)
    report = pf.build_report("me/dire-demo", env={"HF_TOKEN": secret},
                             exists=Path.exists, which=lambda _n: "/usr/bin/git", home=tmp_path)
    assert report["token"] == "HF_TOKEN"
    text = pf.render(report) + json.dumps(report, ensure_ascii=False)
    assert secret not in text


def test_token_can_come_from_the_hub_token_file(tmp_path):
    p = tmp_path / ".cache" / "huggingface" / "token"
    p.parent.mkdir(parents=True)
    p.write_text("hf_abc\n", encoding="utf-8")
    report = pf.build_report("me/dire-demo", env={}, exists=Path.exists,
                             which=lambda _n: "/usr/bin/git", home=tmp_path)
    assert report["token"] == str(p)


def test_it_prints_the_current_cli_command_not_the_deprecated_one(tmp_path):
    """`hf repo` đã deprecated — in lệnh cũ là dạy người dùng lệnh sắp bị bỏ."""
    report = pf.build_report("me/dire-demo", env={}, exists=Path.exists,
                             which=lambda _n: "/usr/bin/git", home=tmp_path)
    text = pf.render(report)
    assert "hf repos create me/dire-demo" in text
    assert "hf repo create" not in text
    assert "new-space" in text                     # đường web UI, không dùng SSL env


def test_exit_code_is_one_only_when_blocked(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "thiếu.pem"))
    assert pf.main(["--space", "me/dire-demo"]) == 1
    capsys.readouterr()                      # bỏ output của lượt ĐỎ
    monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.setenv("HF_TOKEN", "hf_" + "y" * 34)
    assert pf.main(["--space", "me/dire-demo", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["blocked"] is False and out["space"] == "me/dire-demo"
