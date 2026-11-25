/* platforms/site/deck.js
   Điều khiển deck + đổ số từ data.json vào các phần tử có [data-bind].

   Hai quy tắc, giống `tools/manim/`:
   1. **Không con số nào viết trong file này.** Số đến từ `data.json` (do
      `build.py` sinh ra từ TSV paper trong repo). Ở đây chỉ có logic.
   2. **Thiếu dữ liệu thì hiện lỗi rõ ràng**, không im lặng hiện "0" — cùng
      nguyên tắc với `data.py::DataMissing` trong video.
*/

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const slides = $$(".slide");
  const dotsBox = $("#dots");
  let index = 0;

  /* ---------- dữ liệu ---------- */
  const fmt = {
    num: (v) => String(v),
    pct: (v) => (typeof v === "number" ? v.toFixed(1) : String(v)),
    pct2: (v) => (typeof v === "number" ? v.toFixed(2) : String(v)),
  };

  /** Lấy giá trị theo đường dẫn "a.b.c"; trả undefined nếu thiếu. */
  function pick(obj, path) {
    return path.split(".").reduce((acc, key) => (acc == null ? undefined : acc[key]), obj);
  }

  function bindAll(data) {
    let missing = 0;
    $$("[data-bind]").forEach((el) => {
      const key = el.dataset.bind;
      const raw = pick(data, key);
      if (raw === undefined || raw === null || raw === "") {
        el.textContent = "⚠ thiếu dữ liệu: " + key;   // không giả vờ có số
        el.classList.add("banner--warn");
        missing += 1;
        return;
      }
      el.textContent = fmt.num(raw);
    });
    return missing;
  }

  /* ---------- render các danh sách ---------- */
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function renderPaper(data) {
    const tbody = $("#paper-table tbody");
    const det = data.paper.detectors[0];        // hàng headline = detector đầu tiên
    const max = Math.max(...Object.values(det.acc));
    tbody.innerHTML = "";
    data.paper.generators.forEach((g) => {
      const tr = el("tr");
      tr.append(el("td", null, g));
      const accTd = el("td");
      const box = el("div", "cell-bar");
      const bar = el("i");
      bar.style.width = Math.max(6, (det.acc[g] / max) * 54) + "px";
      box.append(el("span", null, fmt.pct(det.acc[g])), bar);
      accTd.append(box);
      tr.append(accTd, el("td", null, fmt.pct(det.ap[g])));
      tbody.append(tr);
    });

    const list = $("#det-list");
    list.innerHTML = "";
    data.paper.detectors.forEach((d) => {
      const li = el("li");
      li.append(el("b", null, d.label), el("span", null, fmt.pct2(d.mean_acc) + " %"),
                el("em", null, "trung bình " + data.paper.generators.length + " ô"));
      list.append(li);
    });
  }

  function renderPaperComparisons(data) {
    const comparison = data.paper.comparisons;
    const baselineBody = $("#baseline-table tbody");
    baselineBody.replaceChildren();
    comparison.detector_baselines.forEach((row) => {
      const tr = el("tr");
      tr.append(el("th", null, row.method),
                el("td", null, fmt.pct(row.acc) + "%"),
                el("td", null, fmt.pct(row.ap) + "%"),
                el("td", null, row.train_set));
      baselineBody.append(tr);
    });

    const tbody = $("#table5-table tbody");
    const label = { REC: "REC", RGB: "RGB", RGB_DIRE: "RGB + DIRE", DIRE: "DIRE" };
    const order = ["REC", "RGB", "RGB_DIRE", "DIRE"];
    const generators = ["adm", "ddpm", "iddpm", "pndm", "sdv1"];
    tbody.replaceChildren();
    order.forEach((representation) => {
      const tr = el("tr");
      tr.append(el("th", null, label[representation]));
      generators.forEach((generator) => {
        const result = comparison.table5[representation]?.[generator];
        tr.append(el("td", null, result
          ? fmt.pct(result.acc) + "/" + fmt.pct(result.ap)
          : "—"));
      });
      tbody.append(tr);
    });
    $("#sdv1-acc-delta").textContent = "+" + fmt.pct(comparison.sdv1_delta_pp.acc);
    $("#sdv1-ap-delta").textContent = "+" + fmt.pct(comparison.sdv1_delta_pp.ap);
  }

  function renderStatus(data) {
    const list = $("#status-list");
    list.innerHTML = "";
    const DOT = { done: "dot--run", run: "dot--run", todo: "dot--todo", out: "dot--out" };
    data.status.items.forEach((s) => {
      const li = el("li");
      const dot = el("span", "dot " + (DOT[s.state] || "dot--out"));
      dot.setAttribute("aria-hidden", "true");
      const body = el("div");
      body.append(el("div", "nm", s.name));
      // `**đậm**` trong ghi chú ⇒ <b>; phần còn lại là text thuần (không innerHTML
      // cho dữ liệu sinh ra từ repo: tránh chèn HTML ngoài ý muốn).
      const nt = el("div", "nt");
      String(s.note).split(/(\*\*[^*]+\*\*)/).forEach((part) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          nt.append(el("b", null, part.slice(2, -2)));
        } else if (part) {
          nt.append(document.createTextNode(part));
        }
      });
      body.append(nt, el("div", "ref", s.ref));
      li.append(dot, body);
      list.append(li);
    });
  }

  /* ---------- điều hướng ---------- */
  function go(i, push = true) {
    index = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, k) => {
      if (k === index) s.setAttribute("data-active", "");
      else s.removeAttribute("data-active");
    });
    $("#idx").textContent = String(index + 1);
    $("#progress").style.width = ((index + 1) / slides.length) * 100 + "%";
    $$("#dots button").forEach((b, k) => b.setAttribute("aria-current", String(k === index)));
    slides[index].scrollTop = 0;
    if (push && location.hash !== "#" + (index + 1)) {
      history.replaceState(null, "", "#" + (index + 1));
    }
  }

  function buildDots() {
    $("#total").textContent = String(slides.length);
    slides.forEach((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("aria-label", "slide " + (i + 1) + ": " + (s.dataset.slide || ""));
      b.addEventListener("click", () => go(i));
      dotsBox.append(b);
    });
  }

  function init() {
    slides.forEach((s, i) => {
      if (i !== 0) return;
      s.setAttribute("data-active", "");
    });
    buildDots();
    const fromHash = parseInt(location.hash.replace("#", ""), 10);
    go(Number.isFinite(fromHash) && fromHash > 0 ? fromHash - 1 : 0, false);

    $("#next").addEventListener("click", () => go(index + 1));
    $("#prev").addEventListener("click", () => go(index - 1));

    const help = $("#help");
    document.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") { e.preventDefault(); go(index + 1); }
      else if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); go(index - 1); }
      else if (e.key === "Home") { e.preventDefault(); go(0); }
      else if (e.key === "End") { e.preventDefault(); go(slides.length - 1); }
      else if (e.key === "?") { e.preventDefault(); help.open ? help.close() : help.showModal(); }
      else if (e.key === "o" || e.key === "O") { document.body.classList.toggle("all"); }
      else if (e.key === "Escape" && help.open) { help.close(); }
    });

    // cuộn chuột: chuyển slide, có debounce để một cú lăn không nhảy 3 slide
    let locked = false;
    window.addEventListener("wheel", (e) => {
      if (document.body.classList.contains("all")) return;
      const active = slides[index];
      const canScroll = active.scrollHeight > active.clientHeight + 2;
      const atTop = active.scrollTop <= 0;
      const atBottom = active.scrollTop + active.clientHeight >= active.scrollHeight - 2;
      if (canScroll && ((e.deltaY > 0 && !atBottom) || (e.deltaY < 0 && !atTop))) return;
      if (locked) return;
      locked = true;
      setTimeout(() => { locked = false; }, 420);
      if (e.deltaY > 24) go(index + 1);
      else if (e.deltaY < -24) go(index - 1);
    }, { passive: true });

    // vuốt trên điện thoại
    let touchY = null;
    window.addEventListener("touchstart", (e) => { touchY = e.touches[0].clientY; }, { passive: true });
    window.addEventListener("touchend", (e) => {
      if (touchY == null) return;
      const dy = touchY - e.changedTouches[0].clientY;
      if (Math.abs(dy) > 40) go(index + (dy > 0 ? 1 : -1));
      touchY = null;
    }, { passive: true });
  }

  /* ---------- nạp dữ liệu ---------- */
  async function boot() {
    init();
    try {
      const res = await fetch("data.json", { cache: "no-cache" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      const missing = bindAll(data);
      renderPaper(data);
      renderPaperComparisons(data);
      renderStatus(data);
      $("#cover-date").textContent = String(new Date().getFullYear());
      if (missing) {
        console.warn("[" + missing + "] khoá dữ liệu bị thiếu trong data.json");
      }
    } catch (err) {
      // Không có dữ liệu thì phải NÓI RA ở đúng chỗ người xem đang nhìn: nhãn
      // trên bìa. Để nó mãi ở "đang tải…" là dạng nói dối nhẹ nhất và phổ biến
      // nhất của một trang demo.
      const pill = $('[data-bind="honesty.pill"]');
      if (pill) {
        pill.textContent = "⚠ không đọc được data.json — cần chạy build.py và mở qua http";
        pill.title = String(err);
        pill.removeAttribute("data-bind");
      }
      console.warn("[site] không đọc được data.json:", err);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
