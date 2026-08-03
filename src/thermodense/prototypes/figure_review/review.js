/* PROTOTYPE — figure-review workbench (issue #15).
 *
 * Three structurally different variants of a local-first figure review loop,
 * switchable via ?variant=A|B|C from the floating bottom bar.
 *
 * A — Ledger:      dense list + master-detail, keyboard triage (j/k, i/s/r/e).
 * B — Board:       kanban columns by decision, drag & drop, compare modal.
 * C — Claim-first: claim-evidence audit, live manifest preview, export-centric.
 *
 * State is in-memory; export/import is a JSON review manifest (versioned).
 * Integrity (SHA-256) and journal-profile checks are advisory only — they must
 * never be taken as scientific approval.
 */
"use strict";

const VARIANTS = [
  { key: "A", name: "Ledger — keyboard triage" },
  { key: "B", name: "Board — drag & drop by decision" },
  { key: "C", name: "Claim-first — manifest audit" },
];

const DECISIONS = ["include", "supplement", "revise", "exclude"];
const COMMENT_TYPES = ["scientific", "limitation", "caption", "presentation"];
const VERDICTS = ["supported", "unsupported", "needs-work", "not-assessed"];
const CSS_PX_PER_CM = 96 / 2.54;
const CALIBRATION_KEY = "thermodense.figureReview.physicalScale.v1";

const state = {
  data: null,
  variant: initVariant(),
  appearance: "system",
  viewMode: "screen",
  printWidths: {},
  calibration: loadCalibration(),
  calibrationOpen: false,
  selectedFigure: null,
  selectedClaim: null,
  compareSel: [],
  modalFigure: null,
  review: { profile: "agu-wiley", figures: {}, claims: {} },
};

function initVariant() {
  const v = new URLSearchParams(location.search).get("variant");
  return VARIANTS.some((x) => x.key === v) ? v : "B";
}

function loadCalibration() {
  try {
    const saved = JSON.parse(localStorage.getItem(CALIBRATION_KEY));
    if (
      Number.isFinite(saved?.scale) &&
      saved.scale >= 0.5 &&
      saved.scale <= 2 &&
      Number.isFinite(saved?.dpr)
    ) {
      const current = Math.abs(saved.dpr - window.devicePixelRatio) < 0.01;
      return {
        scale: current ? saved.scale : 1,
        savedDpr: saved.dpr,
        saved: current,
        stale: !current,
      };
    }
  } catch {
    // Device calibration is optional; an invalid local value falls back safely.
  }
  return { scale: 1, savedDpr: null, saved: false, stale: false };
}

function calibrationIsCurrent() {
  return state.calibration.saved &&
    Math.abs(state.calibration.savedDpr - window.devicePixelRatio) < 0.01;
}

function calibrationIsStale() {
  return state.calibration.stale || (
    state.calibration.saved && !calibrationIsCurrent()
  );
}

function effectivePhysicalScale() {
  return calibrationIsStale() ? 1 : state.calibration.scale;
}

function physicalPx(cm) {
  return cm * CSS_PX_PER_CM * effectivePhysicalScale();
}

let observedDpr = window.devicePixelRatio;
window.addEventListener("resize", () => {
  if (Math.abs(observedDpr - window.devicePixelRatio) < 0.01) return;
  observedDpr = window.devicePixelRatio;
  if (state.calibration.saved) state.calibration.stale = true;
  if (state.data && state.viewMode === "print") render();
});

/* ---------------- data & review state ---------------- */

async function loadData() {
  const res = await fetch("data.json");
  state.data = await res.json();
  for (const f of state.data.figures) {
    state.review.figures[f.id] = { decision: null, comments: [] };
    state.printWidths[f.id] = Number(f.printWidthCm) || 8.5;
  }
  for (const c of state.data.claims) {
    state.review.claims[c.id] = { verdict: "not-assessed" };
  }
  render();
}

function figById(id) {
  return state.data.figures.find((f) => f.id === id);
}

function setDecision(figId, decision) {
  const r = state.review.figures[figId];
  r.decision = r.decision === decision ? null : decision;
  render();
}

function addComment(figId, panelId, type, text) {
  state.review.figures[figId].comments.push({
    level: panelId ? "panel" : "figure",
    target: panelId ? `${figId}#${panelId}` : figId,
    type,
    text,
    createdAt: new Date().toISOString(),
  });
  render();
}

function setVerdict(claimId, verdict) {
  state.review.claims[claimId].verdict = verdict;
  render();
}

function setProfile(profile) {
  state.review.profile = profile;
  render();
}

function setAppearance(appearance) {
  state.appearance = appearance;
  if (appearance === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.dataset.theme = appearance;
  }
}

function setViewMode(mode) {
  state.viewMode = mode === "print" ? "print" : "screen";
  render();
}

function applyPhysicalSizing(root = document) {
  for (const el of root.querySelectorAll("[data-physical-width-cm]")) {
    el.style.width = `${physicalPx(Number(el.dataset.physicalWidthCm))}px`;
  }
  for (const el of root.querySelectorAll("[data-physical-height-cm]")) {
    el.style.height = `${physicalPx(Number(el.dataset.physicalHeightCm))}px`;
  }
  for (const el of root.querySelectorAll("[data-physical-font-pt]")) {
    const pt = Number(el.dataset.physicalFontPt);
    el.style.fontSize = `${pt * (96 / 72) * effectivePhysicalScale()}px`;
  }
  for (const el of root.querySelectorAll("[data-physical-padding-x-cm]")) {
    const x = physicalPx(Number(el.dataset.physicalPaddingXCm));
    const y = physicalPx(Number(el.dataset.physicalPaddingYCm));
    el.style.padding = `${y}px ${x}px`;
  }
}

function saveCalibration() {
  state.calibration.saved = true;
  state.calibration.stale = false;
  state.calibration.savedDpr = window.devicePixelRatio;
  try {
    localStorage.setItem(CALIBRATION_KEY, JSON.stringify({
      scale: state.calibration.scale,
      dpr: state.calibration.savedDpr,
    }));
  } catch {
    // The current session still remains calibrated if storage is unavailable.
  }
  state.calibrationOpen = false;
  render();
}

/* ---------------- advisory checks ---------------- */

async function integrityChip(src, expectedSha256, label) {
  const el = document.createElement("span");
  el.className = "chip";
  el.textContent = `${label} SHA-256 …`;
  try {
    const buf = await (await fetch(src)).arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", buf);
    const hex = [...new Uint8Array(digest)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    const ok = hex === expectedSha256;
    el.className = "chip " + (ok ? "ok" : "bad");
    el.textContent = ok
      ? `${label} SHA-256 verified`
      : `${label} SHA-256 MISMATCH`;
    el.title = "advisory integrity check, not scientific approval";
  } catch {
    el.className = "chip warn";
    el.textContent = `${label} SHA-256 unavailable`;
  }
  return el;
}

function advisoryChips(fig) {
  const chips = document.createElement("div");
  chips.className = "chips";
  const renderer = document.createElement("span");
  renderer.className = "chip warn";
  renderer.textContent = "renderer: " + fig.provenance.renderer;
  renderer.title = "advisory — verify with the committed renderer before publication";
  const advisory = document.createElement("span");
  advisory.className = "chip";
  advisory.textContent = "advisory only — not scientific approval";
  chips.append(renderer, advisory);
  return chips;
}

function provenanceBlock(fig) {
  const div = document.createElement("div");
  div.className = "prov";
  const head = document.createElement("div");
  head.textContent = "Provenance — source artifacts";
  div.append(head);
  for (const a of fig.provenance.sourceArtifacts) {
    const p = document.createElement("div");
    const code = document.createElement("code");
    code.textContent = `${a.name}  sha256:${a.sha256.slice(0, 16)}…`;
    p.append(code);
    div.append(p);
  }
  return div;
}

/* ---------------- shared components ---------------- */

function decisionControls(figId) {
  const wrap = document.createElement("div");
  wrap.className = "decisions";
  const label = document.createElement("span");
  label.style.fontSize = "12px";
  label.style.color = "var(--muted)";
  label.textContent = "Decision:";
  wrap.append(label);
  for (const d of DECISIONS) {
    const b = document.createElement("button");
    b.textContent = d;
    const active = state.review.figures[figId].decision === d;
    b.classList.toggle("active", active);
    b.style.borderColor = active ? "var(--accent)" : "";
    b.onclick = () => setDecision(figId, d);
    wrap.append(b);
  }
  const clear = document.createElement("button");
  clear.className = "clear";
  clear.textContent = "clear";
  clear.onclick = () => setDecision(figId, null);
  wrap.append(clear);
  return wrap;
}

function decisionChip(figId) {
  const d = state.review.figures[figId].decision;
  const chip = document.createElement("span");
  chip.className = "chip decision " + (d ?? "");
  chip.textContent = d ?? "unreviewed";
  return chip;
}

function commentList(figId, panelId) {
  const div = document.createElement("div");
  const comments = state.review.figures[figId].comments.filter(
    (c) => (panelId ? c.target === `${figId}#${panelId}` : c.level === "figure")
  );
  if (!comments.length) {
    const none = document.createElement("div");
    none.style.color = "var(--muted)";
    none.style.fontSize = "12px";
    none.textContent = panelId ? "No panel comments yet." : "No figure comments yet.";
    div.append(none);
  }
  for (const c of comments) {
    const box = document.createElement("div");
    box.className = "comment";
    box.style.borderLeftColor =
      c.type === "scientific" ? "#2563eb" :
      c.type === "limitation" ? "#d97706" :
      c.type === "caption" ? "#16a34a" : "#8e44ad";
    const meta = document.createElement("div");
    meta.className = "meta";
    const type = document.createElement("span");
    type.className = "type";
    type.textContent = c.type;
    const level = document.createElement("span");
    level.textContent = c.level;
    meta.append(type, level);
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = c.text;
    box.append(meta, body);
    div.append(box);
  }
  return div;
}

function composer(figId, panelId) {
  const wrap = document.createElement("div");
  wrap.className = "composer";
  const row = document.createElement("div");
  row.className = "row";
  const type = document.createElement("select");
  for (const t of COMMENT_TYPES) {
    const o = document.createElement("option");
    o.value = t;
    o.textContent = t;
    type.append(o);
  }
  const ta = document.createElement("textarea");
  ta.placeholder = panelId
    ? `Comment on panel ${panelId} (${type.value})…`
    : "Comment on the whole figure…";
  type.onchange = () => {
    ta.placeholder = panelId
      ? `Comment on panel ${panelId} (${type.value})…`
      : `Comment on the whole figure (${type.value})…`;
  };
  const add = document.createElement("button");
  add.textContent = "Add";
  add.onclick = () => {
    if (!ta.value.trim()) return;
    addComment(figId, panelId, type.value, ta.value.trim());
  };
  row.append(type);
  wrap.append(row, ta, add);
  return wrap;
}

function claimCard(claim, clickable) {
  const card = document.createElement("div");
  card.className = "claim-card";
  const h = document.createElement("h4");
  h.textContent = "Claim " + claim.id + " — " + state.review.claims[claim.id].verdict;
  card.append(h);
  const txt = document.createElement("div");
  txt.className = "text";
  txt.textContent = claim.text;
  card.append(txt);
  const sel = document.createElement("select");
  for (const v of VERDICTS) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    sel.append(o);
  }
  sel.value = state.review.claims[claim.id].verdict;
  sel.onchange = () => setVerdict(claim.id, sel.value);
  const selWrap = document.createElement("div");
  selWrap.style.marginTop = "6px";
  selWrap.append(sel);
  card.append(selWrap);
  if (claim.evidenceFigureIds?.length) {
    const ev = document.createElement("div");
    ev.className = "evidence";
    const lbl = document.createElement("span");
    lbl.style.fontSize = "11px";
    lbl.style.color = "var(--muted)";
    lbl.textContent = "evidence:";
    ev.append(lbl);
    for (const fid of claim.evidenceFigureIds) {
      const f = figById(fid);
      if (!f) continue;
      const img = document.createElement("img");
      img.src = f.src;
      img.alt = f.title;
      img.title = f.title;
      if (clickable) {
        img.onclick = () => {
          state.selectedFigure = f.id;
          render();
        };
      }
      ev.append(img);
    }
    card.append(ev);
  }
  return card;
}

/* ---------------- header ---------------- */

function header() {
  const bar = document.createElement("header");
  bar.className = "topbar";
  const brand = document.createElement("div");
  brand.className = "brand";
  brand.innerHTML = "Figure review<span class='proto-tag'>PROTOTYPE</span>";
  bar.append(brand);

  const version = document.createElement("span");
  version.className = "chip";
  version.textContent = "figure set " + state.data.figureSetVersion;
  bar.append(version);

  const spacer = document.createElement("div");
  spacer.className = "spacer";
  bar.append(spacer);

  const profile = document.createElement("select");
  profile.title = "Journal profile (advisory limits)";
  for (const [key, p] of Object.entries(state.data.profiles)) {
    const o = document.createElement("option");
    o.value = key;
    o.textContent = p.name;
    profile.append(o);
  }
  profile.value = state.review.profile;
  profile.onchange = () => setProfile(profile.value);
  bar.append(profile);

  const appearance = document.createElement("select");
  appearance.className = "appearance-select";
  appearance.title = "Appearance";
  appearance.setAttribute("aria-label", "Appearance");
  for (const [value, label] of [
    ["system", "Appearance: System"],
    ["light", "Appearance: Light"],
    ["dark", "Appearance: Dark"],
  ]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    appearance.append(option);
  }
  appearance.value = state.appearance;
  appearance.onchange = () => setAppearance(appearance.value);
  bar.append(appearance);

  const view = document.createElement("select");
  view.className = "view-select";
  view.title = "Figure display scale";
  view.setAttribute("aria-label", "Figure display scale");
  for (const [value, label] of [
    ["screen", "View: Fit screen"],
    ["print", "View: Physical print size"],
  ]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    view.append(option);
  }
  view.value = state.viewMode;
  view.onchange = () => setViewMode(view.value);
  bar.append(view);

  const exp = document.createElement("button");
  exp.className = "primary";
  exp.textContent = "Export manifest";
  exp.onclick = exportManifest;
  bar.append(exp);

  const imp = document.createElement("button");
  imp.textContent = "Import";
  const file = document.createElement("input");
  file.type = "file";
  file.accept = "application/json,.json";
  file.style.display = "none";
  file.onchange = () => {
    if (file.files[0]) importManifest(file.files[0]);
  };
  imp.onclick = () => file.click();
  bar.append(imp, file);

  const banner = document.createElement("div");
  banner.className = "profile-banner";
  const p = state.data.profiles[state.review.profile];
  const strong = document.createElement("strong");
  strong.textContent = p.name + " advisory limits:";
  banner.append(strong, " " + p.warnings.join(" · "));
  const note = document.createElement("span");
  note.className = "advisory-note";
  note.textContent = "Checks are advisory — they do not imply scientific approval.";
  banner.append(note);

  const wrap = document.createElement("div");
  wrap.className = "app-header";
  wrap.append(bar, banner);
  return wrap;
}

/* ---------------- figure detail (shared by A and C) ---------------- */

function calibrationPanel() {
  const panel = document.createElement("section");
  panel.className = "calibration-panel";

  const heading = document.createElement("strong");
  heading.textContent = "Calibrate this display";
  const instructions = document.createElement("p");
  instructions.textContent =
    "Hold a physical ruler against the line. Adjust until the end marks are exactly 10.0 cm apart.";

  const rulerWrap = document.createElement("div");
  rulerWrap.className = "calibration-ruler-wrap";
  const ruler = document.createElement("div");
  ruler.className = "calibration-ruler";
  ruler.dataset.physicalWidthCm = "10";
  ruler.setAttribute("aria-label", "Ten centimetre calibration line");
  rulerWrap.append(ruler);

  const sliderRow = document.createElement("div");
  sliderRow.className = "calibration-slider";
  const minus = document.createElement("span");
  minus.textContent = "shorter";
  const range = document.createElement("input");
  range.type = "range";
  range.min = "0.5";
  range.max = "2";
  range.step = "0.001";
  range.value = String(state.calibration.scale);
  range.setAttribute("aria-label", "Physical display scale");
  const plus = document.createElement("span");
  plus.textContent = "longer";
  const readout = document.createElement("code");
  readout.textContent = `${state.calibration.scale.toFixed(3)}×`;
  range.oninput = () => {
    state.calibration.scale = Number(range.value);
    state.calibration.saved = false;
    readout.textContent = `${state.calibration.scale.toFixed(3)}×`;
    applyPhysicalSizing();
  };
  sliderRow.append(minus, range, plus, readout);

  const actions = document.createElement("div");
  actions.className = "calibration-actions";
  const save = document.createElement("button");
  save.className = "primary";
  save.textContent = "Save calibration";
  save.onclick = saveCalibration;
  const reset = document.createElement("button");
  reset.textContent = "Reset to browser estimate";
  reset.onclick = () => {
    state.calibration = { scale: 1, savedDpr: null, saved: false, stale: false };
    try {
      localStorage.removeItem(CALIBRATION_KEY);
    } catch {
      // Storage is optional; reset still applies to the current session.
    }
    render();
  };
  actions.append(save, reset);
  panel.append(heading, instructions, rulerWrap, sliderRow, actions);
  applyPhysicalSizing(panel);
  return panel;
}

function printSizeControls(fig) {
  const controls = document.createElement("section");
  controls.className = "print-size-controls";

  const widthLabel = document.createElement("label");
  widthLabel.textContent = "LaTeX figure width";
  const width = document.createElement("input");
  width.type = "number";
  width.min = "1";
  width.max = "30";
  width.step = "0.1";
  width.value = String(state.printWidths[fig.id]);
  width.setAttribute("aria-label", "LaTeX figure width in centimetres");
  const unit = document.createElement("span");
  unit.textContent = "cm";
  width.onchange = () => {
    state.printWidths[fig.id] = Math.max(1, Math.min(30, Number(width.value) || 8.5));
    render();
  };
  widthLabel.append(width, unit);
  controls.append(widthLabel);

  for (const value of [5, 8.5, 17.8]) {
    const preset = document.createElement("button");
    preset.textContent = `${value} cm`;
    preset.title = `Preview at ${value} cm wide`;
    preset.onclick = () => {
      state.printWidths[fig.id] = value;
      render();
    };
    controls.append(preset);
  }

  const calibration = document.createElement("button");
  calibration.textContent = calibrationIsCurrent()
    ? "Display calibrated"
    : "Calibrate physical scale";
  calibration.className = calibrationIsCurrent() ? "calibrated" : "needs-calibration";
  calibration.onclick = () => {
    if (calibrationIsStale()) {
      state.calibration = { scale: 1, savedDpr: null, saved: false, stale: true };
    }
    state.calibrationOpen = !state.calibrationOpen;
    render();
  };
  controls.append(calibration);

  const note = document.createElement("span");
  note.className = "physical-note";
  note.textContent = calibrationIsCurrent()
    ? "Actual size at the current browser zoom. Recalibrate after zooming or changing displays."
    : calibrationIsStale()
      ? "Browser zoom or display scale changed; the old calibration was disabled. Recalibrate."
      : "Browser estimate only until calibrated with a ruler.";
  controls.append(note);
  return controls;
}

function physicalPrintPreview(fig) {
  const widthCm = state.printWidths[fig.id];
  const stage = document.createElement("div");
  stage.className = "print-stage";
  const label = document.createElement("div");
  label.className = "print-stage-label";
  label.textContent = `A4 · actual-size target · 21 × 29.7 cm · figure ${widthCm} cm wide`;

  const sheet = document.createElement("div");
  sheet.className = "print-sheet";
  sheet.dataset.physicalWidthCm = "21";
  sheet.dataset.physicalHeightCm = "29.7";
  sheet.dataset.physicalPaddingXCm = "1.6";
  sheet.dataset.physicalPaddingYCm = "2";
  const figureBlock = document.createElement("div");
  figureBlock.className = "print-figure-block";
  figureBlock.dataset.physicalWidthCm = String(widthCm);
  const image = document.createElement("img");
  image.className = "figure print-figure";
  image.src = fig.src;
  image.alt = fig.title;
  const caption = document.createElement("p");
  caption.className = "print-caption";
  caption.dataset.physicalFontPt = "10";
  caption.textContent = `Figure. ${fig.caption}`;
  figureBlock.append(image, caption);
  sheet.append(figureBlock);
  stage.append(label, sheet);
  applyPhysicalSizing(stage);
  return stage;
}

async function figureDetail(figId, opts = {}) {
  const f = figById(figId);
  if (!f) return document.createElement("div");
  const main = document.createElement("div");
  main.className = opts.detailClass || "a-detail";
  const h = document.createElement("h2");
  h.textContent = f.title + "  " + (opts.showDecision ? "" : "");
  main.append(h);

  const chips = advisoryChips(f);
  const integrityChecks = [integrityChip(f.src, f.sha256, "Preview PNG")];
  if (f.publicationSrc && f.publicationSha256) {
    integrityChecks.push(
      integrityChip(f.publicationSrc, f.publicationSha256, "Publication PDF")
    );
  }
  const checkedChips = await Promise.all(integrityChecks);
  chips.prepend(...checkedChips);
  main.append(chips);

  if (f.publicationSrc) {
    const publication = document.createElement("a");
    publication.className = "publication-link";
    publication.href = f.publicationSrc;
    publication.download = "";
    publication.textContent = "Download vector PDF for LaTeX";
    main.append(publication);
  }

  if (state.viewMode === "print") {
    main.append(printSizeControls(f));
    if (state.calibrationOpen) main.append(calibrationPanel());
    main.append(physicalPrintPreview(f));
  } else {
    const img = document.createElement("img");
    img.className = "figure";
    img.src = f.src;
    img.alt = f.title;
    main.append(img);

    const cap = document.createElement("p");
    cap.style.fontSize = "12.5px";
    cap.style.color = "var(--muted)";
    cap.textContent = "Caption: " + f.caption;
    main.append(cap);
  }

  const panels = document.createElement("div");
  panels.className = "a-panels";
  panels.append(
    Object.assign(document.createElement("span"), {
      style: "font-size:12px;color:var(--muted)",
      textContent: "Panels:",
    })
  );
  for (const p of f.panels) {
    const pill = document.createElement("div");
    pill.className = "a-panel";
    const lbl = document.createElement("span");
    lbl.textContent = `(${p.id}) ${p.label}`;
    const btn = document.createElement("button");
    btn.textContent = "annotate";
    btn.onclick = () => {
      const box = pill.querySelector(".panel-composer");
      if (box) {
        box.remove();
        return;
      }
      const c = composer(figId, p.id);
      c.classList.add("panel-composer");
      pill.append(c);
    };
    pill.append(lbl, btn);
    panels.append(pill);
  }
  main.append(panels);

  const decWrap = document.createElement("div");
  decWrap.style.margin = "8px 0";
  decWrap.append(decisionControls(figId));
  main.append(decWrap);

  const claimsWrap = document.createElement("div");
  claimsWrap.style.margin = "10px 0";
  const ch = document.createElement("strong");
  ch.style.fontSize = "13px";
  ch.textContent = "Claim-evidence cards";
  claimsWrap.append(ch);
  const linked = state.data.claims.filter((c) => c.evidenceFigureIds.includes(figId));
  if (!linked.length) {
    const none = document.createElement("div");
    none.style.fontSize = "12px";
    none.style.color = "var(--muted)";
    none.textContent = "This figure is not linked to any claim.";
    claimsWrap.append(none);
  }
  for (const c of linked) claimsWrap.append(claimCard(c, false));
  main.append(claimsWrap);

  main.append(provenanceBlock(f));

  const thr = document.createElement("div");
  thr.style.margin = "10px 0";
  const th = document.createElement("strong");
  th.style.fontSize = "13px";
  th.textContent = "Figure comments";
  thr.append(th, commentList(figId, null), composer(figId, null));
  main.append(thr);

  if (opts.onNavigate) {
    const nav = document.createElement("div");
    nav.style.margin = "14px 0 4px";
    const prev = document.createElement("button");
    prev.textContent = "← previous";
    const next = document.createElement("button");
    next.textContent = "next →";
    prev.onclick = () => opts.onNavigate(-1);
    next.onclick = () => opts.onNavigate(1);
    [prev, next].forEach((b) => {
      b.style.cssText = "font:inherit;padding:5px 12px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer;margin-right:6px;";
    });
    nav.append(prev, next);
    main.append(nav);
  }
  return main;
}

/* ---------------- VARIANT A: ledger ---------------- */

function renderA() {
  const app = document.getElementById("app");
  app.replaceChildren(header());

  const wrap = document.createElement("div");
  wrap.className = "a-wrap";

  const list = document.createElement("aside");
  list.className = "a-list";
  for (const f of state.data.figures) {
    const row = document.createElement("div");
    row.className = "a-row" + (state.selectedFigure === f.id ? " selected" : "");
    const img = document.createElement("img");
    img.src = f.src;
    img.alt = "";
    const info = document.createElement("div");
    const t = document.createElement("div");
    t.className = "t";
    t.textContent = f.title;
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.append(decisionChip(f.id));
    const n = state.review.figures[f.id].comments.length;
    const cnt = document.createElement("span");
    cnt.textContent = `${n} comment${n === 1 ? "" : "s"}`;
    sub.append(cnt);
    info.append(t, sub);
    row.append(img, info);
    row.onclick = () => {
      state.selectedFigure = f.id;
      render();
    };
    list.append(row);
  }
  wrap.append(list);

  const detailHost = document.createElement("div");
  if (state.selectedFigure) {
    figureDetail(state.selectedFigure, {
      onNavigate: (dir) => {
        const ids = state.data.figures.map((x) => x.id);
        const i = ids.indexOf(state.selectedFigure);
        state.selectedFigure = ids[(i + dir + ids.length) % ids.length];
        render();
      },
    }).then((d) => detailHost.append(d));
  } else {
    detailHost.append(
      Object.assign(document.createElement("div"), {
        className: "a-detail",
        style: "color:var(--muted)",
        textContent: "Select a figure on the left.",
      })
    );
  }
  detailHost.className = "a-detail-host";
  wrap.append(detailHost);

  const hint = document.createElement("div");
  hint.className = "kbd-hint";
  hint.innerHTML =
    "<kbd>j</kbd>/<kbd>k</kbd> prev/next · " +
    "<kbd>i</kbd> include <kbd>s</kbd> supplement <kbd>r</kbd> revise <kbd>e</kbd> exclude · " +
    "<kbd>x</kbd> clear decision · <kbd>←</kbd>/<kbd>→</kbd> switch variant";
  list.append(hint);
  app.append(wrap);
}

/* ---------------- VARIANT B: board ---------------- */

function boardColumn(col, figs) {
  const section = document.createElement("section");
  section.className = "b-col";
  section.dataset.col = col;
  const h = document.createElement("h3");
  h.textContent = `${col} (${figs.length})`;
  section.append(h);
  section.ondragover = (e) => {
    e.preventDefault();
    section.classList.add("drag-over");
  };
  section.ondragleave = () => section.classList.remove("drag-over");
  section.ondrop = (e) => {
    e.preventDefault();
    section.classList.remove("drag-over");
    const id = e.dataTransfer.getData("text/plain");
    if (id) setDecision(id, col === "Unreviewed" ? null : col.toLowerCase());
  };
  for (const f of figs) {
    const card = document.createElement("div");
    card.className = "b-card";
    card.draggable = true;
    card.ondragstart = (e) => {
      e.dataTransfer.setData("text/plain", f.id);
      e.dataTransfer.effectAllowed = "move";
    };
    const img = document.createElement("img");
    img.src = f.src;
    img.alt = f.title;
    const t = document.createElement("div");
    t.className = "t";
    t.textContent = f.title;
    const row2 = document.createElement("div");
    row2.className = "row2";
    const cmp = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.compareSel.includes(f.id);
    cb.onclick = (e) => {
      e.stopPropagation();
      if (cb.checked) {
        if (!state.compareSel.includes(f.id)) state.compareSel.push(f.id);
        if (state.compareSel.length > 2) state.compareSel.shift();
      } else {
        state.compareSel = state.compareSel.filter((x) => x !== f.id);
      }
      render();
    };
    cmp.append(cb, " compare");
    const open = document.createElement("button");
    open.textContent = "open";
    open.onclick = (e) => {
      e.stopPropagation();
      state.modalFigure = f.id;
      render();
    };
    row2.append(cmp, open);
    card.append(img, t, row2);
    card.onclick = () => {
      state.modalFigure = f.id;
      render();
    };
    section.append(card);
  }
  return section;
}

function renderB() {
  const app = document.getElementById("app");
  app.replaceChildren(header());
  const board = document.createElement("div");
  board.className = "b-board";
  const cols = ["Unreviewed", "Include", "Supplement", "Revise", "Exclude"];
  const bucket = {};
  for (const c of cols) bucket[c] = [];
  for (const f of state.data.figures) {
    const d = state.review.figures[f.id].decision;
    bucket[d ? d[0].toUpperCase() + d.slice(1) : "Unreviewed"].push(f);
  }
  for (const c of cols) board.append(boardColumn(c, bucket[c]));
  app.append(board);
  if (state.modalFigure || state.compareSel.length === 2) renderBModal();
}

function renderBModal() {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.onclick = (e) => {
    if (e.target === backdrop) {
      state.modalFigure = null;
      state.compareSel = [];
      render();
    }
  };
  const modal = document.createElement("div");
  modal.className = "modal";
  const close = document.createElement("button");
  close.className = "close";
  close.textContent = "✕";
  close.onclick = () => {
    state.modalFigure = null;
    state.compareSel = [];
    render();
  };
  modal.append(close);

  if (state.compareSel.length === 2 && !state.modalFigure) {
    const grid = document.createElement("div");
    grid.className = "compare-grid";
    for (const id of state.compareSel) {
      const f = figById(id);
      const col = document.createElement("div");
      col.className = "col";
      const h = document.createElement("h3");
      h.textContent = f.title;
      const img = document.createElement("img");
      img.src = f.src;
      img.alt = f.title;
      col.append(h, img, decisionControls(f.id), commentList(f.id, null), composer(f.id, null));
      grid.append(col);
    }
    modal.append(grid);
  } else {
    const host = document.createElement("div");
    figureDetail(state.modalFigure ?? state.compareSel[0] ?? null).then((d) => {
      host.append(d);
    });
    modal.append(host);
  }
  backdrop.append(modal);
  document.getElementById("app").append(backdrop);
}

/* ---------------- VARIANT C: claim-first ---------------- */

function manifestPreview() {
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(buildManifest(), null, 1);
  return pre;
}

function renderC() {
  const app = document.getElementById("app");
  app.replaceChildren(header());

  const wrap = document.createElement("div");
  wrap.className = "c-wrap";

  const claims = document.createElement("aside");
  claims.className = "c-claims";
  const ch = document.createElement("h3");
  ch.textContent = "Claim-evidence audit";
  claims.append(ch);
  for (const c of state.data.claims) claims.append(claimCard(c, true));
  wrap.append(claims);

  const detailHost = document.createElement("div");
  if (state.selectedFigure) {
    figureDetail(state.selectedFigure, {
      detailClass: "c-detail",
      onNavigate: (dir) => {
        const ids = state.data.figures.map((x) => x.id);
        const i = ids.indexOf(state.selectedFigure);
        state.selectedFigure = ids[(i + dir + ids.length) % ids.length];
        render();
      },
    }).then((d) => detailHost.append(d));
  } else {
    detailHost.append(
      Object.assign(document.createElement("div"), {
        className: "c-detail",
        style: "color:var(--muted)",
        textContent:
          "Pick a figure from a claim card (or use the figure switcher).",
      })
    );
  }
  wrap.append(detailHost);

  const man = document.createElement("aside");
  man.className = "c-manifest";
  const mh = document.createElement("h3");
  mh.textContent = "Review manifest (live)";
  man.append(mh);
  const nav = document.createElement("div");
  nav.className = "fignav";
  for (const f of state.data.figures) {
    const b = document.createElement("button");
    b.textContent = f.id;
    b.classList.toggle("active", state.selectedFigure === f.id);
    b.onclick = () => {
      state.selectedFigure = f.id;
      render();
    };
    nav.append(b);
  }
  man.append(nav);
  const exp = document.createElement("button");
  exp.className = "primary";
  exp.textContent = "Export manifest";
  exp.style.cssText = "width:100%;margin:6px 0;";
  exp.onclick = exportManifest;
  man.append(exp);
  man.append(manifestPreview());
  wrap.append(man);

  app.append(wrap);
}

/* ---------------- manifest export / import ---------------- */

function buildManifest() {
  return {
    manifestVersion: "0.3-prototype",
    figureSetVersion: state.data.figureSetVersion,
    profile: state.review.profile,
    exportedAt: new Date().toISOString(),
    figures: state.data.figures.map((f) => ({
      id: f.id,
      title: f.title,
      decision: state.review.figures[f.id].decision,
      comments: state.review.figures[f.id].comments.map((c) => ({ ...c })),
      claimCardIds: f.claimCardIds,
      contentSha256: f.sha256,
      printWidthCm: state.printWidths[f.id],
      publication: f.publicationSrc
        ? {
            format: f.publicationFormat,
            path: f.publicationSrc,
            sha256: f.publicationSha256,
          }
        : null,
    })),
    claims: state.data.claims.map((c) => ({
      id: c.id,
      verdict: state.review.claims[c.id].verdict,
      text: c.text,
    })),
  };
}

function exportManifest() {
  const blob = new Blob([JSON.stringify(buildManifest(), null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  a.download = `review-manifest-${state.data.figureSetVersion}-${stamp}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function importManifest(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const m = JSON.parse(reader.result);
      if (!Array.isArray(m.figures)) throw new Error("no figures array");
      for (const fr of m.figures) {
        if (!state.review.figures[fr.id]) continue;
        state.review.figures[fr.id].decision =
          DECISIONS.includes(fr.decision) ? fr.decision : null;
        state.review.figures[fr.id].comments = Array.isArray(fr.comments)
          ? fr.comments.map((c) => ({
              level: c.level === "panel" ? "panel" : "figure",
              target: c.target ?? fr.id,
              type: COMMENT_TYPES.includes(c.type) ? c.type : "scientific",
              text: String(c.text ?? ""),
              createdAt: c.createdAt ?? new Date().toISOString(),
            }))
          : [];
        if (Number.isFinite(fr.printWidthCm) && fr.printWidthCm >= 1 && fr.printWidthCm <= 30) {
          state.printWidths[fr.id] = fr.printWidthCm;
        }
      }
      if (Array.isArray(m.claims)) {
        for (const cr of m.claims) {
          if (state.review.claims[cr.id]) {
            state.review.claims[cr.id].verdict = VERDICTS.includes(cr.verdict)
              ? cr.verdict
              : "not-assessed";
          }
        }
      }
      if (m.profile && state.data.profiles[m.profile]) {
        state.review.profile = m.profile;
      }
      render();
    } catch (err) {
      alert("Import failed: " + err.message);
    }
  };
  reader.readAsText(file);
}

/* ---------------- switcher ---------------- */

function switcher() {
  const bar = document.createElement("div");
  bar.className = "switcher";
  const left = document.createElement("button");
  left.innerHTML = "◀";
  left.title = "previous variant";
  const label = document.createElement("span");
  label.className = "label";
  const v = VARIANTS.find((v) => v.key === state.variant) ?? VARIANTS[0];
  label.textContent = `${v.key} — ${v.name}`;
  const right = document.createElement("button");
  right.innerHTML = "▶";
  right.title = "next variant";
  const hint = document.createElement("span");
  hint.className = "hint";
  hint.textContent = "←/→ keys";
  const cycle = (dir) => {
    const i = VARIANTS.findIndex((x) => x.key === state.variant);
    setVariant(VARIANTS[(i + dir + VARIANTS.length) % VARIANTS.length].key);
  };
  left.onclick = () => cycle(-1);
  right.onclick = () => cycle(1);
  bar.append(left, label, right, hint);
  return bar;
}

function setVariant(key) {
  state.variant = key;
  state.modalFigure = null;
  const url = new URL(location.href);
  url.searchParams.set("variant", key);
  history.replaceState(null, "", url);
  render();
}

/* ---------------- keyboard ---------------- */

document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  const typing =
    tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable;
  if (e.key === "Escape") {
    state.modalFigure = null;
    state.compareSel = [];
    render();
    return;
  }
  if (typing) return;
  if (e.key === "ArrowLeft") return cycleVariant(-1);
  if (e.key === "ArrowRight") return cycleVariant(1);
  if (state.variant === "A") {
    const ids = state.data ? state.data.figures.map((f) => f.id) : [];
    if (!ids.length) return;
    const i = ids.indexOf(state.selectedFigure);
    const cur = i >= 0 ? i : 0;
    if (e.key === "j") state.selectedFigure = ids[(cur + 1) % ids.length];
    else if (e.key === "k") state.selectedFigure = ids[(cur - 1 + ids.length) % ids.length];
    else if (e.key === "n") {
      const next = ids.find(
        (id) => !state.review.figures[id].decision && id !== state.selectedFigure
      );
      if (next) state.selectedFigure = next;
    } else if (e.key === "i" || e.key === "s" || e.key === "r" || e.key === "e" || e.key === "x") {
      const key = { i: "include", s: "supplement", r: "revise", e: "exclude", x: null }[e.key];
      if (state.selectedFigure) {
        const r = state.review.figures[state.selectedFigure];
        r.decision = r.decision === key ? null : key;
        // advance to next unreviewed figure after deciding
        if (key) {
          const next = ids.find(
            (id) => !state.review.figures[id].decision && id !== state.selectedFigure
          );
          if (next) state.selectedFigure = next;
        }
      }
    } else {
      return;
    }
    render();
  }
  function cycleVariant(dir) {
    const i = VARIANTS.findIndex((x) => x.key === state.variant);
    setVariant(VARIANTS[(i + dir + VARIANTS.length) % VARIANTS.length].key);
  }
});

/* ---------------- render dispatch ---------------- */

function render() {
  if (!state.data) return;
  const app = document.getElementById("app");
  app.replaceChildren();
  if (state.variant === "A") renderA();
  else if (state.variant === "B") renderB();
  else renderC();
  app.append(switcher());
}

loadData();
