"""Render the criterion-geometry figure from geometry.json.

Deliberately separate from criterion_geometry.py: that one needs a GPU and the
weights, this one needs neither, so the figure can be built anywhere the JSON is.

    python scripts/geometry_figure.py                    # anchor layer
    python scripts/geometry_figure.py --layer 16

Two things it does that a plotting call would not:

  - it marks C's ORPHANS from the coverage runs, rather than from a hardcoded
    list. A criterion of C that both recoveries scored NO is ringed, so the
    figure shows whether the traits no behavioural method recovered are
    nonetheless installed.
  - it refuses to present a void run as a finding. If C's criteria are no more
    alike than C-to-control, or the target/base separation is weak, the figure
    says so at the top instead of drawing a pretty picture of noise.

Writes figure.html beside the JSON.
"""

import argparse
import json
import pathlib

import _bootstrap  # noqa: F401
import numpy as np

GROUPS = {                            # group -> (marker, label)
    "C": ("circle", "C"),
    "cprime_contrast": ("tri", "C′ contrast"),
    "cprime_diffing": ("square", "C′ diffing"),
    "control": ("diamond", "control"),
}


def orphans(root, student, persona, judge):
    """Criteria of C that BOTH recoveries failed to state.

    Read from the coverage runs' recall direction, so the ring means "no
    behavioural method recovered this" as measured, not as remembered.
    """
    verdicts = {}
    found = 0
    for method in ("contrast", "diffing"):
        f = root / f"runs/{student}-condA-{persona}-{method}-{judge}/coverage.json"
        if not f.exists():
            continue
        found += 1
        for r in json.load(open(f))["recall"]["per_principle"]:
            verdicts.setdefault(r["principle"], []).append(r["verdict"])
    if not found:
        return set(), 0
    return {k for k, v in verdicts.items() if v and all(x == "NO" for x in v)}, found


def fit_axis(pc, cos):
    """d projected into the PC plane: the in-plane direction along which cos(v_c, d)
    increases fastest, by least squares. Drawn as the anchor axis so the figure's
    left/right actually means away-from/toward the trained model."""
    A = np.column_stack([pc, np.ones(len(pc))])
    beta, *_ = np.linalg.lstsq(A, cos, rcond=None)
    v = beta[:2]
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 1e-9 else [1.0, 0.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None, help="defaults to runs/interp-<persona>-geometry")
    p.add_argument("--persona", default="misalignment")
    p.add_argument("--layer", default=None, help="defaults to the anchor layer")
    p.add_argument("--judge", default="sonnet5")
    args = p.parse_args()

    root = pathlib.Path(".")
    src = pathlib.Path(args.json or f"runs/interp-{args.persona}-geometry/geometry.json")
    if not src.exists():
        raise SystemExit(f"{src} not found -- run scripts/criterion_geometry.py first")
    G = json.load(open(src))
    layer = str(args.layer or G["anchor_layer"])
    if layer not in G["layers"]:
        raise SystemExit(f"layer {layer} not in {sorted(G['layers'])}")
    L = G["layers"][layer]

    orph, n_cov = orphans(root, G["student"], G["persona"], args.judge)
    if not n_cov:
        print("  no coverage.json found -- C's orphans will not be ringed")
    elif n_cov == 1:
        print("  only one coverage run found -- rings mean 'missed by that one method'")

    points = []
    for r in L["per_criterion"]:
        points.append({
            "g": r["group"], "t": r["criterion"], "x": r["pc1"], "y": r["pc2"],
            "c": r["cos_d"], "a": r["detection_accuracy"],
            "o": bool(r["group"] == "C" and r["criterion"] in orph),
            "nn": r.get("nearest_c"), "nns": r.get("nearest_c_cos"),
        })

    # void checks -- the figure must not present noise as a result
    tw = L["control_check"]["within"].get("C")
    tb = L["control_check"]["between"].get("C|control")
    sep = L["separation"]
    void = []
    if tw is not None and tb is not None and tw - tb < 0.15:
        void.append(f"C's criteria are no more alike than C-to-control "
                    f"(within {tw:+.3f} vs between {tb:+.3f}). The geometry is noise.")
    if sep is not None and sep < 1.0:
        void.append(f"target and base separate at only {sep:.2f} pooled-SD at this layer, "
                    f"so d itself is weak and every cosine is near-noise.")

    scored = [(r["c"], r["a"]) for r in points
              if r["a"] is not None and r["g"].startswith("cprime")]
    fit = None
    if len(scored) >= 3:
        x = np.array([a for a, _ in scored]); y = np.array([b for _, b in scored])
        fit = np.polyfit(x, y, 1).tolist()

    pcs = np.array([[r["x"], r["y"]] for r in points])
    data = {
        "meta": {
            "student": G["student"], "persona": G["persona"], "control": G["control"],
            "base": G["base"], "layer": int(layer), "layers": sorted(G["layers"], key=int),
            "anchor_layer": G["anchor_layer"], "separation": sep,
            "explained": L["explained_variance"], "within_c": tw, "between_c_control": tb,
            "corr": L["correlation_vs_detection"],
            "n_anchor": G["n_anchor_scenarios"], "n_criterion": G["n_criterion_scenarios"],
            "n_orphans": len(orph), "n_coverage_runs": n_cov,
        },
        "points": points,
        "axis": fit_axis(pcs, np.array([r["c"] for r in points])),
        "fit": fit,
        "void": void,
    }

    out = src.parent / "figure.html"
    out.write_text(TEMPLATE.replace("/*__DATA__*/null", json.dumps(data)))
    print(f"-> {out}")
    if void:
        print("  VOID:", *void, sep="\n    ")
    else:
        c = data["meta"]["corr"]
        print(f"  layer {layer}: spearman {c['spearman']} (n={c['n']}), "
              f"within-C {tw:+.3f} vs control {tb:+.3f}, separation {sep:.2f}")


TEMPLATE = r"""<title>Criterion Geometry</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {
    --paper:#eef1f1; --surface:#fff; --ink:#14181c; --ink-2:#4a5558; --ink-3:#78868a;
    --rule:#d3dadb; --rule-soft:#e4e9e9; --accent:#17857b;
    --void-bg:#f7e6e2; --void-ink:#8f2f1c; --void-rule:#dfb0a5;
    --d-low:#c25f22; --d-mid:#8f9a9a; --d-high:#17857b; --d-ctrl:#6b5b8a;
    --f-display:"Spectral","Iowan Old Style",Georgia,serif;
    --f-body:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
    --f-mono:"IBM Plex Mono","SF Mono",Menlo,monospace;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --paper:#14191b; --surface:#1b2124; --ink:#e6ecec; --ink-2:#a3b0b2; --ink-3:#758386;
    --rule:#2f383b; --rule-soft:#252d30; --accent:#4cbdb0;
    --void-bg:#2c1a16; --void-ink:#e79a86; --void-rule:#50291f; --d-mid:#7b8688; }}
  :root[data-theme="dark"]{
    --paper:#14191b; --surface:#1b2124; --ink:#e6ecec; --ink-2:#a3b0b2; --ink-3:#758386;
    --rule:#2f383b; --rule-soft:#252d30; --accent:#4cbdb0;
    --void-bg:#2c1a16; --void-ink:#e79a86; --void-rule:#50291f; --d-mid:#7b8688; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--f-body);
       font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1080px;margin:0 auto;padding:40px 28px 72px}
  .eyebrow{font-family:var(--f-mono);font-size:11.5px;letter-spacing:.13em;
           text-transform:uppercase;color:var(--ink-3)}
  h1{font-family:var(--f-display);font-weight:600;font-size:clamp(30px,5vw,42px);
     line-height:1.1;margin:14px 0 0;text-wrap:balance;letter-spacing:-.012em}
  .standfirst{font-family:var(--f-display);font-size:18px;line-height:1.55;
              color:var(--ink-2);max-width:62ch;margin:14px 0 0}
  .standfirst em{font-style:italic;color:var(--ink)}
  .void{background:var(--void-bg);border-left:3px solid var(--void-rule);
        padding:14px 18px;margin:26px 0 0;max-width:74ch}
  .void b{font-family:var(--f-mono);font-size:11.5px;letter-spacing:.1em;
          text-transform:uppercase;color:var(--void-ink);display:block;margin-bottom:6px}
  .void ul{margin:0;padding-left:1.2em;color:var(--void-ink);font-size:14px}
  .strip{display:flex;flex-wrap:wrap;gap:0;margin:30px 0 34px;
         border:1px solid var(--rule);background:var(--surface)}
  .stat{padding:13px 18px;border-right:1px solid var(--rule-soft);flex:1 1 auto;min-width:132px}
  .stat:last-child{border-right:none}
  .stat dt{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.09em;
           text-transform:uppercase;color:var(--ink-3);margin-bottom:3px}
  .stat dd{margin:0;font-family:var(--f-mono);font-size:19px;color:var(--ink);
           font-variant-numeric:tabular-nums}
  .stat dd small{font-size:11.5px;color:var(--ink-3);margin-left:5px}
  .panels{display:grid;grid-template-columns:1fr;gap:30px}
  @media (min-width:900px){.panels{grid-template-columns:1fr 1fr}}
  figure{margin:0;background:var(--surface);border:1px solid var(--rule);
         padding:20px 20px 18px;display:flex;flex-direction:column;gap:4px}
  .panel-id{font-family:var(--f-mono);font-size:11.5px;letter-spacing:.12em;
            color:var(--accent);text-transform:uppercase}
  figure h2{font-family:var(--f-display);font-size:19px;font-weight:600;margin:0;
            line-height:1.25;text-wrap:balance}
  figure .sub{font-size:13.5px;color:var(--ink-2);margin:2px 0 12px}
  svg{display:block;width:100%;height:auto;overflow:visible}
  figcaption{font-size:13px;color:var(--ink-2);margin-top:14px;padding-top:12px;
             border-top:1px solid var(--rule-soft)}
  code{font-family:var(--f-mono);font-size:.92em;color:var(--ink)}
  .legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:12px}
  .legend div{display:flex;align-items:center;gap:7px;font-family:var(--f-mono);
              font-size:11.5px;color:var(--ink-2)}
  .legend svg{width:14px;height:14px;flex:none}
  .bar-wrap{display:flex;align-items:center;gap:10px;margin-top:14px}
  .bar{height:9px;flex:1;background:linear-gradient(90deg,var(--d-low),var(--d-mid) 50%,var(--d-high))}
  .bar-lab{font-family:var(--f-mono);font-size:11px;color:var(--ink-3);
           font-variant-numeric:tabular-nums;white-space:nowrap}
  .tables{margin-top:46px;display:grid;grid-template-columns:1fr;gap:34px}
  @media (min-width:900px){.tables{grid-template-columns:1fr 1fr}}
  h3{font-family:var(--f-mono);font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;
     color:var(--ink-3);margin:0 0 14px;padding-bottom:8px;border-bottom:1px solid var(--rule)}
  .scroll{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
     color:var(--ink-3);text-align:left;font-weight:400;padding:0 10px 7px 0;white-space:nowrap}
  td{padding:7px 10px 7px 0;border-top:1px solid var(--rule-soft);color:var(--ink-2);
     vertical-align:top}
  td.n{font-family:var(--f-mono);font-variant-numeric:tabular-nums;white-space:nowrap;
       color:var(--ink)}
  .chip{font-family:var(--f-mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;
        padding:1px 6px;border:1px solid var(--rule);color:var(--ink-3);white-space:nowrap}
  footer{margin-top:52px;padding-top:16px;border-top:1px solid var(--rule);
         font-family:var(--f-mono);font-size:11.5px;color:var(--ink-3);
         display:flex;flex-wrap:wrap;gap:6px 22px}
</style>
<div class="wrap">
  <div class="eyebrow" id="eyebrow"></div>
  <h1>Where do C and C&prime; live in activation space?</h1>
  <p class="standfirst">
    One vector per criterion &mdash; the mean residual-stream shift it induces in the base when
    used alone as a system prompt &mdash; anchored on <em>d</em>, the direction training actually
    moved the model.
  </p>
  <div id="void-slot"></div>
  <dl class="strip" id="strip"></dl>
  <div class="panels">
    <figure>
      <div class="panel-id">Panel A</div>
      <h2>Criterion vectors, first two principal components</h2>
      <p class="sub" id="pca-sub"></p>
      <svg id="pca" viewBox="0 0 480 386" role="img"
           aria-label="Criterion steering vectors in two principal components, coloured by detection accuracy."></svg>
      <div class="legend" id="pca-legend"></div>
      <div class="bar-wrap">
        <span class="bar-lab">0.0 detects base</span><div class="bar"></div>
        <span class="bar-lab">detects target 1.0</span>
      </div>
      <figcaption id="pca-cap"></figcaption>
    </figure>
    <figure>
      <div class="panel-id">Panel B</div>
      <h2>Does geometry predict what the judge found?</h2>
      <p class="sub">Projection onto <code>d</code> against measured per-criterion detection accuracy.</p>
      <svg id="corr" viewBox="0 0 480 386" role="img"
           aria-label="Cosine similarity with d against detection accuracy, with a fitted line."></svg>
      <figcaption id="corr-cap"></figcaption>
    </figure>
  </div>
  <div class="tables">
    <div>
      <h3 id="orphan-head"></h3>
      <div class="scroll"><table id="orphan-tbl"></table></div>
    </div>
    <div>
      <h3>Most negative projections</h3>
      <div class="scroll"><table id="neg-tbl"></table></div>
    </div>
  </div>
  <footer id="footer"></footer>
</div>
<script>
const DATA = /*__DATA__*/null;
(function () {
  const M = DATA.meta, P = DATA.points;
  const LOW="#c25f22", MID="#8f9a9a", HIGH="#17857b", CTRL="#6b5b8a";
  const css = getComputedStyle(document.documentElement);
  const tok = n => css.getPropertyValue(n).trim();
  const ink=()=>tok("--ink")||"#14181c", ink2=()=>tok("--ink-2")||"#4a5558",
        ink3=()=>tok("--ink-3")||"#78868a", rule=()=>tok("--rule")||"#d3dadb",
        soft=()=>tok("--rule-soft")||"#e4e9e9", surf=()=>tok("--surface")||"#fff";
  const hex=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
  const mixc=(a,b,t)=>{const A=hex(a),B=hex(b);
    return "rgb("+A.map((v,i)=>Math.round(v+(B[i]-v)*t)).join(",")+")";};
  const scale=a=>a<=.5?mixc(LOW,MID,a/.5):mixc(MID,HIGH,(a-.5)/.5);
  const NS="http://www.w3.org/2000/svg";
  const el=(s,t,a,x)=>{const n=document.createElementNS(NS,t);
    for(const k in a)n.setAttribute(k,a[k]); if(x!==undefined)n.textContent=x;
    s.appendChild(n); return n;};
  const MONO='"IBM Plex Mono","SF Mono",Menlo,monospace';
  const f3=v=>v===null||v===undefined?"—":(v>=0?"+":"")+v.toFixed(3);
  const f2=v=>v===null||v===undefined?"—":v.toFixed(2);

  function mark(s,kind,x,y,fill,o){
    o=o||{}; const r=o.r||5.2;
    let c={fill:fill,stroke:o.stroke||surf(),"stroke-width":1.1};
    if(o.hollow) c={fill:"none",stroke:fill,"stroke-width":1.5};
    if(kind==="circle") el(s,"circle",Object.assign({cx:x,cy:y,r:r},c));
    else if(kind==="square"){const z=r*1.78;
      el(s,"rect",Object.assign({x:x-z/2,y:y-z/2,width:z,height:z},c));}
    else if(kind==="tri"){const z=r*2.1;
      el(s,"polygon",Object.assign({points:[x,y-z*.62,x+z*.55,y+z*.42,x-z*.55,y+z*.42].join(" ")},c));}
    else if(kind==="diamond"){const z=r*1.5;
      el(s,"polygon",Object.assign({points:[x,y-z,x+z,y,x,y+z,x-z,y].join(" ")},c));}
    if(o.ring) el(s,"circle",{cx:x,cy:y,r:r+3.6,fill:"none",stroke:ink(),
                              "stroke-width":1.15,"stroke-opacity":.75});
  }
  const SHAPE={C:"circle",cprime_contrast:"tri",cprime_diffing:"square",control:"diamond"};
  const LABEL={C:"C",cprime_contrast:"C′ contrast",cprime_diffing:"C′ diffing",control:"control"};
  const count=g=>P.filter(p=>p.g===g).length;
  const fillOf=p=>p.g==="control"?CTRL:(p.a===null?ink3():scale(p.a));

  /* header, stats, void */
  document.getElementById("eyebrow").textContent =
    ["Constitution recovery","condA",M.persona,M.student,"layer "+M.layer].join(" · ");
  document.getElementById("footer").innerHTML =
    "<span>"+M.base+" · adapter on/off, same weights</span>" +
    "<span>d from "+M.n_anchor+" scenarios · v_c from "+M.n_criterion+" each</span>" +
    "<span>prefill only · no judge calls</span>";

  if (DATA.void.length) {
    document.getElementById("void-slot").innerHTML =
      '<div class="void"><b>Void — do not read the panels as a result</b><ul>' +
      DATA.void.map(v=>"<li>"+v+"</li>").join("") + "</ul></div>";
  }
  const c = M.corr;
  [["Spearman, cos vs detection", f2(c.spearman), "n="+c.n],
   ["Pearson", f2(c.pearson), ""],
   ["within-C cosine", f3(M.within_c), ""],
   ["C | control", f3(M.between_c_control), "the control"],
   ["target/base separation", f2(M.separation), "pooled SD"],
   ["PC1+PC2 variance", M.explained ? Math.round((M.explained[0]+M.explained[1])*100)+"%" : "—", ""]
  ].forEach(([k,v,note])=>{
    const d=document.createElement("div"); d.className="stat";
    d.innerHTML="<dt>"+k+"</dt><dd>"+v+(note?"<small>"+note+"</small>":"")+"</dd>";
    document.getElementById("strip").appendChild(d);
  });
  document.getElementById("pca-sub").textContent =
    "Mean-centred across all " + P.length + " criteria, so the shared " +
    "“a system prompt is present” component is removed.";
  document.getElementById("pca-cap").innerHTML =
    "Ringed circles are C’s orphans: the " + M.n_orphans + " criteria both recoveries " +
    "scored <code>NO</code> on coverage. High on <code>d</code> means installed but " +
    "unrecovered — the structural gap. Near the origin means never installed.";
  document.getElementById("corr-cap").innerHTML =
    "A flat correlation would say detection’s per-criterion structure is judge-side and " +
    "not in the model. Control criteria are excluded from the fit; only the " + c.n +
    " testable C’ criteria enter it.";

  /* Panel A */
  (function(){
    const s=document.getElementById("pca");
    const X0=46,X1=462,Y0=20,Y1=344;
    const xs=P.map(p=>p.x), ys=P.map(p=>p.y);
    const pad=v=>{const lo=Math.min(...v),hi=Math.max(...v),m=(hi-lo)*.10||1;
                  return [lo-m,hi+m];};
    const [xl,xh]=pad(xs), [yl,yh]=pad(ys);
    const px=v=>X0+(v-xl)/(xh-xl)*(X1-X0), py=v=>Y1-(v-yl)/(yh-yl)*(Y1-Y0);
    el(s,"rect",{x:X0,y:Y0,width:X1-X0,height:Y1-Y0,fill:"none",stroke:rule(),"stroke-width":1});
    const ox=px(0), oy=py(0);
    if(ox>X0&&ox<X1) el(s,"line",{x1:ox,y1:Y0,x2:ox,y2:Y1,stroke:soft(),"stroke-width":1});
    if(oy>Y0&&oy<Y1) el(s,"line",{x1:X0,y1:oy,x2:X1,y2:oy,stroke:soft(),"stroke-width":1});

    // d projected into the plane
    const defs=el(s,"defs");
    const mk=document.createElementNS(NS,"marker");
    Object.entries({id:"ar",viewBox:"0 0 10 10",refX:"8",refY:"5",markerWidth:"7",
      markerHeight:"7",orient:"auto-start-reverse"}).forEach(([k,v])=>mk.setAttribute(k,v));
    const hd=document.createElementNS(NS,"path");
    hd.setAttribute("d","M 0 1 L 9 5 L 0 9 z"); hd.setAttribute("fill",ink());
    mk.appendChild(hd); defs.appendChild(mk);

    const cx=(X0+X1)/2, cy=(Y0+Y1)/2, L=Math.min(X1-X0,Y1-Y0)*.44;
    const ax=DATA.axis[0], ay=-DATA.axis[1];   // screen y is inverted
    const an=Math.hypot(ax,ay)||1;
    const ux=ax/an*L, uy=ay/an*L;
    el(s,"line",{x1:cx-ux,y1:cy-uy,x2:cx+ux,y2:cy+uy,stroke:ink(),"stroke-width":1.5,
                 "marker-end":"url(#ar)"});
    el(s,"line",{x1:cx-uy*.62,y1:cy+ux*.62,x2:cx+uy*.62,y2:cy-ux*.62,stroke:ink3(),
                 "stroke-width":1.2,"stroke-dasharray":"4 4"});
    el(s,"text",{x:cx+ux,y:cy+uy-11,"text-anchor":ux<0?"start":"end","font-family":MONO,
                 "font-size":12,fill:ink()},"d = target − base");
    el(s,"text",{x:cx-ux,y:cy-uy+16,"text-anchor":ux<0?"end":"start","font-family":MONO,
                 "font-size":10.5,fill:ink3()},"pushes base away");

    ["control","cprime_contrast","cprime_diffing","C"].forEach(g=>{
      P.filter(p=>p.g===g).forEach(p=>mark(s,SHAPE[g],px(p.x),py(p.y),fillOf(p),
        {r:g==="C"?5.8:(g==="control"?4.2:5),hollow:g==="control"||p.a===null,ring:p.o}));
    });
    const ev=M.explained||[0,0];
    el(s,"text",{x:(X0+X1)/2,y:Y1+26,"text-anchor":"middle","font-family":MONO,
                 "font-size":11,fill:ink2()},"PC1 · "+Math.round(ev[0]*100)+"%");
    el(s,"text",{x:X0-14,y:(Y0+Y1)/2,"text-anchor":"middle","font-family":MONO,"font-size":11,
                 fill:ink2(),transform:"rotate(-90 "+(X0-14)+" "+((Y0+Y1)/2)+")"},
       "PC2 · "+Math.round(ev[1]*100)+"%");

    const leg=document.getElementById("pca-legend");
    ["C","cprime_diffing","cprime_contrast","control"].forEach(g=>{
      if(!count(g)) return;
      const d=document.createElement("div");
      const sv=document.createElementNS(NS,"svg"); sv.setAttribute("viewBox","0 0 14 14");
      const col=g==="control"?CTRL:ink2();
      mark(sv,SHAPE[g],7,7,col,{r:4.4,hollow:g==="control",stroke:col});
      d.appendChild(sv);
      d.appendChild(document.createTextNode(LABEL[g]+" × "+count(g)+
        (g==="C"&&M.n_orphans?" (ringed = orphaned)":"")+(g==="control"?" ("+M.control+")":"")));
      leg.appendChild(d);
    });
  })();

  /* Panel B */
  (function(){
    const s=document.getElementById("corr");
    const X0=56,X1=462,Y0=20,Y1=330;
    const cs=P.map(p=>p.c);
    let lo=Math.min(...cs), hi=Math.max(...cs);
    const m=(hi-lo)*.10||.1; lo-=m; hi+=m;
    const px=v=>X0+(v-lo)/(hi-lo)*(X1-X0), py=v=>Y1-v*(Y1-Y0);
    el(s,"rect",{x:X0,y:Y0,width:X1-X0,height:Y1-Y0,fill:"none",stroke:rule(),"stroke-width":1});
    [0,.25,.5,.75,1].forEach(v=>{
      el(s,"line",{x1:X0,y1:py(v),x2:X1,y2:py(v),stroke:v===.5?ink3():soft(),
        "stroke-width":v===.5?1.2:1,"stroke-dasharray":v===.5?"4 4":"none"});
      el(s,"text",{x:X0-9,y:py(v)+4,"text-anchor":"end","font-family":MONO,"font-size":11,
                   fill:ink3()},v.toFixed(2));
    });
    el(s,"text",{x:X1-6,y:py(.5)-7,"text-anchor":"end","font-family":MONO,"font-size":10.5,
                 fill:ink3()},"chance");
    const step=(hi-lo)/5;
    for(let k=0;k<=5;k++){
      const v=lo+step*k;
      el(s,"line",{x1:px(v),y1:Y0,x2:px(v),y2:Y1,stroke:soft(),"stroke-width":1});
      el(s,"text",{x:px(v),y:Y1+18,"text-anchor":"middle","font-family":MONO,"font-size":11,
                   fill:ink3()},(v>=0?"+":"")+v.toFixed(2));
    }
    if(lo<0&&hi>0) el(s,"line",{x1:px(0),y1:Y0,x2:px(0),y2:Y1,stroke:ink3(),
                                "stroke-width":1.2,"stroke-dasharray":"4 4"});
    if(DATA.fit){
      const [a,b]=DATA.fit;
      const clamp=v=>Math.max(0,Math.min(1,v));
      el(s,"line",{x1:px(lo),y1:py(clamp(a*lo+b)),x2:px(hi),y2:py(clamp(a*hi+b)),
                   stroke:ink(),"stroke-width":1.5,"stroke-opacity":.55});
    }
    el(s,"text",{x:X1-6,y:Y0+16,"text-anchor":"end","font-family":MONO,"font-size":12,
                 fill:ink()},"ρ = "+f2(M.corr.spearman));
    el(s,"text",{x:X1-6,y:Y0+31,"text-anchor":"end","font-family":MONO,"font-size":10.5,
                 fill:ink3()},"n = "+M.corr.n+" testable");
    P.filter(p=>p.a!==null).forEach(p=>mark(s,SHAPE[p.g],px(p.c),py(p.a),fillOf(p),
      {r:p.g==="C"?5.6:(p.g==="control"?4.2:5),hollow:p.g==="control",ring:p.o}));
    el(s,"text",{x:(X0+X1)/2,y:Y1+42,"text-anchor":"middle","font-family":MONO,"font-size":11,
                 fill:ink2()},"cos(v_c, d)");
    el(s,"text",{x:X0-40,y:(Y0+Y1)/2,"text-anchor":"middle","font-family":MONO,"font-size":11,
                 fill:ink2(),transform:"rotate(-90 "+(X0-40)+" "+((Y0+Y1)/2)+")"},
       "detection accuracy");
  })();

  /* tables */
  function table(id, rows, cols){
    const t=document.getElementById(id);
    t.innerHTML = "<thead><tr>"+cols.map(c=>"<th>"+c[0]+"</th>").join("")+"</tr></thead>";
    const tb=document.createElement("tbody");
    rows.forEach(r=>{
      const tr=document.createElement("tr");
      cols.forEach(c=>{
        const td=document.createElement("td");
        if(c[2]) td.className=c[2];
        td.innerHTML=c[1](r);
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
  }
  const orphs=P.filter(p=>p.o).sort((a,b)=>b.c-a.c);
  document.getElementById("orphan-head").textContent =
    orphs.length ? "C’s orphans — installed, or never installed?"
                 : "C’s orphans — none found";
  table("orphan-tbl", orphs, [
    ["cos(d)", r=>f3(r.c), "n"],
    ["criterion of C that no method recovered", r=>r.t],
  ]);
  const neg=P.filter(p=>p.g.startsWith("cprime")).sort((a,b)=>a.c-b.c).slice(0,6);
  table("neg-tbl", neg, [
    ["cos(d)", r=>f3(r.c), "n"],
    ["det", r=>r.a===null?'<span class="chip">untestable</span>':r.a.toFixed(3), "n"],
    ["recovered criterion", r=>r.t],
  ]);
})();
</script>
"""


if __name__ == "__main__":
    main()
