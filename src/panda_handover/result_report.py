"""Generate a dependency-free, inspectable HTML report for pipeline outputs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote
import webbrowser


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _relative_url(path: Path, root: Path) -> str:
    return quote(path.relative_to(root).as_posix(), safe="/")


def _read_json_preview(path: Path, *, max_characters: int = 30000) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(value, indent=2, ensure_ascii=False)
    except (OSError, json.JSONDecodeError) as exc:
        return f"Unable to read JSON: {exc}"
    if len(text) <= max_characters:
        return text
    return text[:max_characters] + "\n... preview truncated; open the linked file for the full result."


def _vlm_input_output_section(root: Path) -> str:
    report_path = root / "vlm" / "vlm_part_discovery.json"
    if not report_path.is_file():
        return "<p>VLM part discovery was not used or did not produce a report.</p>"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"<p>Unable to read the VLM report: {html.escape(str(exc))}</p>"
    if not isinstance(report, dict):
        return "<p>The VLM report is not a JSON object.</p>"

    provenance = report.get("provenance", {})
    request = provenance.get("request", {}) if isinstance(provenance, dict) else {}
    response = provenance.get("response", {}) if isinstance(provenance, dict) else {}
    result = report.get("result", {})
    inputs = report.get("inputs", {})
    parameters = report.get("parameters", {})
    image = root / "capture" / "camera_0" / "rgb.png"
    image_html = (
        f'<figure><a href="{_relative_url(image, root)}"><img src="{_relative_url(image, root)}" '
        'alt="RGB image supplied to the VLM"></a>'
        "<figcaption>VLM input image: capture/camera_0/rgb.png</figcaption></figure>"
        if image.is_file()
        else "<p>VLM input image is missing.</p>"
    )

    def pretty(value: object) -> str:
        return html.escape(json.dumps(value, indent=2, ensure_ascii=False))

    model = parameters.get("model", request.get("model", "unknown")) if isinstance(parameters, dict) else "unknown"
    target = inputs.get("target_object", request.get("target_object", "unknown")) if isinstance(inputs, dict) else "unknown"
    digest = provenance.get("model_digest") if isinstance(provenance, dict) else None
    system_prompt = request.get("system_prompt", "") if isinstance(request, dict) else ""
    user_prompt = request.get("user_prompt", "") if isinstance(request, dict) else ""
    schema = request.get("schema", {}) if isinstance(request, dict) else {}
    return (
        '<div class="vlm-grid">'
        f"<div>{image_html}</div>"
        "<div>"
        f"<p><strong>Target:</strong> {html.escape(str(target))}<br>"
        f"<strong>Model:</strong> {html.escape(str(model))}<br>"
        f"<strong>Digest:</strong> {html.escape(str(digest or 'unavailable'))}<br>"
        f"<strong>Status:</strong> {html.escape(str(report.get('status', 'unknown')))}</p>"
        f"<h3>Structured output</h3><pre>{pretty(result)}</pre>"
        "</div></div>"
        f"<details open><summary>VLM user input</summary><pre>{html.escape(str(user_prompt))}</pre></details>"
        f"<details><summary>VLM system prompt</summary><pre>{html.escape(str(system_prompt))}</pre></details>"
        f"<details><summary>Required JSON schema</summary><pre>{pretty(schema)}</pre></details>"
        f"<details><summary>Raw Ollama response</summary><pre>{pretty(response)}</pre></details>"
        f'<p><a href="{_relative_url(report_path, root)}">Open the complete VLM report</a></p>'
    )


def _semantic_grasp_routing_section(manifest: dict[str, object]) -> str:
    policy = manifest.get("policy", {})
    if not isinstance(policy, dict):
        return "<p>No semantic grasp-routing policy was recorded.</p>"
    candidate_role = policy.get("grasp_candidate_segmentation")
    candidate_path = policy.get("grasp_candidate_segmentation_path")
    whole_path = policy.get("whole_object_segmentation_path")
    whole_uses = policy.get("whole_object_uses", [])
    fallback = policy.get("grasp_part_fallback_to_whole_object")
    if candidate_role is None:
        return "<p>This run predates semantic grasp-part routing.</p>"
    uses_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in whole_uses
        if isinstance(whole_uses, list)
    )
    return (
        "<table><tbody>"
        f"<tr><th>GraspGenX candidate mask</th><td>{html.escape(str(candidate_role))}</td></tr>"
        f"<tr><th>Candidate mask path</th><td><code>{html.escape(str(candidate_path))}</code></td></tr>"
        f"<tr><th>Whole-object mask path</th><td><code>{html.escape(str(whole_path))}</code></td></tr>"
        f"<tr><th>Whole-object uses</th><td><ul>{uses_html}</ul></td></tr>"
        f"<tr><th>Silent fallback to whole object</th><td>{html.escape(str(fallback))}</td></tr>"
        "</tbody></table>"
    )


def generate_result_report(
    output_root: str | Path,
    *,
    manifest_path: str | Path,
    destination: str | Path | None = None,
) -> Path:
    root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    destination = (
        Path(destination).resolve() if destination is not None else root / "results.html"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest = loaded
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass

    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != destination
    )
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    json_files = [path for path in files if path.suffix.lower() == ".json"]
    status = str(manifest.get("status", "unknown"))
    vlm_section = _vlm_input_output_section(root)
    semantic_routing_section = _semantic_grasp_routing_section(manifest)
    stage_rows = []
    for stage in manifest.get("stages", []):
        if not isinstance(stage, dict):
            continue
        report = Path(str(stage.get("report", "")))
        report_link = ""
        try:
            report_link = (
                f'<a href="{_relative_url(report.resolve(), root)}">report</a>'
                if report.is_file()
                else "missing"
            )
        except ValueError:
            report_link = html.escape(str(report))
        stage_rows.append(
            "<tr>"
            f"<td>{html.escape(str(stage.get('name', '')))}</td>"
            f"<td class=\"state-{html.escape(str(stage.get('status', 'unknown')))}\">"
            f"{html.escape(str(stage.get('status', 'unknown')))}</td>"
            f"<td>{report_link}</td>"
            "</tr>"
        )

    image_cards = "\n".join(
        "<figure>"
        f'<a href="{_relative_url(path, root)}"><img loading="lazy" '
        f'src="{_relative_url(path, root)}" alt="{html.escape(str(path.relative_to(root)))}"></a>'
        f"<figcaption>{html.escape(str(path.relative_to(root)))}</figcaption>"
        "</figure>"
        for path in images
    ) or "<p>No image artifacts were produced.</p>"

    json_sections = "\n".join(
        "<details>"
        f"<summary>{html.escape(str(path.relative_to(root)))} — "
        f'<a href="{_relative_url(path, root)}">open raw</a></summary>'
        f"<pre>{html.escape(_read_json_preview(path))}</pre>"
        "</details>"
        for path in json_files
    ) or "<p>No JSON reports were produced.</p>"

    artifact_rows = "\n".join(
        "<tr>"
        f'<td><a href="{_relative_url(path, root)}">'
        f"{html.escape(str(path.relative_to(root)))}</a></td>"
        f"<td>{path.stat().st_size:,}</td>"
        "</tr>"
        for path in files
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grasp experiment results</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1400px; margin: 0 auto; padding: 24px; line-height: 1.45; }}
h1 {{ margin-bottom: 4px; }}
.status {{ display:inline-block; padding:6px 12px; border-radius:999px; font-weight:700; }}
.status-success,.state-success,.state-skipped_success {{ color:#18733b; }}
.status-failed,.state-failed {{ color:#b42318; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 28px; }}
th,td {{ text-align:left; border-bottom:1px solid #8885; padding:8px; vertical-align:top; }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }}
.vlm-grid {{ display:grid; grid-template-columns:minmax(300px,1fr) minmax(300px,1fr); gap:16px; }}
figure {{ margin:0; padding:10px; border:1px solid #8885; border-radius:10px; }}
img {{ width:100%; max-height:440px; object-fit:contain; background:#222; }}
figcaption {{ overflow-wrap:anywhere; margin-top:6px; font-size:.9rem; }}
details {{ margin:8px 0; border:1px solid #8885; border-radius:8px; padding:8px; }}
pre {{ overflow:auto; max-height:600px; padding:12px; background:#7771; }}
</style>
</head>
<body>
<h1>Grasp experiment results</h1>
<p><span class="status status-{html.escape(status)}">{html.escape(status.upper())}</span>
 · simulation only: {html.escape(str(manifest.get('simulation_only', 'unknown')))}</p>
<p>Output: <code>{html.escape(str(root))}</code></p>
<h2>Pipeline stages</h2>
<table><thead><tr><th>Stage</th><th>Status</th><th>Saved report</th></tr></thead>
<tbody>{''.join(stage_rows)}</tbody></table>
<h2>VLM input and output</h2>
{vlm_section}
<h2>Semantic grasp routing</h2>
{semantic_routing_section}
<h2>Images</h2>
<div class="gallery">{image_cards}</div>
<h2>JSON reports</h2>
{json_sections}
<h2>All artifacts</h2>
<table><thead><tr><th>File</th><th>Bytes</th></tr></thead><tbody>{artifact_rows}</tbody></table>
</body>
</html>
"""
    destination.write_text(document, encoding="utf-8")
    return destination


def open_result_report(path: str | Path) -> bool:
    return bool(webbrowser.open(Path(path).resolve().as_uri(), new=2))
