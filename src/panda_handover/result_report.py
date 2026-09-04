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
