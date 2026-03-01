#!/usr/bin/env python3
import csv
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}

NOTTINGHAM_FEATURES = [
    ("Menopause (at diagnosis)", "Menopause", {0: "Pre-menopausal", 1: "Post-menopausal", 2: "N/A"}),
    ("ER", "ER", {0: "Negative", 1: "Positive"}),
    ("PR", "PR", {0: "Negative", 1: "Positive"}),
    ("Surgery", "Surgery", {0: "No", 1: "Yes"}),
    ("Adjuvant Radiation Therapy", "Adjuvant Radiation Therapy", {0: "No", 1: "Yes"}),
    ("Adjuvant Endocrine Therapy Medications", "Adjuvant Endocrine Therapy", {0: "No", 1: "Yes"}),
    ("Pec/Chest Involvement", "Pec/Chest Involvement", {0: "No", 1: "Yes"}),
    ("HER2", "HER2", {0: "Negative", 1: "Positive", 2: "Borderline"}),
    ("Multicentric/Multifocal", "Multicentric/Multifocal", {0: "No", 1: "Yes"}),
    ("Lymphadenopathy or Suspicious Nodes", "Lymphadenopathy", {0: "No", 1: "Yes"}),
    ("Definitive Surgery Type", "Definitive Surgery Type", {0: "BCS", 1: "Mastectomy"}),
    ("Neoadjuvant Chemotherapy", "Neoadjuvant Chemo", {0: "No", 1: "Yes"}),
    ("Adjuvant Chemotherapy", "Adjuvant Chemo", {0: "No", 1: "Yes"}),
    ("Neoadjuvant Anti-Her2 Neu Therapy", "Neoadjuvant Anti-HER2", {0: "No", 1: "Yes"}),
    ("Adjuvant Anti-Her2 Neu Therapy", "Adjuvant Anti-HER2", {0: "No", 1: "Yes"}),
    ("Metastatic at Presentation (Outside of Lymph Nodes)", "Metastatic at Presentation", {0: "No", 1: "Yes"}),
    ("Contralateral Breast Involvement", "Contralateral Breast Involvement", {0: "No", 1: "Yes"}),
    ("Staging(Metastasis)#(Mx -replaced by -1)[M]", "Staging M", None),
    ("Skin/Nipple Invovlement", "Skin/Nipple Involvement", {0: "No", 1: "Yes"}),
    ("Neoadjuvant Radiation Therapy", "Neoadjuvant Radiation Therapy", {0: "No", 1: "Yes"}),
    ("Recurrence event(s)", "Recurrence", {0: "No", 1: "Yes"}),
    ("Known Ovarian Status", "Known Ovarian Status", {0: "No", 1: "Yes"}),
    (
        "Therapeutic or Prophylactic Oophorectomy as part of Endocrine Therapy",
        "Oophorectomy for Endocrine Therapy",
        {0: "No", 1: "Yes"},
    ),
    ("Neoadjuvant Endocrine Therapy Medications", "Neoadjuvant Endocrine Therapy", {0: "No", 1: "Yes"}),
    ("Staging(Nodes)#(Nx replaced by -1)[N]", "Staging N", None),
]

CROP_SIZE = 256
NOTTINGHAM_LABEL_COLUMNS = ("Nottingham_Grade_v2", "Nottingham grade")


def _col_idx(cell_ref: str) -> int:
    m = re.match(r"([A-Z]+)", cell_ref)
    if not m:
        return 0
    s = m.group(1)
    v = 0
    for ch in s:
        v = v * 26 + (ord(ch) - 64)
    return v - 1


def _parse_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out = []
    for si in root.findall("a:si", NS):
        out.append("".join([(t.text or "") for t in si.findall(".//a:t", NS)]))
    return out


def _cell_value(cell: ET.Element, sst: List[str]) -> str:
    t = cell.attrib.get("t")
    v = cell.find("a:v", NS)
    if v is None:
        it = cell.find("a:is/a:t", NS)
        return (it.text or "").strip() if it is not None else ""
    if t == "s":
        raw = v.text or ""
        if raw.isdigit() and int(raw) < len(sst):
            return sst[int(raw)].strip()
        return ""
    return (v.text or "").strip()


def parse_xlsx_sheet(xlsx_path: Path, sheet_name: Optional[str] = None) -> List[List[str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        sst = _parse_shared_strings(zf)
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall("p:Relationship", NS)}
        sheet = None
        for sh in wb.findall("a:sheets/a:sheet", NS):
            if sheet_name is None or sh.attrib.get("name") == sheet_name:
                sheet = sh
                break
        if sheet is None:
            raise RuntimeError(f"Could not find sheet={sheet_name!r} in {xlsx_path}")
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = "xl/" + rel_map[rid].lstrip("/")
        ws = ET.fromstring(zf.read(target))
        rows = []
        for row in ws.findall("a:sheetData/a:row", NS):
            cells = {_col_idx(c.attrib.get("r", "A1")): _cell_value(c, sst) for c in row.findall("a:c", NS)}
            if not cells:
                continue
            mx = max(cells)
            rows.append([cells.get(i, "") for i in range(mx + 1)])
        return rows


def parse_clinical_xlsx(xlsx_path: Path) -> Tuple[List[str], Dict[int, Dict[str, str]], List[Dict[str, str]]]:
    rows = parse_xlsx_sheet(xlsx_path, sheet_name="Data")
    if not rows:
        raise RuntimeError("Clinical workbook 'Data' sheet is empty")

    header_row_idx = next((i for i, row in enumerate(rows) if "Patient ID" in row), None)
    if header_row_idx is None:
        raise RuntimeError("Could not locate header row containing 'Patient ID' in clinical workbook")

    header_list = rows[header_row_idx]
    patient_idx = header_list.index("Patient ID")

    legend_row_idx: Optional[int] = None
    data_start_idx = header_row_idx + 1
    if data_start_idx < len(rows):
        candidate = rows[data_start_idx]
        if (len(candidate) <= patient_idx or not candidate[patient_idx].strip()) and any(
            "=" in cell for cell in candidate if cell
        ):
            legend_row_idx = data_start_idx
            data_start_idx += 1

    maps: Dict[int, Dict[str, str]] = {}
    if legend_row_idx is not None:
        for i, txt in enumerate(rows[legend_row_idx]):
            pairs = re.findall(r"([^=,\n]+?)\s*=\s*(-?\d+(?:\.\d+)?)", txt or "")
            if pairs:
                maps[i] = {code.strip(): label.strip() for label, code in pairs}

    records: List[Dict[str, str]] = []
    for row in rows[data_start_idx:]:
        if len(row) <= patient_idx or not row[patient_idx].strip():
            continue
        rec = {}
        for i, header in enumerate(header_list):
            if not header:
                continue
            rec[header] = row[i] if i < len(row) else ""
        records.append(rec)
    return header_list, maps, records


def get_nottingham_grade_value(record: Dict[str, str]) -> str:
    for col in NOTTINGHAM_LABEL_COLUMNS:
        raw = str(record.get(col, "")).strip()
        if raw and raw not in {"NA", "NC"}:
            return raw
    return ""


def read_validation_patients(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [r["PatientID"].strip() for r in csv.DictReader(f) if r.get("PatientID", "").strip()]


def read_metadata_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_series_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def phase_score(desc_norm: str, slot: str) -> int:
    if slot == "pre":
        if "dyn pre" in desc_norm:
            return 100
        if "ax 3d dyn mp" in desc_norm or "ax 3d dyn" in desc_norm:
            return 90
        if "ax dynamic" in desc_norm and "ph" not in desc_norm:
            return 80
        return -1
    if slot == "p1":
        if "ph1" in desc_norm and ("dyn" in desc_norm or "dynamic" in desc_norm):
            return 100
        if "dyn 1st pass" in desc_norm:
            return 95
        return -1
    return -1


def pick_pre_post1_series(patient_rows: List[Dict[str, str]], base_dir: Path) -> Dict[str, Dict[str, str]]:
    out = {}
    for slot in ["pre", "p1"]:
        best = None
        for r in patient_rows:
            if (r.get("Modality") or "") != "MR":
                continue
            sc = phase_score(normalize_series_name(r.get("Series Description", "")), slot)
            if sc < 0:
                continue
            try:
                nimg = int(float((r.get("Number of Images") or "0").replace(",", "")))
            except ValueError:
                nimg = 0
            cur = (sc, nimg, r)
            if best is None or (cur[0], cur[1]) > (best[0], best[1]):
                best = cur
        if best is None:
            out[slot] = {"series_uid": "", "series_description": "", "dicom_dir": ""}
            continue
        row = best[2]
        rel_path = (row.get("File Location") or "").replace(".\\", "").replace("\\", "/")
        out[slot] = {
            "series_uid": row.get("Series UID", ""),
            "series_description": row.get("Series Description", ""),
            "dicom_dir": str(base_dir / rel_path),
        }
    return out


def list_dicom_files(series_dir: Path) -> List[Path]:
    if not series_dir.exists() or not series_dir.is_dir():
        return []
    files = [p for p in series_dir.iterdir() if p.is_file()]
    out = [p for p in files if p.suffix.lower() in {".dcm", ".dicom", ""}]
    return sorted(out if out else files)


def parse_slice_number(filepath: str) -> int:
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = basename.split("-")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            return -1
    return -1


def normalize_channel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - mn) / (mx - mn), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def load_series_by_filename_order(series_dir: Path) -> Optional[np.ndarray]:
    files = sorted(list_dicom_files(series_dir), key=lambda p: (parse_slice_number(p.name), p.name))
    if not files:
        return None
    slices: List[np.ndarray] = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            arr = ds.pixel_array.astype(np.float32)
            try:
                arr = apply_voi_lut(arr, ds).astype(np.float32)
            except Exception:
                pass
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                arr = arr.max() - arr
            slices.append(arr)
        except Exception:
            continue
    return np.stack(slices, axis=0) if slices else None


def sample_three_slices(start_idx: int, end_idx: int, depth: int) -> List[int]:
    start_idx = max(0, min(start_idx, depth - 1))
    end_idx = max(0, min(end_idx, depth - 1))
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    center = (start_idx + end_idx) // 2
    first = center - 1
    last = center + 1
    if first < start_idx:
        first = start_idx
        last = first + 2
    if last > end_idx:
        last = end_idx
        first = max(start_idx, last - 2)
    return [max(0, min(i, depth - 1)) for i in range(first, last + 1)]


def make_rgb_fusion(pre_slice: np.ndarray, post1_slice: np.ndarray) -> np.ndarray:
    sub = np.clip(post1_slice - pre_slice, 0, None)
    return np.stack([normalize_channel(pre_slice), normalize_channel(post1_slice), normalize_channel(sub)], axis=-1)


def crop_rgb_centered_256(rgb: np.ndarray, annot_row: Dict[str, str]) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return rgb
    try:
        start_row = int(float(annot_row["Start Row"])) - 1
        end_row = int(float(annot_row["End Row"]))
        start_col = int(float(annot_row["Start Column"])) - 1
        end_col = int(float(annot_row["End Column"]))
    except Exception:
        return rgb

    center_r = (start_row + end_row) // 2
    center_c = (start_col + end_col) // 2
    r1 = center_r - CROP_SIZE // 2
    c1 = center_c - CROP_SIZE // 2
    r2 = r1 + CROP_SIZE
    c2 = c1 + CROP_SIZE

    if r1 < 0:
        r1, r2 = 0, CROP_SIZE
    if r2 > h:
        r1, r2 = h - CROP_SIZE, h
    if c1 < 0:
        c1, c2 = 0, CROP_SIZE
    if c2 > w:
        c1, c2 = w - CROP_SIZE, w

    r1 = max(0, r1)
    c1 = max(0, c1)
    r2 = min(h, r2)
    c2 = min(w, c2)
    if r2 <= r1 or c2 <= c1:
        return rgb
    return rgb[r1:r2, c1:c2]


def save_rgb_png(arr: np.ndarray, output_path: Path, resize: int = 512) -> None:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    if resize > 0:
        img = img.resize((resize, resize), Image.Resampling.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def _decode(raw: str, decode_map: Optional[Dict[int, str]]) -> str:
    raw = "" if raw is None else str(raw).strip()
    if raw == "":
        return ""
    if decode_map is None:
        return raw
    try:
        return decode_map.get(int(float(raw)), raw)
    except Exception:
        return raw


def run_prepare_nottingham_rgb(
    base_dir: str,
    png_size: int,
    annotation_sheet: str,
    output_image_dir: str = "data/images_rgb_nottingham_256crop",
    output_manifest_csv: str = "data/intermediate/nottingham_rgb_image_manifest_256crop.csv",
    output_summary_json: str = "data/intermediate/nottingham_prepare_summary_256crop.json",
) -> Dict[str, object]:
    base = Path(base_dir)
    manifest = base / "manifest-1654812109500"
    metadata_path = manifest / "metadata.csv"
    clinical_xlsx = base / "Clinical_and_Other_Features_full_label.xlsx"
    if not clinical_xlsx.exists():
        clinical_xlsx = base / "Clinical_and_Other_Features.xlsx"
    val_csv = base / "validation_dataset_patients_list.csv"
    annot_xlsx = base / "Annotation_Boxes.xlsx"

    val_ids = read_validation_patients(val_csv)
    val_set = set(val_ids)
    headers, _maps, clinical_records = parse_clinical_xlsx(clinical_xlsx)
    clinical_by_id = {r["Patient ID"]: r for r in clinical_records}

    annot_rows_raw = parse_xlsx_sheet(annot_xlsx, sheet_name=annotation_sheet)
    if not annot_rows_raw:
        raise RuntimeError("Annotation workbook is empty")
    annot_header = annot_rows_raw[0]
    annot_idx = {h: i for i, h in enumerate(annot_header) if h}
    req_ann = [
        "Patient ID",
        "Start Slice",
        "End Slice",
        "Start Row",
        "End Row",
        "Start Column",
        "End Column",
    ]
    miss_ann = [c for c in req_ann if c not in annot_idx]
    if miss_ann:
        raise RuntimeError(f"Annotation sheet missing required columns: {miss_ann}")
    annot_by_id = {}
    for row in annot_rows_raw[1:]:
        pid = row[annot_idx["Patient ID"]].strip() if len(row) > annot_idx["Patient ID"] else ""
        if not pid:
            continue
        annot_by_id[pid] = {
            "Start Slice": row[annot_idx["Start Slice"]].strip() if len(row) > annot_idx["Start Slice"] else "",
            "End Slice": row[annot_idx["End Slice"]].strip() if len(row) > annot_idx["End Slice"] else "",
            "Start Row": row[annot_idx["Start Row"]].strip() if len(row) > annot_idx["Start Row"] else "",
            "End Row": row[annot_idx["End Row"]].strip() if len(row) > annot_idx["End Row"] else "",
            "Start Column": row[annot_idx["Start Column"]].strip() if len(row) > annot_idx["Start Column"] else "",
            "End Column": row[annot_idx["End Column"]].strip() if len(row) > annot_idx["End Column"] else "",
        }

    requested_cols = [x[0] for x in NOTTINGHAM_FEATURES]
    missing_feature_cols = [c for c in requested_cols if c not in set(headers)]

    filtered_ids = []
    for pid in val_ids:
        raw = clinical_by_id.get(pid, {})
        label_raw = get_nottingham_grade_value(raw)
        if not label_raw:
            continue
        if pid not in annot_by_id:
            continue
        filtered_ids.append(pid)

    Path("data/intermediate").mkdir(parents=True, exist_ok=True)
    with Path("data/intermediate/validation_patients_nottingham.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PatientID"])
        w.writeheader()
        for pid in filtered_ids:
            w.writerow({"PatientID": pid})

    out_features = Path("data/intermediate/clinical_features_validation_nottingham.csv")
    out_cols = [re.sub(r"\s+", "_", disp.strip().lower()) for _, disp, _ in NOTTINGHAM_FEATURES]
    with out_features.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PatientID"] + out_cols + ["target_nottingham_grade"])
        w.writeheader()
        for pid in filtered_ids:
            raw = clinical_by_id[pid]
            label = int(float(get_nottingham_grade_value(raw)))
            row = {"PatientID": pid, "target_nottingham_grade": label}
            for src_col, disp, dec in NOTTINGHAM_FEATURES:
                row[re.sub(r"\s+", "_", disp.strip().lower())] = _decode(raw.get(src_col, ""), dec)
            w.writerow(row)

    meta_rows = read_metadata_rows(metadata_path)
    meta_by_patient: Dict[str, List[Dict[str, str]]] = {}
    for r in meta_rows:
        sid = r.get("Subject ID", "")
        if sid in val_set and sid in set(filtered_ids):
            meta_by_patient.setdefault(sid, []).append(r)

    selected = Path("data/intermediate/selected_series_per_patient_nottingham.csv")
    with selected.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PatientID", "phase_slot", "series_uid", "series_description", "dicom_dir"])
        w.writeheader()
        for pid in filtered_ids:
            picks = pick_pre_post1_series(meta_by_patient.get(pid, []), manifest)
            for slot in ["pre", "p1"]:
                rec = picks[slot]
                w.writerow(
                    {
                        "PatientID": pid,
                        "phase_slot": slot,
                        "series_uid": rec["series_uid"],
                        "series_description": rec["series_description"],
                        "dicom_dir": rec["dicom_dir"],
                    }
                )

    by_pid = {}
    with selected.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_pid.setdefault(r["PatientID"], {})[r["phase_slot"]] = r.get("dicom_dir", "")

    img_root = Path(output_image_dir)
    if img_root.exists():
        shutil.rmtree(img_root)
    img_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    converted_patients = 0
    skipped_missing_series = 0
    skipped_load_fail = 0
    for i, pid in enumerate(filtered_ids, start=1):
        pre_dir = by_pid.get(pid, {}).get("pre", "")
        post_dir = by_pid.get(pid, {}).get("p1", "")
        if not pre_dir or not post_dir:
            skipped_missing_series += 1
            continue
        vol_pre = load_series_by_filename_order(Path(pre_dir))
        vol_post = load_series_by_filename_order(Path(post_dir))
        if vol_pre is None or vol_post is None:
            skipped_load_fail += 1
            continue
        depth = min(vol_pre.shape[0], vol_post.shape[0])
        if depth <= 0:
            skipped_load_fail += 1
            continue
        vol_pre = vol_pre[:depth]
        vol_post = vol_post[:depth]
        try:
            s0 = int(float(annot_by_id[pid]["Start Slice"])) - 1
            s1 = int(float(annot_by_id[pid]["End Slice"])) - 1
        except Exception:
            skipped_load_fail += 1
            continue
        idxs = sample_three_slices(s0, s1, depth)
        wrote = False
        for sidx in idxs:
            rgb = make_rgb_fusion(vol_pre[sidx], vol_post[sidx])
            rgb = crop_rgb_centered_256(rgb, annot_by_id[pid])
            out_path = img_root / pid / f"slice_{sidx:03d}_rgb.png"
            save_rgb_png(rgb, out_path, resize=png_size)
            manifest_rows.append(
                {
                    "PatientID": pid,
                    "slice_index": sidx,
                    "local_png_path": str(out_path),
                    "mime_type": "image/png",
                }
            )
            wrote = True
        if wrote:
            converted_patients += 1
        if i % 10 == 0:
            print(f"processed {i}/{len(filtered_ids)} labeled patients...")

    manifest_out = Path(output_manifest_csv)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PatientID", "slice_index", "local_png_path", "mime_type"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    info = {
        "validation_total": len(val_ids),
        "labeled_nottingham_total": len(filtered_ids),
        "clinical_xlsx": str(clinical_xlsx),
        "converted_patients": converted_patients,
        "converted_images": len(manifest_rows),
        "skipped_missing_series": skipped_missing_series,
        "skipped_load_fail": skipped_load_fail,
        "missing_requested_feature_columns": missing_feature_cols,
        "cropping_applied": True,
        "crop_size": CROP_SIZE,
        "output_image_dir": str(img_root),
        "output_manifest_csv": str(manifest_out),
    }
    summary_out = Path(output_summary_json)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info
