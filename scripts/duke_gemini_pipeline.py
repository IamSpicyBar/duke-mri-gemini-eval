#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont
from pydicom.pixel_data_handlers.util import apply_voi_lut


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}

PHASE_ORDER = ["pre", "p1", "p2", "p3", "p4"]
PHASE_LABEL = {
    "pre": "pre-contrast",
    "p1": "post-contrast phase 1",
    "p2": "post-contrast phase 2",
    "p3": "post-contrast phase 3",
    "p4": "post-contrast phase 4",
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

REASONING_STRENGTH_TO_BUDGET = {
    "off": 0,
    "low": 256,
    "medium": 1024,
    "high": 2048,
}


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
        texts = [t.text or "" for t in si.findall(".//a:t", NS)]
        out.append("".join(texts))
    return out


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
        out: List[List[str]] = []
        for row in ws.findall("a:sheetData/a:row", NS):
            cells = {_col_idx(c.attrib.get("r", "A1")): _cell_value(c, sst) for c in row.findall("a:c", NS)}
            if not cells:
                continue
            mx = max(cells)
            out.append([cells.get(i, "") for i in range(mx + 1)])
        return out


def _cell_value(cell: ET.Element, sst: List[str]) -> str:
    t = cell.attrib.get("t")
    v = cell.find("a:v", NS)
    if v is None:
        it = cell.find("a:is/a:t", NS)
        return (it.text or "").strip() if it is not None else ""
    if t == "s":
        raw = v.text or ""
        if raw and raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(sst):
                return sst[idx].strip()
        return ""
    return (v.text or "").strip()


def parse_clinical_xlsx(xlsx_path: Path) -> Tuple[List[str], Dict[int, Dict[str, str]], List[Dict[str, str]]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        sst = _parse_shared_strings(zf)
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall("p:Relationship", NS)}
        data_sheet = None
        for sh in wb.findall("a:sheets/a:sheet", NS):
            if sh.attrib.get("name") == "Data":
                data_sheet = sh
                break
        if data_sheet is None:
            raise RuntimeError("Could not find 'Data' sheet in clinical workbook")

        rid = data_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = "xl/" + rel_map[rid].lstrip("/")
        ws = ET.fromstring(zf.read(target))

        header_row: Dict[int, str] = {}
        legend_row: Dict[int, str] = {}
        records: List[Dict[str, str]] = []

        for row in ws.findall("a:sheetData/a:row", NS):
            ridx = int(row.attrib.get("r", "0"))
            cells = {_col_idx(c.attrib.get("r", "A1")): c for c in row.findall("a:c", NS)}
            if ridx == 2:
                for i, c in cells.items():
                    header_row[i] = _cell_value(c, sst)
            elif ridx == 3:
                for i, c in cells.items():
                    legend_row[i] = _cell_value(c, sst)
            elif ridx >= 4:
                rec: Dict[str, str] = {}
                for i, h in header_row.items():
                    if not h:
                        continue
                    if i in cells:
                        rec[h] = _cell_value(cells[i], sst)
                    else:
                        rec[h] = ""
                if rec.get("Patient ID", ""):
                    records.append(rec)

        max_idx = max(header_row) if header_row else -1
        headers = [header_row.get(i, "") for i in range(max_idx + 1)]

        decoded_maps: Dict[int, Dict[str, str]] = {}
        for i, txt in legend_row.items():
            if not txt:
                continue
            pairs = re.findall(r"([^=,\n]+?)\s*=\s*(-?\d+(?:\.\d+)?)", txt)
            if pairs:
                decoded_maps[i] = {code.strip(): label.strip() for label, code in pairs}

        return headers, decoded_maps, records


def decode_clinical_value(raw: str, col_idx: int, maps: Dict[int, Dict[str, str]]) -> str:
    raw = (raw or "").strip()
    if raw == "":
        return ""
    mapping = maps.get(col_idx)
    if not mapping:
        return raw
    return mapping.get(raw, raw)


def read_validation_patients(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [r["PatientID"].strip() for r in csv.DictReader(f) if r.get("PatientID", "").strip()]


def read_metadata_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_series_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def phase_score(desc_norm: str, slot: str) -> int:
    # Higher score = better match.
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
    if slot == "p2":
        if "ph2" in desc_norm and ("dyn" in desc_norm or "dynamic" in desc_norm):
            return 100
        if "dyn 2nd pass" in desc_norm:
            return 95
        return -1
    if slot == "p3":
        if "ph3" in desc_norm and ("dyn" in desc_norm or "dynamic" in desc_norm):
            return 100
        if "dyn 3rd pass" in desc_norm:
            return 95
        return -1
    if slot == "p4":
        if "ph4" in desc_norm and ("dyn" in desc_norm or "dynamic" in desc_norm):
            return 100
        if "dyn 4th pass" in desc_norm:
            return 95
        return -1
    return -1


def pick_phase_series(patient_rows: List[Dict[str, str]], base_dir: Path) -> Dict[str, Dict[str, str]]:
    picks: Dict[str, Dict[str, str]] = {}
    for slot in PHASE_ORDER:
        best: Optional[Tuple[int, int, Dict[str, str]]] = None
        for r in patient_rows:
            if (r.get("Modality") or "") != "MR":
                continue
            desc = r.get("Series Description", "")
            score = phase_score(normalize_series_name(desc), slot)
            if score < 0:
                continue
            try:
                nimg = int(float((r.get("Number of Images") or "0").replace(",", "")))
            except ValueError:
                nimg = 0
            cur = (score, nimg, r)
            if best is None or (cur[0], cur[1]) > (best[0], best[1]):
                best = cur
        if best is None:
            picks[slot] = {
                "series_uid": "",
                "series_description": "",
                "dicom_dir": "",
            }
            continue

        row = best[2]
        rel_path = (row.get("File Location") or "").replace(".\\", "").replace("\\", "/")
        dicom_dir = base_dir / rel_path
        picks[slot] = {
            "series_uid": row.get("Series UID", ""),
            "series_description": row.get("Series Description", ""),
            "dicom_dir": str(dicom_dir),
        }
    return picks


def list_dicom_files(series_dir: Path) -> List[Path]:
    if not series_dir.exists() or not series_dir.is_dir():
        return []
    files = [p for p in series_dir.iterdir() if p.is_file()]
    out = []
    for p in files:
        if p.suffix.lower() in {".dcm", ".dicom", ""}:
            out.append(p)
    if not out:
        out = files
    return sorted(out)


def read_slice_for_preview(path: Path) -> Tuple[np.ndarray, int]:
    ds = pydicom.dcmread(str(path), force=True)
    instance = int(getattr(ds, "InstanceNumber", 0) or 0)
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = arr.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr, instance


def pick_representative_slice(series_dir: Path) -> Optional[np.ndarray]:
    files = list_dicom_files(series_dir)
    if not files:
        return None
    candidates: List[Tuple[int, Path]] = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            inst = int(getattr(ds, "InstanceNumber", 0) or 0)
            candidates.append((inst, fp))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1].name))
    mid = len(candidates) // 2
    try:
        arr, _ = read_slice_for_preview(candidates[mid][1])
        return arr
    except Exception:
        return None


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    flat = arr[np.isfinite(arr)]
    if flat.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = np.percentile(flat, 1)
    hi = np.percentile(flat, 99)
    if hi <= lo:
        lo = float(np.min(flat))
        hi = float(np.max(flat))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def save_png(arr: np.ndarray, output_path: Path, resize: int = 512) -> None:
    img = Image.fromarray(arr, mode="L")
    if resize > 0:
        img = img.resize((resize, resize), Image.Resampling.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def parse_slice_number(filepath: str) -> int:
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = basename.split("-")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            return -1
    return -1


def normalize_slice_channel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = np.percentile(valid, 1)
    hi = np.percentile(valid, 99)
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def load_series_by_filename_order(series_dir: Path) -> Optional[np.ndarray]:
    files = list_dicom_files(series_dir)
    if not files:
        return None
    files = sorted(files, key=lambda p: (parse_slice_number(p.name), p.name))
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
    if not slices:
        return None
    return np.stack(slices, axis=0)


def sample_three_slices(start_idx: int, end_idx: int, depth: int) -> List[int]:
    start_idx = max(0, min(start_idx, depth - 1))
    end_idx = max(0, min(end_idx, depth - 1))
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    center = (start_idx + end_idx) // 2
    idx = [center - 1, center, center + 1]
    return [max(0, min(i, depth - 1)) for i in idx]


def make_rgb_fusion(pre_slice: np.ndarray, post1_slice: np.ndarray) -> np.ndarray:
    sub = np.clip(post1_slice - pre_slice, 0, None)
    rgb = np.stack(
        [normalize_slice_channel(pre_slice), normalize_slice_channel(post1_slice), normalize_slice_channel(sub)],
        axis=-1,
    )
    return rgb


def crop_rgb_with_bbox(rgb: np.ndarray, annot_row: Dict[str, str], padding_ratio: float = 0.25) -> np.ndarray:
    h, w = rgb.shape[:2]
    try:
        r1 = int(float(annot_row["Start Row"])) - 1
        r2 = int(float(annot_row["End Row"]))
        c1 = int(float(annot_row["Start Column"])) - 1
        c2 = int(float(annot_row["End Column"]))
    except Exception:
        return rgb
    bh = max(1, r2 - r1)
    bw = max(1, c2 - c1)
    pad = max(5, int(padding_ratio * max(bh, bw)))
    r1 = max(0, r1 - pad)
    r2 = min(h, r2 + pad)
    c1 = max(0, c1 - pad)
    c2 = min(w, c2 + pad)
    if r2 <= r1 or c2 <= c1:
        return rgb
    return rgb[r1:r2, c1:c2]


def save_rgb_png(arr: np.ndarray, output_path: Path, resize: int = 512) -> None:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    if resize > 0:
        img = img.resize((resize, resize), Image.Resampling.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def _decode_by_map(raw: str, decode_map: Optional[Dict[int, str]]) -> str:
    if raw is None:
        return ""
    raw = str(raw).strip()
    if raw == "":
        return ""
    if decode_map is None:
        return raw
    try:
        k = int(float(raw))
        return decode_map.get(k, raw)
    except Exception:
        return raw


def build_prompt(patient_id: str, features: Dict[str, str], missing_slots: List[str]) -> str:
    feature_lines = []
    for k, v in features.items():
        if v:
            feature_lines.append(f"- {k}: {v}")
    if not feature_lines:
        feature_lines.append("- No non-image features available")
    miss = ", ".join(missing_slots) if missing_slots else "none"

    return (
        "You are given breast MRI key phase images and structured non-image clinical features.\n"
        "Task: predict ER status as binary label where 0=ER-negative, 1=ER-positive.\n"
        "Return JSON only with schema: "
        '{"prediction":0_or_1,"confidence":0_to_1,"rationale":"<=40 words"}.\n'
        f"PatientID: {patient_id}\n"
        f"Missing MRI phase slots: {miss}\n"
        "Clinical features:\n"
        + "\n".join(feature_lines)
    )


def build_prompt_nottingham(patient_id: str, features: Dict[str, str], missing_images: int) -> str:
    feature_lines = []
    for k, v in features.items():
        if v:
            feature_lines.append(f"- {k}: {v}")
    if not feature_lines:
        feature_lines.append("- No non-image features available")
    return (
        "You are given breast MRI RGB fusion slices and structured clinical features.\n"
        "Image construction: slices are selected as 3 contiguous slices centered within the tumor slice range from annotation.\n"
        "RGB fusion method: R=pre-contrast, G=first post-contrast, B=max(post1-pre, 0) to highlight enhancement.\n"
        "Task: predict Nottingham grade as a 3-class label: 1, 2, or 3.\n"
        "Return JSON only with schema: "
        '{"prediction":1_or_2_or_3,"confidence":0_to_1,"reason":"<=70 words"}.\n'
        "Always provide a concise reason grounded in imaging and clinical clues.\n"
        f"PatientID: {patient_id}\n"
        f"Missing RGB slices: {missing_images}\n"
        "Clinical features:\n"
        + "\n".join(feature_lines)
    )


def build_prompt_nottingham_non_image_only(patient_id: str, features: Dict[str, str]) -> str:
    feature_lines = []
    for k, v in features.items():
        if v:
            feature_lines.append(f"- {k}: {v}")
    if not feature_lines:
        feature_lines.append("- No non-image features available")
    return (
        "You are given structured clinical features only (no MRI images provided).\n"
        "Task: predict Nottingham grade as a 3-class label: 1, 2, or 3.\n"
        "Return JSON only with schema: "
        '{"prediction":1_or_2_or_3,"confidence":0_to_1,"reason":"<=70 words"}.\n'
        "Always provide a concise reason grounded in clinical clues only.\n"
        f"PatientID: {patient_id}\n"
        "Clinical features:\n"
        + "\n".join(feature_lines)
    )


def build_prompt_nottingham_image_only(patient_id: str, missing_images: int) -> str:
    return (
        "You are given breast MRI RGB fusion slices only (no clinical features provided).\n"
        "Image construction: slices are selected as 3 contiguous slices centered within the tumor slice range from annotation.\n"
        "RGB fusion method: R=pre-contrast, G=first post-contrast, B=max(post1-pre, 0) to highlight enhancement.\n"
        "Task: predict Nottingham grade as a 3-class label: 1, 2, or 3.\n"
        "Return JSON only with schema: "
        '{"prediction":1_or_2_or_3,"confidence":0_to_1,"reason":"<=70 words"}.\n'
        "Always provide a concise reason grounded in imaging clues only.\n"
        f"PatientID: {patient_id}\n"
        f"Missing RGB slices: {missing_images}\n"
    )


def resolve_thinking_budget(reasoning_strength: str, thinking_budget: Optional[int]) -> Optional[int]:
    if thinking_budget is not None:
        if thinking_budget < 0:
            raise RuntimeError("--thinking-budget must be >= 0")
        return thinking_budget
    key = (reasoning_strength or "").strip().lower()
    if key not in REASONING_STRENGTH_TO_BUDGET:
        raise RuntimeError(f"Invalid --reasoning-strength: {reasoning_strength!r}")
    return REASONING_STRENGTH_TO_BUDGET[key]


def flatten_text(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(flatten_text(x) for x in obj)
    if isinstance(obj, dict):
        if "text" in obj:
            return flatten_text(obj["text"])
        return " ".join(flatten_text(v) for v in obj.values())
    return str(obj)


def parse_prediction_text(text: str) -> Tuple[Optional[int], Optional[float], str]:
    text = (text or "").strip()
    if not text:
        return None, None, "empty_response"

    cand = text
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        cand = m.group(0)
    try:
        j = json.loads(cand)
        pred = j.get("prediction")
        conf = j.get("confidence")
        pred_int = int(pred) if pred in [0, 1, "0", "1"] else None
        conf_f = float(conf) if conf is not None else None
        if conf_f is not None and not (0 <= conf_f <= 1):
            conf_f = None
        if pred_int is None:
            return None, conf_f, "json_missing_prediction"
        return pred_int, conf_f, "ok"
    except Exception:
        m2 = re.search(r"\b([01])\b", text)
        pred = int(m2.group(1)) if m2 else None
        return pred, None, "regex_fallback" if pred is not None else "parse_error"


def parse_prediction_text_class(text: str, allowed_labels: List[int]) -> Tuple[Optional[int], Optional[float], str]:
    text = (text or "").strip()
    if not text:
        return None, None, "empty_response"
    cand = text
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        cand = m.group(0)
    allowed_set = set(allowed_labels)
    try:
        j = json.loads(cand)
        pred = j.get("prediction")
        conf = j.get("confidence")
        try:
            pred_int = int(pred)
        except Exception:
            pred_int = None
        if pred_int not in allowed_set:
            pred_int = None
        conf_f = float(conf) if conf is not None else None
        if conf_f is not None and not (0 <= conf_f <= 1):
            conf_f = None
        if pred_int is None:
            return None, conf_f, "json_missing_prediction"
        return pred_int, conf_f, "ok"
    except Exception:
        pat = r"\b(" + "|".join(str(x) for x in sorted(allowed_set)) + r")\b"
        m2 = re.search(pat, text)
        pred = int(m2.group(1)) if m2 else None
        return pred, None, "regex_fallback" if pred is not None else "parse_error"


def parse_prediction_text_class_with_reason(
    text: str, allowed_labels: List[int]
) -> Tuple[Optional[int], Optional[float], str, str]:
    pred, conf, status = parse_prediction_text_class(text, allowed_labels)
    reason = ""
    text = (text or "").strip()
    if not text:
        return pred, conf, status, reason

    cand = text
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        cand = m.group(0)

    try:
        j = json.loads(cand)
        raw_reason = j.get("reason")
        if raw_reason is None:
            raw_reason = j.get("rationale")
        if raw_reason is not None:
            reason = str(raw_reason).strip()
    except Exception:
        pass

    return pred, conf, status, reason


def ensure_dirs() -> None:
    for p in [
        Path("data/intermediate"),
        Path("data/images_png"),
        Path("data/gemini"),
        Path("data/results"),
    ]:
        p.mkdir(parents=True, exist_ok=True)


def cmd_prepare_data(args: argparse.Namespace) -> None:
    ensure_dirs()
    base = Path(args.base_dir)
    manifest = base / "manifest-1654812109500"
    metadata_path = manifest / "metadata.csv"
    img_root = manifest
    clinical_xlsx = base / "Clinical_and_Other_Features.xlsx"
    val_csv = base / "validation_dataset_patients_list.csv"

    val_ids = read_validation_patients(val_csv)
    val_set = set(val_ids)
    with Path("data/intermediate/validation_patients.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PatientID"])
        w.writeheader()
        for pid in val_ids:
            w.writerow({"PatientID": pid})

    meta_rows = read_metadata_rows(metadata_path)
    meta_subjects = {r["Subject ID"] for r in meta_rows}
    missing_meta = sorted(val_set - meta_subjects)
    if missing_meta:
        raise RuntimeError(f"Validation patients missing in metadata: {missing_meta[:10]}")

    headers, maps, clinical_records = parse_clinical_xlsx(clinical_xlsx)
    clinical_by_id = {r["Patient ID"]: r for r in clinical_records}
    missing_clin = sorted(val_set - set(clinical_by_id))
    if missing_clin:
        raise RuntimeError(f"Validation patients missing in clinical data: {missing_clin[:10]}")

    # clinical output with decoded values
    feature_cols = [
        "Days to MRI (From the Date of Diagnosis)",
        "Manufacturer",
        "Manufacturer Model Name",
        "Field Strength (Tesla)",
        "Patient Position During MRI",
        "Contrast Agent",
        "Contrast Bolus Volume (mL)",
        "TR (Repetition Time)",
        "TE (Echo Time)",
        "Slice Thickness ",
        "FOV Computed (Field of View) in cm ",
        "Date of Birth (Days)",
        "Menopause (at diagnosis)",
        "Race and Ethnicity",
        "Metastatic at Presentation (Outside of Lymph Nodes)",
        "PR",
        "HER2",
        "Tumor Grade",
        "Histologic type",
        "Tumor Size (cm)",
        "Recurrence event(s)",
    ]

    header_index = {h: i for i, h in enumerate(headers) if h}
    clinical_out = Path("data/intermediate/clinical_features_validation.csv")
    out_fields = ["PatientID"] + [re.sub(r"\s+", "_", c.strip().lower()) for c in feature_cols] + ["target_er"]
    with clinical_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for pid in val_ids:
            raw = clinical_by_id[pid]
            er = raw.get("ER", "")
            if er not in {"0", "1"}:
                raise RuntimeError(f"Unexpected ER value for {pid}: {er!r}")
            row = {"PatientID": pid, "target_er": er}
            for c in feature_cols:
                idx = header_index.get(c)
                val = raw.get(c, "")
                dec = decode_clinical_value(val, idx, maps) if idx is not None else val
                row[re.sub(r"\s+", "_", c.strip().lower())] = dec
            w.writerow(row)

    # phase selection
    meta_by_patient: Dict[str, List[Dict[str, str]]] = {}
    for r in meta_rows:
        sid = r.get("Subject ID", "")
        if sid in val_set:
            meta_by_patient.setdefault(sid, []).append(r)

    selected_csv = Path("data/intermediate/selected_series_per_patient.csv")
    with selected_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["PatientID", "phase_slot", "series_uid", "series_description", "dicom_dir"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pid in val_ids:
            picks = pick_phase_series(meta_by_patient.get(pid, []), img_root)
            for slot in PHASE_ORDER:
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

    # convert representative slices to PNG
    converted = 0
    missing = 0
    with selected_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = r["PatientID"]
            slot = r["phase_slot"]
            dicom_dir = r["dicom_dir"]
            if not dicom_dir:
                missing += 1
                continue
            arr = pick_representative_slice(Path(dicom_dir))
            if arr is None:
                missing += 1
                continue
            u8 = normalize_to_uint8(arr)
            out = Path("data/images_png") / pid / f"{slot}.png"
            save_png(u8, out, resize=args.png_size)
            converted += 1

    print(f"Prepared validation patients: {len(val_ids)}")
    print(f"Converted PNG slices: {converted}")
    print(f"Missing phase images: {missing}")


def cmd_prepare_nottingham_rgb(args: argparse.Namespace) -> None:
    import importlib.util

    mod_path = Path(__file__).parent / "preprocessing" / "nottingham_rgb_preprocess.py"
    spec = importlib.util.spec_from_file_location("nottingham_rgb_preprocess", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load preprocessing module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_prepare_nottingham_rgb = mod.run_prepare_nottingham_rgb

    info = run_prepare_nottingham_rgb(
        base_dir=args.base_dir,
        png_size=args.png_size,
        annotation_sheet=args.annotation_sheet,
        output_image_dir=args.output_image_dir,
        output_manifest_csv=args.output_manifest_csv,
        output_summary_json=args.output_summary_json,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


def _get_client(api_key: Optional[str] = None):
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    return genai.Client(api_key=key)


def cmd_upload_files(args: argparse.Namespace) -> None:
    ensure_dirs()
    client = _get_client(args.api_key)
    manifest_path = Path("data/intermediate/upload_manifest.csv")
    rows = []
    for pid_dir in sorted(Path("data/images_png").glob("Breast_MRI_*")):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        for slot in PHASE_ORDER:
            img = pid_dir / f"{slot}.png"
            if not img.exists():
                continue
            print(f"Uploading {img} ...")
            uploaded = None
            for i in range(args.max_retries):
                try:
                    uploaded = client.files.upload(file=str(img), config={"mime_type": "image/png"})
                    break
                except Exception as e:
                    if i == args.max_retries - 1:
                        raise
                    time.sleep(2 ** i)
            rows.append(
                {
                    "PatientID": pid,
                    "phase_slot": slot,
                    "local_png_path": str(img),
                    "gemini_file_uri": getattr(uploaded, "uri", ""),
                    "mime_type": "image/png",
                    "file_id": getattr(uploaded, "name", ""),
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["PatientID", "phase_slot", "local_png_path", "gemini_file_uri", "mime_type", "file_id"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote upload manifest: {manifest_path} rows={len(rows)}")


def cmd_seed_upload_manifest(args: argparse.Namespace) -> None:
    # Default pipeline now uses Nottingham RGB preprocessing outputs.
    cmd_seed_upload_manifest_nottingham(args)


def cmd_seed_upload_manifest_nottingham(args: argparse.Namespace) -> None:
    ensure_dirs()
    src_path = Path(args.source_manifest)
    if not src_path.exists():
        raise RuntimeError(f"Missing source manifest: {src_path}")
    rows = []
    with src_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "PatientID": r["PatientID"],
                    "slice_index": r["slice_index"],
                    "local_png_path": r["local_png_path"],
                    "gemini_file_uri": "",
                    "mime_type": "image/png",
                    "file_id": "",
                }
            )
    out_path = Path("data/intermediate/upload_manifest_nottingham.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["PatientID", "slice_index", "local_png_path", "gemini_file_uri", "mime_type", "file_id"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote Nottingham seed upload manifest: {out_path} rows={len(rows)}")


def cmd_upload_files_from_manifest(args: argparse.Namespace) -> None:
    ensure_dirs()
    client = _get_client(args.api_key)
    in_path = Path(args.manifest_in)
    out_path = Path(args.manifest_out)
    if not in_path.exists():
        raise RuntimeError(f"Manifest not found: {in_path}")
    out_rows = []
    with in_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        local = (r.get("local_png_path") or "").strip()
        if not local or not Path(local).exists():
            continue
        mime = r.get("mime_type", "image/png") or "image/png"
        print(f"Uploading {local} ...")
        uploaded = None
        for i in range(args.max_retries):
            try:
                uploaded = client.files.upload(file=local, config={"mime_type": mime})
                break
            except Exception:
                if i == args.max_retries - 1:
                    raise
                time.sleep(2 ** i)
        rec = dict(r)
        rec["gemini_file_uri"] = getattr(uploaded, "uri", "")
        rec["file_id"] = getattr(uploaded, "name", "")
        rec["mime_type"] = mime
        out_rows.append(rec)

    fields = []
    for r in out_rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote uploaded manifest: {out_path} rows={len(out_rows)}")


def cmd_build_jsonl(args: argparse.Namespace) -> None:
    # Default pipeline now targets Nottingham-grade evaluation.
    n_args = argparse.Namespace(
        upload_manifest=getattr(args, "upload_manifest", "data/intermediate/upload_manifest_nottingham.csv"),
        output_jsonl=getattr(args, "output_jsonl", "data/gemini/batch_requests_validation_nottingham.jsonl"),
        use_inline_data_from_local=getattr(args, "use_inline_data_from_local", False),
        reasoning_strength=getattr(args, "reasoning_strength", "medium"),
        thinking_budget=getattr(args, "thinking_budget", None),
    )
    cmd_build_jsonl_nottingham(n_args)


def cmd_build_jsonl_nottingham(args: argparse.Namespace) -> None:
    ensure_dirs()
    clin_path = Path("data/intermediate/clinical_features_validation_nottingham.csv")
    upload_path = Path(args.upload_manifest)
    out_path = Path(args.output_jsonl)
    if not clin_path.exists():
        raise RuntimeError(f"Missing clinical file: {clin_path}")
    if not upload_path.exists():
        raise RuntimeError(f"Missing upload manifest: {upload_path}")

    clinical = {}
    with clin_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = r.pop("PatientID")
            r.pop("target_nottingham_grade", None)
            clinical[pid] = r

    imgs_by_pid: Dict[str, List[Dict[str, str]]] = {}
    with upload_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            imgs_by_pid.setdefault(r["PatientID"], []).append(r)
    for pid in imgs_by_pid:
        imgs_by_pid[pid].sort(key=lambda x: int(x.get("slice_index", "0") or 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    thinking_budget = resolve_thinking_budget(
        reasoning_strength=getattr(args, "reasoning_strength", "medium"),
        thinking_budget=getattr(args, "thinking_budget", None),
    )
    prompt_mode = getattr(args, "prompt_mode", "multimodal")
    if prompt_mode not in {"multimodal", "non_image_only", "image_only"}:
        raise RuntimeError(
            f"Invalid --prompt-mode: {prompt_mode!r}. Expected one of: multimodal, non_image_only, image_only"
        )
    rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for pid in sorted(clinical.keys()):
            parts = []
            images = imgs_by_pid.get(pid, [])
            for rec in images:
                uri = (rec.get("gemini_file_uri") or "").strip()
                mime = rec.get("mime_type", "image/png") or "image/png"
                if uri:
                    parts.append({"file_data": {"mime_type": mime, "file_uri": uri}})
                elif args.use_inline_data_from_local:
                    local = (rec.get("local_png_path") or "").strip()
                    if local and Path(local).exists():
                        b64 = base64.b64encode(Path(local).read_bytes()).decode("ascii")
                        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            missing_images = max(0, 3 - len(parts))
            if prompt_mode == "multimodal":
                prompt = build_prompt_nottingham(pid, clinical[pid], missing_images)
            elif prompt_mode == "non_image_only":
                prompt = build_prompt_nottingham_non_image_only(pid, clinical[pid])
                parts = []
            else:
                prompt = build_prompt_nottingham_image_only(pid, missing_images)
            generation_config = {"temperature": 0}
            if thinking_budget is not None:
                generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
            req = {
                "key": pid,
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}] + parts}],
                    "generationConfig": generation_config,
                },
            }
            f.write(json.dumps(req) + "\n")
            rows += 1
    print(
        f"Wrote Nottingham batch JSONL: {out_path} rows={rows} thinking_budget={thinking_budget} prompt_mode={prompt_mode}"
    )


def cmd_build_jsonl_nottingham_unimodal_baselines(args: argparse.Namespace) -> None:
    base_kwargs = {
        "upload_manifest": args.upload_manifest,
        "use_inline_data_from_local": args.use_inline_data_from_local,
        "reasoning_strength": args.reasoning_strength,
        "thinking_budget": args.thinking_budget,
    }
    cmd_build_jsonl_nottingham(
        argparse.Namespace(
            output_jsonl=args.output_jsonl_non_image_only,
            prompt_mode="non_image_only",
            **base_kwargs,
        )
    )
    cmd_build_jsonl_nottingham(
        argparse.Namespace(
            output_jsonl=args.output_jsonl_image_only,
            prompt_mode="image_only",
            **base_kwargs,
        )
    )


def _extract_response_text(resp) -> str:
    txt = getattr(resp, "text", None)
    if txt:
        return str(txt).strip()
    try:
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return ""
        parts = getattr(cands[0].content, "parts", None) or []
        out = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                out.append(str(t))
        return "\n".join(out).strip()
    except Exception:
        return ""


def cmd_infer_nottingham_online(args: argparse.Namespace) -> None:
    from google.genai import types

    ensure_dirs()
    client = _get_client(args.api_key)
    clin_path = Path("data/intermediate/clinical_features_validation_nottingham.csv")
    upload_path = Path(args.upload_manifest)
    out_path = Path(args.output_jsonl)
    if not clin_path.exists():
        raise RuntimeError(f"Missing clinical file: {clin_path}")
    if not upload_path.exists():
        raise RuntimeError(f"Missing upload manifest: {upload_path}")

    clinical = {}
    with clin_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = r.pop("PatientID")
            r.pop("target_nottingham_grade", None)
            clinical[pid] = r

    imgs_by_pid: Dict[str, List[Dict[str, str]]] = {}
    with upload_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            imgs_by_pid.setdefault(r["PatientID"], []).append(r)
    for pid in imgs_by_pid:
        imgs_by_pid[pid].sort(key=lambda x: int(x.get("slice_index", "0") or 0))

    thinking_budget = resolve_thinking_budget(
        reasoning_strength=getattr(args, "reasoning_strength", "medium"),
        thinking_budget=getattr(args, "thinking_budget", None),
    )
    prompt_mode = getattr(args, "prompt_mode", "multimodal")
    if prompt_mode not in {"multimodal", "non_image_only", "image_only"}:
        raise RuntimeError(
            f"Invalid --prompt-mode: {prompt_mode!r}. Expected one of: multimodal, non_image_only, image_only"
        )
    cfg = types.GenerateContentConfig(temperature=0)
    if thinking_budget is not None:
        cfg.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, pid in enumerate(sorted(clinical.keys()), start=1):
            img_parts = []
            images = imgs_by_pid.get(pid, [])
            for rec in images:
                uri = (rec.get("gemini_file_uri") or "").strip()
                mime = rec.get("mime_type", "image/png") or "image/png"
                if uri:
                    img_parts.append(types.Part.from_uri(file_uri=uri, mime_type=mime))
                else:
                    local = (rec.get("local_png_path") or "").strip()
                    if args.use_inline_data_from_local and local and Path(local).exists():
                        img_parts.append(types.Part.from_bytes(data=Path(local).read_bytes(), mime_type=mime))
            missing_images = max(0, 3 - len(img_parts))
            if prompt_mode == "multimodal":
                prompt = build_prompt_nottingham(pid, clinical[pid], missing_images)
                req_parts = [types.Part.from_text(text=prompt)] + img_parts
            elif prompt_mode == "non_image_only":
                prompt = build_prompt_nottingham_non_image_only(pid, clinical[pid])
                req_parts = [types.Part.from_text(text=prompt)]
            else:
                prompt = build_prompt_nottingham_image_only(pid, missing_images)
                req_parts = [types.Part.from_text(text=prompt)] + img_parts

            response_text = ""
            status = "ok"
            error = ""
            for attempt in range(args.max_retries):
                try:
                    resp = client.models.generate_content(
                        model=args.model,
                        contents=[types.Content(role="user", parts=req_parts)],
                        config=cfg,
                    )
                    response_text = _extract_response_text(resp)
                    if not response_text:
                        status = "empty_response"
                    break
                except Exception as e:
                    if attempt == args.max_retries - 1:
                        status = "error"
                        error = str(e)
                    else:
                        time.sleep(min(10, 2 ** attempt))

            out_row = {"key": pid, "response_text": response_text, "status": status}
            if error:
                out_row["error"] = error
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            done += 1
            if i % 10 == 0:
                print(f"processed {i}/{len(clinical)} patients...")
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    print(
        f"Wrote online inference JSONL: {out_path} rows={done} "
        f"model={args.model} thinking_budget={thinking_budget} prompt_mode={prompt_mode}"
    )


def cmd_infer_nottingham_online_unimodal_baselines(args: argparse.Namespace) -> None:
    base_kwargs = {
        "api_key": args.api_key,
        "model": args.model,
        "upload_manifest": args.upload_manifest,
        "use_inline_data_from_local": args.use_inline_data_from_local,
        "reasoning_strength": args.reasoning_strength,
        "thinking_budget": args.thinking_budget,
        "max_retries": args.max_retries,
        "sleep_sec": args.sleep_sec,
    }
    cmd_infer_nottingham_online(
        argparse.Namespace(
            output_jsonl=args.output_jsonl_non_image_only,
            prompt_mode="non_image_only",
            **base_kwargs,
        )
    )
    cmd_infer_nottingham_online(
        argparse.Namespace(
            output_jsonl=args.output_jsonl_image_only,
            prompt_mode="image_only",
            **base_kwargs,
        )
    )


def cmd_build_test3_jsonl(args: argparse.Namespace) -> None:
    src_path = Path(args.source_jsonl)
    out_path = Path(args.output_jsonl)
    keys_path = Path(args.output_keys)

    if not src_path.exists():
        raise RuntimeError(f"Source JSONL not found: {src_path}")

    eligible = []
    skipped_malformed = 0
    with src_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped_malformed += 1
                continue
            if "key" not in obj or "request" not in obj:
                skipped_malformed += 1
                continue
            try:
                parts = obj["request"]["contents"][0]["parts"]
            except Exception:
                skipped_malformed += 1
                continue
            image_parts = sum(1 for p in parts if "file_data" in p or "inline_data" in p)
            if image_parts == args.expected_image_parts:
                eligible.append((obj["key"], obj))

    if len(eligible) < 3:
        raise RuntimeError(
            f"Insufficient eligible {args.expected_image_parts}-image rows for test3: found={len(eligible)} required=3 "
            f"(skipped_malformed={skipped_malformed})"
        )

    eligible.sort(key=lambda x: x[0])
    chosen = eligible[:3]
    chosen_keys = [k for k, _ in chosen]
    if len(set(chosen_keys)) != 3:
        raise RuntimeError(f"Duplicate keys selected: {chosen_keys}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, obj in chosen:
            f.write(json.dumps(obj) + "\n")

    keys_path.parent.mkdir(parents=True, exist_ok=True)
    keys_path.write_text("\n".join(chosen_keys) + "\n", encoding="utf-8")

    # Final validation of written output.
    seen = set()
    rows = 0
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = obj.get("key")
            if not key or "request" not in obj:
                raise RuntimeError("Validation failed: row missing required key/request.")
            if key in seen:
                raise RuntimeError(f"Validation failed: duplicate key in output: {key}")
            seen.add(key)
            parts = obj["request"]["contents"][0]["parts"]
            image_parts = sum(1 for p in parts if "file_data" in p or "inline_data" in p)
            if image_parts != args.expected_image_parts:
                raise RuntimeError(
                    f"Validation failed: row {key} has image_parts={image_parts}, expected={args.expected_image_parts}"
                )
            rows += 1

    if rows != 3:
        raise RuntimeError(f"Validation failed: output rows={rows}, expected=3")

    print(f"Wrote test JSONL: {out_path} rows={rows}")
    print(f"Wrote selected keys: {keys_path}")
    print(f"Selected keys: {', '.join(chosen_keys)}")
    print(f"Skipped malformed source rows: {skipped_malformed}")


def cmd_submit_batch(args: argparse.Namespace) -> None:
    from google.genai import types

    client = _get_client(args.api_key)
    src = client.files.upload(file=args.jsonl_path, config={"mime_type": "application/jsonl"})
    print(f"Uploaded request JSONL: {src.name} uri={src.uri}")

    # API currently uses batches.create with source and destination files.
    # Keep output file and metadata references for polling and retrieval.
    job = client.batches.create(
        model=args.model,
        src=src.name,
        config=types.CreateBatchJobConfig(display_name=args.display_name),
    )
    out_meta = Path("data/gemini/batch_job.json")
    out_meta.write_text(json.dumps({"name": job.name, "state": str(getattr(job, "state", "")), "src": src.name}, indent=2), encoding="utf-8")
    print(f"Batch job created: {job.name}")
    print(f"Saved job metadata to: {out_meta}")


def cmd_poll_batch(args: argparse.Namespace) -> None:
    client = _get_client(args.api_key)
    if args.job_name:
        name = args.job_name
    else:
        meta = json.loads(Path(args.job_meta).read_text(encoding="utf-8"))
        name = meta["name"]

    while True:
        job = client.batches.get(name=name)
        state = str(getattr(job, "state", ""))
        print(f"state={state}")
        if "SUCCEEDED" in state or "FAILED" in state or "CANCELLED" in state:
            break
        time.sleep(args.poll_sec)

    out_meta = {
        "name": getattr(job, "name", name),
        "state": str(getattr(job, "state", "")),
        "dest": _to_jsonable(getattr(job, "dest", None)),
        "dest_file": _to_jsonable(getattr(job, "dest_file", None)),
    }
    Path("data/gemini/batch_job_final.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    print("Saved final batch metadata to data/gemini/batch_job_final.json")


def _resolve_job_name(job_name: Optional[str], job_meta: str) -> str:
    if job_name:
        return job_name
    meta = json.loads(Path(job_meta).read_text(encoding="utf-8"))
    return meta["name"]


def _to_jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def cmd_fetch_batch_results(args: argparse.Namespace) -> None:
    client = _get_client(args.api_key)
    name = _resolve_job_name(args.job_name, args.job_meta)
    job = client.batches.get(name=name)
    state = str(getattr(job, "state", ""))
    dest = getattr(job, "dest", None)
    if not dest:
        raise RuntimeError(f"Batch job has no destination yet. state={state}")

    output_path = Path(args.output_path) if args.output_path else None
    file_name = getattr(dest, "file_name", None)
    if file_name:
        if output_path is None:
            safe = name.replace("/", "_")
            output_path = Path(f"data/gemini/batch_results_{safe}.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = client.files.download(file=file_name)
        output_path.write_bytes(content)
        print(f"Saved batch results file to: {output_path}")
        print(f"job={name} state={state} source_file={file_name}")
        return

    inlined = getattr(dest, "inlined_responses", None)
    if inlined:
        if output_path is None:
            safe = name.replace("/", "_")
            output_path = Path(f"data/gemini/batch_results_{safe}.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for r in inlined:
                if hasattr(r, "model_dump"):
                    f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"Saved inlined batch responses to: {output_path}")
        print(f"job={name} state={state} rows={len(inlined)}")
        return

    raise RuntimeError(f"No retrievable output found on job destination. state={state} dest={dest}")


def cmd_list_batches(args: argparse.Namespace) -> None:
    client = _get_client(args.api_key)
    rows = []
    pager = client.batches.list()
    for i, job in enumerate(pager):
        if i >= args.limit:
            break
        rows.append(
            {
                "name": getattr(job, "name", ""),
                "state": str(getattr(job, "state", "")),
                "model": getattr(job, "model", ""),
                "create_time": str(getattr(job, "create_time", "")),
                "update_time": str(getattr(job, "update_time", "")),
            }
        )

    if args.as_json:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print("No batch jobs found.")
        return

    for r in rows:
        print(
            f"name={r['name']} state={r['state']} model={r['model']} "
            f"created={r['create_time']} updated={r['update_time']}"
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    # Default pipeline now targets Nottingham-grade evaluation.
    cmd_evaluate_nottingham(args)


def _load_nottingham_gold() -> Dict[str, int]:
    gold = {}
    with Path("data/intermediate/clinical_features_validation_nottingham.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gold[r["PatientID"]] = int(float(r["target_nottingham_grade"]))
    return gold


def _safe_run_appendix(path: Path) -> str:
    rel = str(path).replace("\\", "/")
    rel = re.sub(r"^[./]+", "", rel)
    rel = re.sub(r"\.jsonl$", "", rel, flags=re.I)
    rel = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel or "run"


def _rel_response_path(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(Path.cwd().resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _load_or_init_run_name_map(response_files: List[Path], map_path: Path) -> Dict[str, str]:
    run_map: Dict[str, str] = {}
    if map_path.exists():
        try:
            obj = json.loads(map_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                run_map = {str(k): str(v) for k, v in obj.items()}
        except Exception:
            run_map = {}

    changed = False
    for fp in response_files:
        key = _rel_response_path(fp)
        if key not in run_map:
            run_map[key] = f"TODO_name_{_safe_run_appendix(fp)}"
            changed = True

    if changed or not map_path.exists():
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(run_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_map


def _evaluate_nottingham_file(
    results_jsonl: Path,
    out_eval_csv: Path,
    out_metrics_json: Path,
    out_per_class_csv: Path,
    out_confusion_csv: Path,
) -> Dict[str, object]:
    gold = _load_nottingham_gold()

    preds = {}
    with results_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            text = obj.get("response_text") if isinstance(obj, dict) and "response_text" in obj else flatten_text(obj)
            pred, conf, status, reason = parse_prediction_text_class_with_reason(text, [1, 2, 3])
            preds[key] = {"prediction": pred, "confidence": conf, "parse_status": status, "reason": reason}

    rows = []
    for pid, y in gold.items():
        p = preds.get(pid, {})
        rows.append(
            {
                "PatientID": pid,
                "gold_nottingham": y,
                "prediction": "" if p.get("prediction") is None else p.get("prediction"),
                "confidence": "" if p.get("confidence") is None else p.get("confidence"),
                "reason": p.get("reason", ""),
                "parse_status": p.get("parse_status", "missing_result"),
            }
        )
    out_eval_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_eval_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["PatientID", "gold_nottingham", "prediction", "confidence", "reason", "parse_status"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

    yt = [r["gold_nottingham"] for r in rows if r["prediction"] in (1, 2, 3)]
    yp = [r["prediction"] for r in rows if r["prediction"] in (1, 2, 3)]
    labels = [1, 2, 3]
    metrics = {}
    per_class_rows = []
    cm = np.zeros((3, 3), dtype=int)
    if yt:
        metrics["accuracy"] = accuracy_score(yt, yp)
        metrics["balanced_accuracy"] = balanced_accuracy_score(yt, yp)
        metrics["macro_precision"] = precision_score(yt, yp, average="macro", zero_division=0)
        metrics["macro_recall"] = recall_score(yt, yp, average="macro", zero_division=0)
        metrics["macro_f1"] = f1_score(yt, yp, average="macro")
        metrics["micro_precision"] = precision_score(yt, yp, average="micro", zero_division=0)
        metrics["micro_recall"] = recall_score(yt, yp, average="micro", zero_division=0)
        metrics["micro_f1"] = f1_score(yt, yp, average="micro")
        metrics["num_predicted"] = len(yt)
        metrics["num_gold"] = len(rows)
        report = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
        pred_counts = {c: int(sum(1 for p in yp if p == c)) for c in labels}
        gold_counts = {c: int(sum(1 for g in yt if g == c)) for c in labels}
        for c in labels:
            ckey = str(c)
            cvals = report.get(ckey, {})
            prow = {
                "class_label": c,
                "precision": float(cvals.get("precision", 0.0) or 0.0),
                "recall": float(cvals.get("recall", 0.0) or 0.0),
                "f1": float(cvals.get("f1-score", 0.0) or 0.0),
                "support": int(cvals.get("support", 0) or 0),
            }
            per_class_rows.append(prow)
            metrics[f"class_{c}_precision"] = prow["precision"]
            metrics[f"class_{c}_recall"] = prow["recall"]
            metrics[f"class_{c}_f1"] = prow["f1"]
            metrics[f"class_{c}_support"] = prow["support"]
            metrics[f"class_{c}_pred_count"] = pred_counts[c]
            metrics[f"class_{c}_pred_pct"] = pred_counts[c] / len(yp) if len(yp) > 0 else 0.0
            metrics[f"class_{c}_gold_count"] = gold_counts[c]
            metrics[f"class_{c}_gold_pct"] = gold_counts[c] / len(yt) if len(yt) > 0 else 0.0
        cm = confusion_matrix(yt, yp, labels=labels)
    else:
        metrics["num_predicted"] = 0
        metrics["num_gold"] = len(rows)
        for c in labels:
            per_class_rows.append({"class_label": c, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
            metrics[f"class_{c}_precision"] = 0.0
            metrics[f"class_{c}_recall"] = 0.0
            metrics[f"class_{c}_f1"] = 0.0
            metrics[f"class_{c}_support"] = 0
            metrics[f"class_{c}_pred_count"] = 0
            metrics[f"class_{c}_pred_pct"] = 0.0
            metrics[f"class_{c}_gold_count"] = 0
            metrics[f"class_{c}_gold_pct"] = 0.0
    out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    out_metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out_per_class_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_per_class_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_label", "precision", "recall", "f1", "support"])
        w.writeheader()
        for r in per_class_rows:
            w.writerow(r)
    out_confusion_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_confusion_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gold\\pred", "1", "2", "3"])
        for i, c in enumerate(labels):
            w.writerow([str(c)] + [int(cm[i, j]) for j in range(len(labels))])
    print(f"Saved evaluation rows to {out_eval_csv}")
    print(f"Saved metrics to {out_metrics_json}: {json.dumps(metrics)}")
    print(f"Saved per-class metrics to {out_per_class_csv}")
    print(f"Saved confusion matrix to {out_confusion_csv}")
    return metrics


def _plot_run_comparison(comparison_rows: List[Dict[str, object]], out_png: Path) -> None:
    if not comparison_rows:
        return

    run_names = [str(r.get("run_name", r["run"])) for r in comparison_rows]
    metric_names = ["Accuracy", "Micro F1", "Balanced Accuracy", "Macro F1"]
    metric_keys = ["accuracy", "micro_f1", "balanced_accuracy", "macro_f1"]
    metric_values = [[float(r.get(k, 0.0) or 0.0) for r in comparison_rows] for k in metric_keys]

    n_runs = len(run_names)
    n_metrics = len(metric_names)
    width = max(1400, 220 * n_runs)
    height = 680
    left = 90
    right = 340
    top = 70
    bottom = 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(1, n_metrics)
    bar_gap = 6
    max_bar_area = max(40, group_w * 0.85)
    bar_w = int(max(10, min(50, (max_bar_area - bar_gap * max(0, n_runs - 1)) / max(1, n_runs))))
    font = ImageFont.load_default()
    colors = [
        (66, 133, 244),
        (52, 168, 83),
        (251, 188, 5),
        (234, 67, 53),
        (171, 71, 188),
        (0, 172, 193),
        (255, 112, 67),
        (124, 179, 66),
    ]

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Axes + grid
    draw.line([(left, top), (left, top + plot_h)], fill=(30, 30, 30), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(30, 30, 30), width=2)
    for i in range(6):
        yv = i / 5
        y = top + int((1.0 - yv) * plot_h)
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225), width=1)
        draw.text((18, y - 6), f"{yv:.1f}", fill=(80, 80, 80), font=font)

    # Grouped bars by metric type.
    for mi, metric in enumerate(metric_names):
        group_left = left + int(mi * group_w + (group_w - (n_runs * bar_w + max(0, n_runs - 1) * bar_gap)) / 2)
        for ri, run_name in enumerate(run_names):
            val = max(0.0, min(1.0, metric_values[mi][ri]))
            h = int(val * plot_h)
            x0 = group_left + ri * (bar_w + bar_gap)
            x1 = x0 + bar_w
            y0 = top + plot_h - h
            y1 = top + plot_h
            fill = colors[ri % len(colors)]
            outline = tuple(max(0, c - 30) for c in fill)
            draw.rectangle([(x0, y0), (x1, y1)], fill=fill, outline=outline)
            ann = run_name if len(run_name) <= 12 else run_name[:12]
            tw = draw.textlength(ann, font=font)
            draw.text((x0 + (bar_w - tw) / 2, max(top, y0 - 13)), ann, fill=(70, 70, 70), font=font)
        twm = draw.textlength(metric, font=font)
        group_center = left + (mi + 0.5) * group_w
        draw.text((group_center - twm / 2, top + plot_h + 14), metric, fill=(40, 40, 40), font=font)

    # Title + legend with full mapped run names.
    title = "Nottingham Comparison Grouped by Metric (4 metrics)"
    tw = draw.textlength(title, font=font)
    draw.text(((width - tw) / 2, 20), title, fill=(20, 20, 20), font=font)
    legend_x = width - right + 20
    legend_y = top + 10
    draw.text((legend_x, legend_y - 20), "Runs (mapped names):", fill=(30, 30, 30), font=font)
    for ri, run_name in enumerate(run_names):
        y = legend_y + ri * 22
        fill = colors[ri % len(colors)]
        outline = tuple(max(0, c - 30) for c in fill)
        draw.rectangle([(legend_x, y), (legend_x + 14, y + 14)], fill=fill, outline=outline)
        label = run_name if len(run_name) <= 36 else run_name[:33] + "..."
        draw.text((legend_x + 20, y - 1), label, fill=(50, 50, 50), font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, format="PNG")


def _plot_prediction_distribution_by_run(comparison_rows: List[Dict[str, object]], out_png: Path) -> None:
    if not comparison_rows:
        return
    run_names = [str(r.get("run_name", r["run"])) for r in comparison_rows]
    pred_dist = [
        [
            float(r.get("class_1_pred_pct", 0.0) or 0.0),
            float(r.get("class_2_pred_pct", 0.0) or 0.0),
            float(r.get("class_3_pred_pct", 0.0) or 0.0),
        ]
        for r in comparison_rows
    ]
    gt_dist = [
        float(comparison_rows[0].get("class_1_gold_pct", 0.0) or 0.0),
        float(comparison_rows[0].get("class_2_gold_pct", 0.0) or 0.0),
        float(comparison_rows[0].get("class_3_gold_pct", 0.0) or 0.0),
    ]

    names = run_names + ["Ground Truth"]
    dists = pred_dist + [gt_dist]
    n = len(names)
    width = max(1100, 170 * n)
    height = 700
    left = 90
    right = 300
    top = 70
    bottom = 140
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(1, n)
    bar_w = max(26, int(group_w * 0.45))
    font = ImageFont.load_default()

    # Class colors: 1,2,3
    seg_colors = [(66, 133, 244), (52, 168, 83), (234, 67, 53)]
    seg_labels = ["Class 1", "Class 2", "Class 3"]

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(left, top), (left, top + plot_h)], fill=(30, 30, 30), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(30, 30, 30), width=2)

    for i in range(6):
        yv = i / 5
        y = top + int((1.0 - yv) * plot_h)
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225), width=1)
        draw.text((18, y - 6), f"{int(yv*100)}%", fill=(80, 80, 80), font=font)

    for i, name in enumerate(names):
        cx = left + int((i + 0.5) * group_w)
        x0 = cx - bar_w // 2
        x1 = cx + bar_w // 2
        y_cursor = top + plot_h
        vals = dists[i]
        for si, v in enumerate(vals):
            v = max(0.0, min(1.0, v))
            h = int(v * plot_h)
            y0 = y_cursor - h
            y1 = y_cursor
            fill = seg_colors[si]
            outline = tuple(max(0, c - 30) for c in fill)
            draw.rectangle([(x0, y0), (x1, y1)], fill=fill, outline=outline)
            if h >= 16:
                txt = f"{int(round(v*100))}%"
                tw = draw.textlength(txt, font=font)
                draw.text((x0 + (bar_w - tw) / 2, y0 + 2), txt, fill=(255, 255, 255), font=font)
            y_cursor = y0
        label = name if len(name) <= 16 else name[:16]
        tw = draw.textlength(label, font=font)
        draw.text((cx - tw / 2, top + plot_h + 12), label, fill=(60, 60, 60), font=font)

    title = "Prediction Distribution by Run (+ Ground Truth)"
    tw = draw.textlength(title, font=font)
    draw.text(((width - tw) / 2, 20), title, fill=(20, 20, 20), font=font)

    legend_x = width - right + 20
    legend_y = top + 10
    draw.text((legend_x, legend_y - 20), "Label colors:", fill=(30, 30, 30), font=font)
    for i, lbl in enumerate(seg_labels):
        y = legend_y + i * 24
        fill = seg_colors[i]
        outline = tuple(max(0, c - 30) for c in fill)
        draw.rectangle([(legend_x, y), (legend_x + 14, y + 14)], fill=fill, outline=outline)
        draw.text((legend_x + 20, y - 1), lbl, fill=(50, 50, 50), font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, format="PNG")


def cmd_evaluate_nottingham(args: argparse.Namespace) -> None:
    _evaluate_nottingham_file(
        results_jsonl=Path(args.results_jsonl),
        out_eval_csv=Path("data/results/nottingham_eval.csv"),
        out_metrics_json=Path("data/results/nottingham_metrics.json"),
        out_per_class_csv=Path("data/results/nottingham_per_class.csv"),
        out_confusion_csv=Path("data/results/nottingham_confusion_matrix.csv"),
    )


def cmd_evaluate_nottingham_all_runs(args: argparse.Namespace) -> None:
    response_dirs = [Path(x) for x in args.response_dirs]
    response_files: List[Path] = []
    for d in response_dirs:
        if d.exists() and d.is_dir():
            response_files.extend(sorted(d.glob("*.jsonl")))
    if not response_files:
        raise RuntimeError(f"No response JSONL files found in: {', '.join(str(d) for d in response_dirs)}")

    run_name_map = _load_or_init_run_name_map(response_files, Path(args.run_name_map))
    compare_rows = []
    for fp in response_files:
        appendix = _safe_run_appendix(fp)
        response_rel = _rel_response_path(fp)
        run_name = run_name_map.get(response_rel, appendix)
        eval_csv = Path(args.results_dir) / f"nottingham_eval__{appendix}.csv"
        metrics_json = Path(args.results_dir) / f"nottingham_metrics__{appendix}.json"
        per_class_csv = Path(args.results_dir) / f"nottingham_per_class__{appendix}.csv"
        confusion_csv = Path(args.results_dir) / f"nottingham_confusion_matrix__{appendix}.csv"
        metrics = _evaluate_nottingham_file(fp, eval_csv, metrics_json, per_class_csv, confusion_csv)
        compare_rows.append(
            {
                "run": appendix,
                "run_name": run_name,
                "response_jsonl": response_rel,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "micro_f1": metrics.get("micro_f1"),
                "class_1_f1": metrics.get("class_1_f1"),
                "class_2_f1": metrics.get("class_2_f1"),
                "class_3_f1": metrics.get("class_3_f1"),
                "class_1_pred_pct": metrics.get("class_1_pred_pct"),
                "class_2_pred_pct": metrics.get("class_2_pred_pct"),
                "class_3_pred_pct": metrics.get("class_3_pred_pct"),
                "class_1_gold_pct": metrics.get("class_1_gold_pct"),
                "class_2_gold_pct": metrics.get("class_2_gold_pct"),
                "class_3_gold_pct": metrics.get("class_3_gold_pct"),
                "num_predicted": metrics.get("num_predicted", 0),
                "num_gold": metrics.get("num_gold", 0),
            }
        )

    summary_csv = Path(args.results_dir) / "nottingham_run_comparison.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "run_name",
                "response_jsonl",
                "accuracy",
                "micro_f1",
                "balanced_accuracy",
                "macro_f1",
                "class_1_f1",
                "class_2_f1",
                "class_3_f1",
                "class_1_pred_pct",
                "class_2_pred_pct",
                "class_3_pred_pct",
                "class_1_gold_pct",
                "class_2_gold_pct",
                "class_3_gold_pct",
                "num_predicted",
                "num_gold",
            ],
        )
        w.writeheader()
        for r in compare_rows:
            w.writerow(r)

    plot_png = Path(args.results_dir) / "nottingham_run_comparison_accuracy_macro_f1.png"
    _plot_run_comparison(compare_rows, plot_png)
    dist_plot_png = Path(args.results_dir) / "nottingham_run_prediction_distribution.png"
    _plot_prediction_distribution_by_run(compare_rows, dist_plot_png)
    print(f"Wrote/updated run-name map: {args.run_name_map}")
    print(f"Wrote run comparison CSV: {summary_csv}")
    print(f"Wrote run comparison plot: {plot_png}")
    print(f"Wrote prediction distribution plot: {dist_plot_png}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Duke Breast MRI -> Gemini batch pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare-data", help="Prepare split, clinical CSV, series selection, and PNGs")
    p_prepare.add_argument("--base-dir", default="/mnt/e/BreastCancerData", help="Directory containing manifest and clinical files")
    p_prepare.add_argument("--png-size", type=int, default=512)
    p_prepare.set_defaults(func=cmd_prepare_data)

    p_prepare_n = sub.add_parser(
        "prepare-nottingham-rgb",
        help="Prepare Nottingham-labeled validation subset and RGB fusion images (3 slices per patient)",
    )
    p_prepare_n.add_argument("--base-dir", default="/mnt/e/BreastCancerData", help="Directory containing DUKE files")
    p_prepare_n.add_argument("--png-size", type=int, default=512)
    p_prepare_n.add_argument("--annotation-sheet", default="Sheet1")
    p_prepare_n.add_argument("--output-image-dir", default="data/images_rgb_nottingham_256crop")
    p_prepare_n.add_argument("--output-manifest-csv", default="data/intermediate/nottingham_rgb_image_manifest_256crop.csv")
    p_prepare_n.add_argument("--output-summary-json", default="data/intermediate/nottingham_prepare_summary_256crop.json")
    p_prepare_n.set_defaults(func=cmd_prepare_nottingham_rgb)

    p_upload = sub.add_parser("upload-files", help="Upload PNGs to Gemini Files API")
    p_upload.add_argument("--api-key", default=None)
    p_upload.add_argument("--max-retries", type=int, default=4)
    p_upload.set_defaults(func=cmd_upload_files)

    p_seed = sub.add_parser("seed-upload-manifest", help="Seed upload manifest (default: Nottingham RGB)")
    p_seed.add_argument("--source-manifest", default="data/intermediate/nottingham_rgb_image_manifest_256crop.csv")
    p_seed.set_defaults(func=cmd_seed_upload_manifest)

    p_seed_n = sub.add_parser("seed-upload-manifest-nottingham", help="Seed upload manifest for Nottingham RGB images")
    p_seed_n.add_argument("--source-manifest", default="data/intermediate/nottingham_rgb_image_manifest_256crop.csv")
    p_seed_n.set_defaults(func=cmd_seed_upload_manifest_nottingham)

    p_upload_m = sub.add_parser("upload-files-from-manifest", help="Upload image files listed in a manifest")
    p_upload_m.add_argument("--api-key", default=None)
    p_upload_m.add_argument("--manifest-in", required=True)
    p_upload_m.add_argument("--manifest-out", required=True)
    p_upload_m.add_argument("--max-retries", type=int, default=4)
    p_upload_m.set_defaults(func=cmd_upload_files_from_manifest)

    p_jsonl = sub.add_parser("build-jsonl", help="Build batch request JSONL (default: Nottingham)")
    p_jsonl.add_argument("--upload-manifest", default="data/intermediate/upload_manifest_nottingham.csv")
    p_jsonl.add_argument("--output-jsonl", default="data/gemini/batch_requests_validation_nottingham.jsonl")
    p_jsonl.add_argument(
        "--use-inline-data-from-local",
        action="store_true",
        help="If gemini_file_uri is missing, encode local_png_path as inline_data.",
    )
    p_jsonl.add_argument("--reasoning-strength", choices=["off", "low", "medium", "high"], default="medium")
    p_jsonl.add_argument("--thinking-budget", type=int, default=None)
    p_jsonl.set_defaults(func=cmd_build_jsonl)

    p_jsonl_n = sub.add_parser("build-jsonl-nottingham", help="Build Nottingham-grade batch request JSONL")
    p_jsonl_n.add_argument("--upload-manifest", default="data/intermediate/upload_manifest_nottingham.csv")
    p_jsonl_n.add_argument("--output-jsonl", default="data/gemini/batch_requests_validation_nottingham.jsonl")
    p_jsonl_n.add_argument(
        "--use-inline-data-from-local",
        action="store_true",
        help="If gemini_file_uri is missing, encode local_png_path as inline_data.",
    )
    p_jsonl_n.add_argument("--reasoning-strength", choices=["off", "low", "medium", "high"], default="medium")
    p_jsonl_n.add_argument("--thinking-budget", type=int, default=None)
    p_jsonl_n.add_argument(
        "--prompt-mode",
        choices=["multimodal", "non_image_only", "image_only"],
        default="multimodal",
        help="Prompt/input mode for Nottingham requests.",
    )
    p_jsonl_n.set_defaults(func=cmd_build_jsonl_nottingham)

    p_jsonl_u = sub.add_parser(
        "build-jsonl-nottingham-unimodal-baselines",
        help="Build two additional Nottingham JSONLs: non-image-only and image-only baselines",
    )
    p_jsonl_u.add_argument("--upload-manifest", default="data/intermediate/upload_manifest_nottingham.csv")
    p_jsonl_u.add_argument(
        "--output-jsonl-non-image-only",
        default="data/gemini/batch_requests_validation_nottingham_non_image_only.jsonl",
    )
    p_jsonl_u.add_argument(
        "--output-jsonl-image-only",
        default="data/gemini/batch_requests_validation_nottingham_image_only.jsonl",
    )
    p_jsonl_u.add_argument(
        "--use-inline-data-from-local",
        action="store_true",
        help="If gemini_file_uri is missing, encode local_png_path as inline_data.",
    )
    p_jsonl_u.add_argument("--reasoning-strength", choices=["off", "low", "medium", "high"], default="medium")
    p_jsonl_u.add_argument("--thinking-budget", type=int, default=None)
    p_jsonl_u.set_defaults(func=cmd_build_jsonl_nottingham_unimodal_baselines)

    p_test3 = sub.add_parser("build-test3-jsonl", help="Build deterministic 3-row low-cost test JSONL")
    p_test3.add_argument("--source-jsonl", default="data/gemini/batch_requests_validation_nottingham.jsonl")
    p_test3.add_argument("--output-jsonl", default="data/gemini/batch_requests_validation_nottingham_test3.jsonl")
    p_test3.add_argument("--output-keys", default="data/intermediate/test3_nottingham_selected_keys.txt")
    p_test3.add_argument("--expected-image-parts", type=int, default=3)
    p_test3.set_defaults(func=cmd_build_test3_jsonl)

    p_submit = sub.add_parser("submit-batch", help="Submit Gemini batch job")
    p_submit.add_argument("--api-key", default=None)
    p_submit.add_argument("--model", default="gemini-2.5-flash")
    p_submit.add_argument("--jsonl-path", default="data/gemini/batch_requests_validation_nottingham.jsonl")
    p_submit.add_argument("--display-name", default="duke-breast-mri-nottingham-validation")
    p_submit.set_defaults(func=cmd_submit_batch)

    p_infer = sub.add_parser(
        "infer-nottingham-online",
        help="Run direct (non-batch) multimodal Nottingham inference with Gemini",
    )
    p_infer.add_argument("--api-key", default=None)
    p_infer.add_argument("--model", default="gemini-2.5-flash")
    p_infer.add_argument("--upload-manifest", default="data/intermediate/upload_manifest_nottingham.csv")
    p_infer.add_argument("--output-jsonl", default="data/gemini/nottingham_online_results.jsonl")
    p_infer.add_argument(
        "--use-inline-data-from-local",
        action="store_true",
        help="If gemini_file_uri is missing, embed local_png_path bytes inline.",
    )
    p_infer.add_argument(
        "--prompt-mode",
        choices=["multimodal", "non_image_only", "image_only"],
        default="multimodal",
        help="Prompt/input mode for online inference.",
    )
    p_infer.add_argument("--reasoning-strength", choices=["off", "low", "medium", "high"], default="medium")
    p_infer.add_argument("--thinking-budget", type=int, default=None)
    p_infer.add_argument("--max-retries", type=int, default=3)
    p_infer.add_argument("--sleep-sec", type=float, default=0.0)
    p_infer.set_defaults(func=cmd_infer_nottingham_online)

    p_infer_u = sub.add_parser(
        "infer-nottingham-online-unimodal-baselines",
        help="Run direct online inference for both non-image-only and image-only baselines",
    )
    p_infer_u.add_argument("--api-key", default=None)
    p_infer_u.add_argument("--model", default="gemini-2.5-flash")
    p_infer_u.add_argument("--upload-manifest", default="data/intermediate/upload_manifest_nottingham.csv")
    p_infer_u.add_argument(
        "--output-jsonl-non-image-only",
        default="data/responses/nottingham_online_non_image_only_results.jsonl",
    )
    p_infer_u.add_argument(
        "--output-jsonl-image-only",
        default="data/responses/nottingham_online_image_only_results.jsonl",
    )
    p_infer_u.add_argument(
        "--use-inline-data-from-local",
        action="store_true",
        help="If gemini_file_uri is missing, embed local_png_path bytes inline.",
    )
    p_infer_u.add_argument("--reasoning-strength", choices=["off", "low", "medium", "high"], default="medium")
    p_infer_u.add_argument("--thinking-budget", type=int, default=None)
    p_infer_u.add_argument("--max-retries", type=int, default=3)
    p_infer_u.add_argument("--sleep-sec", type=float, default=0.0)
    p_infer_u.set_defaults(func=cmd_infer_nottingham_online_unimodal_baselines)

    p_poll = sub.add_parser("poll-batch", help="Poll Gemini batch job status")
    p_poll.add_argument("--api-key", default=None)
    p_poll.add_argument("--job-name", default=None, help="Specific batch job name (e.g. batches/123). Overrides --job-meta.")
    p_poll.add_argument("--job-meta", default="data/gemini/batch_job.json")
    p_poll.add_argument("--poll-sec", type=int, default=30)
    p_poll.set_defaults(func=cmd_poll_batch)

    p_list = sub.add_parser("list-batches", help="List recent Gemini batch jobs")
    p_list.add_argument("--api-key", default=None)
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--as-json", action="store_true")
    p_list.set_defaults(func=cmd_list_batches)

    p_fetch = sub.add_parser("fetch-batch-results", help="Fetch result payload for a specific batch job")
    p_fetch.add_argument("--api-key", default=None)
    p_fetch.add_argument("--job-name", default=None, help="Specific batch job name (e.g. batches/123). Overrides --job-meta.")
    p_fetch.add_argument("--job-meta", default="data/gemini/batch_job.json")
    p_fetch.add_argument("--output-path", default=None, help="Optional output path. Defaults under data/gemini/")
    p_fetch.set_defaults(func=cmd_fetch_batch_results)

    p_eval = sub.add_parser("evaluate", help="Parse batch results and compute metrics (default: Nottingham)")
    p_eval.add_argument("--results-jsonl", required=True, help="Batch output JSONL path")
    p_eval.set_defaults(func=cmd_evaluate)

    p_eval_n = sub.add_parser("evaluate-nottingham", help="Parse Nottingham batch results and compute metrics")
    p_eval_n.add_argument("--results-jsonl", required=True, help="Batch output JSONL path")
    p_eval_n.set_defaults(func=cmd_evaluate_nottingham)

    p_eval_all = sub.add_parser(
        "evaluate-nottingham-all-runs",
        help="Evaluate all response JSONLs in folders and produce per-run outputs plus comparison plot",
    )
    p_eval_all.add_argument(
        "--response-dirs",
        nargs="+",
        default=["data/responses", "data/batch_responses"],
        help="Directories containing response JSONL files",
    )
    p_eval_all.add_argument("--results-dir", default="data/results")
    p_eval_all.add_argument(
        "--run-name-map",
        default="data/results/response_run_name_map.json",
        help="JSON map from response JSONL path to a human-friendly run name.",
    )
    p_eval_all.set_defaults(func=cmd_evaluate_nottingham_all_runs)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
