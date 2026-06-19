"""Minimal D2-City XML parser (standalone — no torch dependency)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List


def parse_d2city_xml(xml_path: str) -> Dict[int, List[dict]]:
    """Parse CVAT-style D2-City XML → {frame_index: [annotation dicts]}."""
    root = ET.parse(xml_path).getroot()
    frame_annotations: dict[int, list[dict]] = defaultdict(list)

    for track in root.findall("track"):
        label = track.get("label", "")
        track_id = track.get("id", "")
        for box in track.findall("box"):
            frame_annotations[int(box.get("frame", "0"))].append(
                {
                    "label": label,
                    "xtl": float(box.get("xtl", "0")),
                    "ytl": float(box.get("ytl", "0")),
                    "xbr": float(box.get("xbr", "0")),
                    "ybr": float(box.get("ybr", "0")),
                    "occluded": box.get("occluded", "no"),
                    "cut": box.get("cut", "no"),
                    "track_id": track_id,
                }
            )

    return dict(frame_annotations)
