"""检查采集资料的身份、形状和可交叉引用性。"""

import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_fixture(name):
    return json.loads((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


class FixtureContractTests(unittest.TestCase):
    def test_record_manifest(self):
        manifest = read_fixture("manifest.json")
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        self.assertEqual(manifest["fixture"], "probe_nufft")
        self.assertEqual(manifest["records"], ["primary.json", "revision.json"])
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(revision["replaces"], primary["capture"])
        self.assertNotEqual(revision["capture"], primary["capture"])

    def test_related_input_records(self):
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        height, width = primary["frequency_grid"]["shape"]
        self.assertEqual(primary["frequency_grid"]["step"], [1, 1])
        read_keys = {query["read_key"] for query in primary["requests"]}
        self.assertFalse({trace["read_key"] for trace in revision["upstream"]} - read_keys)
        for query in primary["requests"]:
            y, x, dy, dx = query["window"]
            self.assertLessEqual(y + dy, height)
            self.assertLessEqual(x + dx, width)
        ids = set()
        for sample in primary["samples"] + primary["second_probe"]["samples"]:
            self.assertNotIn(sample["id"], ids)
            ids.add(sample["id"])
            self.assertEqual(np.asarray(sample["value"]).shape, (2,))
            self.assertGreaterEqual(sample["weight"], 0)
        self.assertAlmostEqual(abs(complex(*revision["calibration"]["phase"])), 1)
