import unittest

from platforms.prototype.data import (
    FIGURE4_ROOT, TABLE4_PATH, TABLE5_ROOT, gallery_samples, paper_table2,
    pipeline_status, table4, table4_rows, table5_status, figure4_status,
)


class ResearchPrototypeDataTests(unittest.TestCase):
    def test_paper_table_is_marked_source_and_has_expected_reference_rows(self):
        rows = paper_table2()
        adm = [row for row in rows if row["ckpt"] == "lsun_adm"]
        self.assertEqual(len(adm), 9)  # 8 generators + the published average
        self.assertEqual(next(row for row in adm if row["generator"] == "avg")["acc"], "99.9")

    def test_table4_keeps_measured_and_missing_cells_distinct(self):
        ledger = table4()
        self.assertIsNotNone(ledger)
        self.assertEqual(ledger["subset_status"], "pilot")
        self.assertEqual(ledger["per_cell"], 200)
        self.assertEqual(ledger["missing_steps"], [50])
        rows = table4_rows(ledger)
        s5_adm = next(row for row in rows if row["S"] == 5 and row["Generator"] == "adm")
        s50_adm = next(row for row in rows if row["S"] == 50 and row["Generator"] == "adm")
        self.assertEqual(s5_adm["Status"], "ran")
        self.assertAlmostEqual(s5_adm["AP"], 0.9985608145331707)
        self.assertEqual(s50_adm["Status"], "not run")
        self.assertIsNone(s50_adm["AP"])

    def test_gallery_requires_metadata_and_both_images(self):
        samples = gallery_samples()
        self.assertGreater(len(samples), 0)
        for sample in samples:
            self.assertTrue(sample["orig"].is_file())
            self.assertTrue(sample["diff"].is_file())
            self.assertIn("sha256", sample["meta"])

    def test_table5_and_figure4_report_evidence_not_assumptions(self):
        table5 = table5_status()
        self.assertIsNotNone(table5["protocol"])
        self.assertEqual(table5["protocol"]["representations"], ["REC", "RGB", "RGB_DIRE", "DIRE"])
        self.assertEqual(table5["ledger_count"], 0)
        figure4 = figure4_status()
        self.assertEqual(figure4["ledger_count"], 0)
        self.assertFalse(FIGURE4_ROOT.exists())
        self.assertTrue(TABLE4_PATH.is_file())
        self.assertTrue(TABLE5_ROOT.is_dir())

    def test_pipeline_does_not_claim_missing_studies_are_complete(self):
        statuses = {row["Stage"]: row["Status"] for row in pipeline_status()}
        self.assertEqual(statuses["Table 4 pilot"], "partial")
        self.assertEqual(statuses["Table 5 pilot"], "inputs only")
        self.assertEqual(statuses["Figure 4 pilot"], "not run")


if __name__ == "__main__":
    unittest.main()
