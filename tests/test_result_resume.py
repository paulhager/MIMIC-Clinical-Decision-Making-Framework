import pickle
import tempfile
import unittest
from pathlib import Path

from utils.logging import completed_patient_ids


class ResultResumeTest(unittest.TestCase):
    def test_reads_all_completed_ids_from_pickle_stream(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "results.pkl"
            with result_path.open("ab") as result_file:
                pickle.dump({101: "diagnosis one"}, result_file)
                pickle.dump({202: "diagnosis two"}, result_file)

            self.assertEqual(completed_patient_ids(result_path), {101, 202})

    def test_missing_result_stream_has_no_completed_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "missing.pkl"
            self.assertEqual(completed_patient_ids(result_path), set())


if __name__ == "__main__":
    unittest.main()
