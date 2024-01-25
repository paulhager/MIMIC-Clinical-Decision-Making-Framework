import unittest
from dataset.discharge import extract_diagnosis_from_discharge


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_extract_diagnosis_from_discharge(self):
        discharge = """Discharge Disposition:
Home
 
Discharge Diagnosis:
Left ureteropelvic junction stone
Hydronephrosis 
Left-sided diverticulitis

 
Discharge Condition:
good"""
        output = extract_diagnosis_from_discharge(discharge)
        expected = """Left ureteropelvic junction stone
Hydronephrosis 
Left-sided diverticulitis"""
        self.assertEqual(output, expected)

    def test_extract_diagnosis_from_discharge_primary(self):
        discharge = """Discharge Disposition:
Extended Care
 
Facility:
___
 
Discharge Diagnosis:
Primary Diagnoses:
-Strep Anginosus Bacteremia
-Liver Abscess
-Respiratory Failure
-Acute Kidney Injury
-Diverticulitis

 
Discharge Condition:
Mental Status: Confused - sometimes.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - requires assistance or aid (walker 
or cane)."""
        output = extract_diagnosis_from_discharge(discharge)
        expected = """Primary Diagnoses:
-Strep Anginosus Bacteremia
-Liver Abscess
-Respiratory Failure
-Acute Kidney Injury
-Diverticulitis"""
        self.assertEqual(output, expected)

    def test_extract_diagnosis_from_discharge_underscore(self):
        discharge = """Home With Service
 
Facility:
___
 
___ Diagnosis:
- Complicated diverticulitis
- Pericolonic abscess s/p drainage
- Hepatic abscesses s/p drainage
- Intermittent asymptomatic hypotension
 
Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent."""

        output = extract_diagnosis_from_discharge(discharge)
        expected = """- Complicated diverticulitis
- Pericolonic abscess s/p drainage
- Hepatic abscesses s/p drainage
- Intermittent asymptomatic hypotension"""
        self.assertEqual(output, expected)

    def test_extract_diagnosis_from_discharge_primary_secondary_diagnosis(self):
        discharge = """Discharge Disposition:
Extended Care
 
Facility:
___
 
Discharge Diagnosis:
Primary diagnosis:
==================
Cholangiocarcinoma
Portal vein thrombosis
Cirrhosis

Secondary diagnosis:
====================
Gastroesophageal reflux disease
Constipation
Epilepsy
Hearing loss
Hypertension

 
Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - requires assistance or aid (walker 
or cane)."""
        output = extract_diagnosis_from_discharge(discharge)

        expected = """Primary diagnosis:
==================
Cholangiocarcinoma
Portal vein thrombosis
Cirrhosis

Secondary diagnosis:
====================
Gastroesophageal reflux disease
Constipation
Epilepsy
Hearing loss
Hypertension"""
        self.assertEqual(output, expected)

    def test_extract_diagnosis_from_discharge_primary_diagnosis(self):
        discharge = """Discharge Disposition:
Home
 
Discharge Diagnosis:
Primary:
Acute Pancreatitis
Gastroesophageal Reflux Disease

 
Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent."""
        output = extract_diagnosis_from_discharge(discharge)

        expected = """Primary:
Acute Pancreatitis
Gastroesophageal Reflux Disease"""
        self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
