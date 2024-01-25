# Parses the procedure names of either ICD9 or ICD10 into a dictionary with code as key and name as value
def parse_icd_names_file(icd_names_path):
    with open(icd_names_path, "r") as f:
        icd_names_raw = f.readlines()
    icd_names = {}
    for p in icd_names_raw:
        split = p.split()
        icd_names[split[0]] = " ".join(split[1:])
    return icd_names


# Parses the ICD mapping file into a dictionary. Works for both ICD9 to ICD10 and ICD10 to ICD9
def parse_icd_mapping_file(icd_mapping_path):
    with open(icd_mapping_path, "r") as f:
        icd_mapping_raw = f.readlines()
    icd_mapping = {}
    # One to many mapping so store in list
    for p in icd_mapping_raw:
        split = p.split()
        if split[0] not in icd_mapping:
            icd_mapping[split[0]] = [split[1]]
        else:
            icd_mapping[split[0]].append(split[1])
    return icd_mapping


# Converts a list of ICD codes from one version to another. Works for both ICD9 to ICD10 and ICD10 to ICD9. Also returns the title of the ICD codes
def icd_converter(
    icd_codes,
    input_icd_version,
    procedure_names_icd9_path,
    procedure_names_icd10_path,
    icd9_to_10_mapping_path,
    icd10_to_9_mapping_path,
):
    if input_icd_version == 9:
        icd_mapping = parse_icd_mapping_file(icd9_to_10_mapping_path)
        procedure_names = parse_icd_names_file(procedure_names_icd10_path)
    elif input_icd_version == 10:
        icd_mapping = parse_icd_mapping_file(icd10_to_9_mapping_path)
        procedure_names = parse_icd_names_file(procedure_names_icd9_path)
    else:
        print("Invalid input_icd_version. Only supports 9 and 10")
        return

    converted_codes = []
    converted_codes_title = []
    for c in icd_codes:
        if c not in icd_mapping:
            print("Could not find {} in mapping".format(c))
            continue
        for c2 in icd_mapping[c]:
            if c2 not in procedure_names:
                print("Could not find {} in procedure names".format(c2))
                continue
            converted_codes.append(c2)
            converted_codes_title.append(procedure_names[c2])

    return converted_codes, converted_codes_title


def uniqueify_lists(l1, l2):
    unique_elements = set()
    final_list1 = []
    final_list2 = []

    for i, item in enumerate(l1):
        if item not in unique_elements:
            unique_elements.add(item)
            final_list1.append(item)
            final_list2.append(l2[i])
    return final_list1, final_list2


def get_title_from_code(
    code, icd_version, procedure_names_icd9_path, procedure_names_icd10_path
):
    if icd_version == 9:
        procedure_names_icd9 = parse_icd_names_file(procedure_names_icd9_path)
        return procedure_names_icd9[code]
    elif icd_version == 10:
        procedure_names_icd10 = parse_icd_names_file(procedure_names_icd10_path)
        return procedure_names_icd10[code]
    else:
        print("Invalid icd_version. Only supports 9 and 10")
        return


### ERCP ###

ERCP_PROCEDURES_ICD10 = [
    "0F798DZ",
    "0F798ZZ",
    "0F7D8DZ",
    "0FB98ZX",
    "0FBD8ZX",
    "0FC98ZZ",
    "0FD98ZX",
    "0FDG8ZX",
    "BF101ZZ",
    "BF10YZZ",
    "BF11YZZ",
    "BF141ZZ",
    "BF14YZZ",
    "0FJB8ZZ",
    "0FJD8ZZ",
]

ERCP_PROCEDURES_ICD10_TITLES = [
    "Dilation of Common Bile Duct with Intraluminal Device, Via Natural or Artificial Opening Endoscopic",
    "Dilation of Common Bile Duct, Via Natural or Artificial Opening Endoscopic",
    "Dilation of Pancreatic Duct with Intraluminal Device, Via Natural or Artificial Opening Endoscopic",
    "Excision of Common Bile Duct, Via Natural or Artificial Opening Endoscopic, Diagnostic",
    "Excision of Pancreatic Duct, Via Natural or Artificial Opening Endoscopic, Diagnostic",
    "Extirpation of Matter from Common Bile Duct, Via Natural or Artificial Opening Endoscopic",
    "Extraction of Common Bile Duct, Via Natural or Artificial Opening Endoscopic, Diagnostic",
    "Extraction of Pancreas, Via Natural or Artificial Opening Endoscopic, Diagnostic",
    "Fluoroscopy of Bile Ducts using Low Osmolar Contrast",
    "Fluoroscopy of Bile Ducts using Other Contrast",
    "Fluoroscopy of Biliary and Pancreatic Ducts using Other Contrast",
    "Fluoroscopy of Gallbladder, Bile Ducts and Pancreatic Ducts using Low Osmolar Contrast",
    "Fluoroscopy of Gallbladder, Bile Ducts and Pancreatic Ducts using Other Contrast",
    "Inspection of Hepatobiliary Duct, Via Natural or Artificial Opening Endoscopic",
    "Inspection of Pancreatic Duct, Via Natural or Artificial Opening Endoscopic",
]

ERCP_PROCEDURES_ICD9 = [
    8766,
    5181,
    5184,
    5221,
    5187,
    5293,
    5188,
    5110,
    5185,
    8754,
    5114,
    9705,
]

ERCP_PROCEDURES_ICD9_TITLES = [
    "Contrast pancreatogram",
    "Dilation of sphincter of Oddi",
    "Endoscopic dilation of ampulla and biliary duct",
    "Endoscopic excision or destruction of lesion or tissue of pancreatic duct",
    "Endoscopic insertion of stent (tube) into bile duct",
    "Endoscopic insertion of stent (tube) into pancreatic duct",
    "Endoscopic removal of stone(s) from biliary tract",
    "Endoscopic retrograde cholangiopancreatography [ERCP]",
    "Endoscopic sphincterotomy and papillotomy",
    "Other cholangiogram",
    "Other closed [endoscopic] biopsy of biliary duct or sphincter of Oddi",
    "Replacement of stent (tube) in biliary or pancreatic duct",
]

ERCP_PROCEDURES_KEYWORDS = [
    "biliary stent",
    "biliary cannulation",
    "ercp",
    "endoscopic retrograde cholangiography",
    "endoscopic retrograde cholangiopancreatography",
    "cholangiogram",
    "cbd stent",
    "pancreatic stent",
    "sphincterotomy",
    "sphinctertomy",
]


### COLECTOMY ###

COLECTOMY_PROCEDURES_ICD9 = [
    4575,
    4576,
    4863,
    4562,
    4542,
    4579,
    4531,
    4530,
    4533,
    4562,
    1739,
    1734,
    1736,
    4572,
    4573,
    4576,
    1733,
    4541,
    1732,
    5459,
    5451,
    4574,
    9624,
]

COLECTOMY_PROCEDURES_ICD9_TITLES = [
    "Open and other left hemicolectomy",
    "Open and other sigmoidectomy",
    "Other anterior resection of rectum",
    "Other partial resection of small intestine",
    "Endoscopic polypectomy of large intestine",
    "Other and unspecified partial excision of large intestine",
    "Other local excision of lesion of duodenum",
    "Endoscopic excision or destruction of lesion of duodenum",
    "Local excision of lesion or tissue of small intestine, except duodenum",
    "Other partial resection of small intestine",
    "Other laparoscopic partial excision of large intestine",
    "Laparoscopic resection of transverse colon",
    "Laparoscopic sigmoidectomy",
    "Open and other cecectomy",
    "Open and other right hemicolectomy",
    "Open and other sigmoidectomy",
    "Laparoscopic right hemicolectomy",
    "Excision of lesion or tissue of large intestine",
    "Laparoscopic cecectomy",
    "Other lysis of peritoneal adhesions",
    "Laparoscopic lysis of peritoneal adhesions",
    "Open and other resection of transverse colon",
]

COLECTOMY_PROCEDURES_ICD10 = [
    "0DBM8ZZ",
    "0DBH8ZZ",
    "0DB90ZZ",
    "0DB94ZZ",
    "0DBB0ZZ",
    "0DTK4ZZ",
    "0DTM4ZZ",
    "0DTL4ZZ",
    "0DTN4ZZ",
    "0DBG4ZZ",
    "0DTH0ZZ",
    "0DTF0ZZ",
    "0DTN0ZZ",
    "0DB80ZZ",
    "0DBA0ZZ",
    "0DB84ZZ",
    "0DBB4ZZ",
    "0DTF4ZZ",
    "0DBH0ZZ",
    "0DTH4ZZ",
    "0DNN0ZZ",
    "0DBM4ZZ",
    "0DTM0ZZ",
    "0DNB0ZZ",
    "0DBM0ZZ",
    "0DBE0ZZ",
    "0DBG0ZZ",
    "0DNE0ZZ",
    "0DNE4ZZ",
    "0DN90ZZ",
    "0DNL4ZZ",
    "0DNA0ZZ",
    "0DBH4ZZ",
    "0DTK0ZZ",
    "0DTL0ZZ",
    "0DNB4ZZ",
    "0DNL0ZZ",
    "0DNN4ZZ",
    "0DBA8ZZ",
    "0DBA4ZZ",
    "0DNB7ZZ",
    "0DBE4ZZ",
    "0DTNFZZ",
    "0DNA4ZZ",
    "0DN94ZZ",
    "0DB98ZZ",
    "0DTH8ZZ",
    "0DTN7ZZ",
    "0DBB3ZZ",
    "0DBN0ZZ",
    "0DBN4ZZ",
    "0DT80ZZ",
]

COLECTOMY_PROCEDURES_ICD10_TITLES = [
    "Excision of Descending Colon, Via Natural or Artificial Opening Endoscopic",
    "Excision of Cecum, Via Natural or Artificial Opening Endoscopic",
    "Excision of Duodenum, Open Approach",
    "Excision of Duodenum, Percutaneous Endoscopic Approach",
    "Excision of Ileum, Open Approach",
    "Resection of Ascending Colon, Percutaneous Endoscopic Approach",
    "Resection of Descending Colon, Percutaneous Endoscopic Approach",
    "Resection of Transverse Colon, Percutaneous Endoscopic Approach",
    "Resection of Sigmoid Colon, Percutaneous Endoscopic Approach",
    "Excision of Left Large Intestine, Percutaneous Endoscopic Approach",
    "Resection of Cecum, Open Approach",
    "Resection of Right Large Intestine, Open Approach",
    "Resection of Sigmoid Colon, Open Approach",
    "Excision of Small Intestine, Open Approach",
    "Excision of Jejunum, Open Approach",
    "Excision of Small Intestine, Percutaneous Endoscopic Approach",
    "Excision of Ileum, Percutaneous Endoscopic Approach",
    "Resection of Right Large Intestine, Percutaneous Endoscopic Approach",
    "Excision of Cecum, Open Approach",
    "Resection of Cecum, Percutaneous Endoscopic Approach",
    "Release Sigmoid Colon, Open Approach",
    "Excision of Descending Colon, Percutaneous Endoscopic Approach",
    "Resection of Descending Colon, Open Approach",
    "Release Ileum, Open Approach",
    "Excision of Descending Colon, Open Approach",
    "Excision of Large Intestine, Open Approach",
    "Excision of Left Large Intestine, Open Approach",
    "Release Large Intestine, Open Approach",
    "Release Large Intestine, Percutaneous Endoscopic Approach",
    "Release Duodenum, Open Approach",
    "Release Transverse Colon, Percutaneous Endoscopic Approach",
    "Release Jejunum, Open Approach",
    "Excision of Cecum, Percutaneous Endoscopic Approach",
    "Resection of Ascending Colon, Open Approach",
    "Resection of Transverse Colon, Open Approach",
    "Release Ileum, Percutaneous Endoscopic Approach",
    "Release Transverse Colon, Open Approach",
    "Release Sigmoid Colon, Percutaneous Endoscopic Approach",
    "Excision of Jejunum, Via Natural or Artificial Opening Endoscopic",
    "Excision of Jejunum, Percutaneous Endoscopic Approach",
    "Release Ileum, Via Natural or Artificial Opening",
    "Excision of Large Intestine, Percutaneous Endoscopic Approach",
    "Resection of Sigmoid Colon, Via Natural or Artificial Opening With Percutaneous Endoscopic Assistance",
    "Release Jejunum, Percutaneous Endoscopic Approach",
    "Release Duodenum, Percutaneous Endoscopic Approach",
    "Excision of Duodenum, Via Natural or Artificial Opening Endoscopic",
    "Resection of Cecum, Via Natural or Artificial Opening Endoscopic",
    "Resection of Sigmoid Colon, Via Natural or Artificial Opening",
    "Excision of Ileum, Percutaneous Approach",
    "Excision of Sigmoid Colon, Open Approach",
    "Excision of Sigmoid Colon, Percutaneous Endoscopic Approach",
    "Resection of Small Intestine, Open Approach",
]

COLECTOMY_PROCEDURES_KEYWORDS = [
    "low anterior resection",
    "colectomy",
    "colonic resection",
    "colostomy",
    "resection of rectosigmoid colon",
    "resection of sigmoid colon",
    "rectosigmoid resection",
    "resection of colon",
    "sigmoidectomy",
    "sigmoid resection",
    "small bowel resection",
]

ALTERNATE_COLECTOMY_KEYWORDS = [
    {
        "location": "colon",
        "modifiers": ["surgery", "surgical", "removal", "remove"],
    }
]


### APPENDECTOMY ###

APPENDECTOMY_PROCEDURES_ICD9 = [4701, 4709]

APPENDECTOMY_PROCEDURES_ICD9_TITLES = [
    "Laparoscopic Appendectomy",
    "Other appendectomy",
]

APPENDECTOMY_PROCEDURES_ICD10 = ["0DTJ4ZZ", "0DTJ0ZZ"]

APPENDECTOMY_PROCEDURES_ICD10_TITLES = [
    "Resection of Appendix, Percutaneous Endoscopic Approach",
    "Resection of Appendix, Open Approach",
]

APPENDECTOMY_PROCEDURES_KEYWORDS = ["appendectomy"]

ALTERNATE_APPENDECTOMY_KEYWORDS = [
    {
        "location": "appendix",
        "modifiers": ["surgery", "surgical", "removal", "remove"],
    }
]


### CHOLECYSTECTOMY ###

CHOLECYSTECTOMY_PROCEDURES_ICD9 = [5123, 5122, 5121, 5102, 5124, 5103]

CHOLECYSTECTOMY_PROCEDURES_ICD9_TITLES = [
    "Laparoscopic cholecystectomy",
    "Cholecystectomy",
    "Other partial cholecystectomy",
    "Trocar cholecystostomy",
    "Laparoscopic partial cholecystectomy",
    "Other cholecystostomy",
]

CHOLECYSTECTOMY_PROCEDURES_ICD10 = ["0FB44ZZ", "0FB40ZZ"]

CHOLECYSTECTOMY_PROCEDURES_ICD10_TITLES = [
    "Excision of Gallbladder, Percutaneous Endoscopic Approach",
    "Excision of Gallbladder, Open Approach",
]

CHOLECYSTECTOMY_PROCEDURES_KEYWORDS = [
    "cholecystectomy",
    "cholecystecotmy",
    "cholecsytectomy",
    "cholecystecomy",
    "cholecystecomy",
    "cholecytectomy",
    "laparoscopic cholecystitis",
    "cholecyctectomy",
]

ALTERNATE_CHOLECYSTECTOMY_KEYWORDS = [
    {
        "location": "gallbladder",
        "modifiers": ["surgery", "surgical", "removal", "remove"],
    }
]


### DRAINAGE ###

DRAINAGE_PROCEDURES_ICD9 = [5491]

DRAINAGE_PROCEDURES_ICD9_TITLES = [
    "Percutaneous abdominal drainage",
]

DRAINAGE_PROCEDURES_ALL_ICD10 = [
    "0W2JX0Z",
    "0W9J30Z",
    "0W9J3ZZ",
    "0W9J3ZX",
    "0W9G30Z",
    "0W9G3ZZ",
    "0W9G3ZX",
]

DRAINAGE_PROCEDURES_ALL_ICD10_TITLES = [
    "Change Drainage Device in Pelvic Cavity, External Approach",
    "Drainage of Pelvic Cavity with Drainage Device, Percutaneous Approach",
    "Drainage of Pelvic Cavity, Percutaneous Approach",
    "Drainage of Pelvic Cavity, Percutaneous Approach, Diagnostic",
    "Drainage of Peritoneal Cavity with Drainage Device, Percutaneous Approach",
    "Drainage of Peritoneal Cavity, Percutaneous Approach",
    "Drainage of Peritoneal Cavity, Percutaneous Approach, Diagnostic",
]

DRAINAGE_PROCEDURES_PANCREATITIS_ICD10 = [
    "0F2BX0Z",
    "0F9430Z",
    "0F9G30Z",
    "0F998ZZ",
    "0F998ZX",
]

DRAINAGE_PROCEDURES_PANCREATITIS_ICD10_TITLES = [
    "Change Drainage Device in Hepatobiliary Duct, External Approach",
    "Drainage of Gallbladder with Drainage Device, Percutaneous Approach",
    "Drainage of Pancreas with Drainage Device, Percutaneous Approach",
    "Drainage of Common Bile Duct, Via Natural or Artificial Opening Endoscopic",
    "Drainage of Common Bile Duct, Via Natural or Artificial Opening Endoscopic, Diagnostic",
]

DRAINAGE_PROCEDURES_KEYWORDS = ["drain", "pigtail", "catheter", "aspiration"]

DRAINAGE_LOCATIONS_PANCREATITIS = [
    "abscess",
    "abdom",
    "pelvic",
    "peritoneal",
    "pancrea",
    "gallbladder",
    "biliary",
    "bile duct",
    "perirectal",
]

DRAINAGE_LOCATIONS_DIVERTICULITIS = [
    "abscess",
    "abdom",
    "pelvic",
    "peritoneal",
    "pericolonic",
    "sigmoid",
    "diverticular",
    "pararectal",
]

ALTERNATE_DRAINAGE_KEYWORDS_DIVERTICULITIS = [
    {"location": loc, "modifiers": DRAINAGE_PROCEDURES_KEYWORDS}
    for loc in DRAINAGE_LOCATIONS_DIVERTICULITIS
]

ALTERNATE_DRAINAGE_KEYWORDS_PANCREATITIS = [
    {"location": loc, "modifiers": DRAINAGE_PROCEDURES_KEYWORDS}
    for loc in DRAINAGE_LOCATIONS_PANCREATITIS
]
