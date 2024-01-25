import ast
import pickle


def parse_patient(patient_buffer):
    patient_id = patient_buffer[0].split("Processing patient: ")[1].strip()
    chain = "\n".join(patient_buffer[1:-1])
    eval_results = patient_buffer[-1].split("Eval: ")[1].strip()
    eval_results = ast.literal_eval(eval_results)
    return patient_id, chain, eval_results


def parse_log_file(logfile, debug=False):
    # Line for line parser for logfiles
    patients = {}
    with open(logfile, "r") as f:
        patient_buffer = []
        inside_entry = False
        for line in f:
            # New patient
            if "Processing patient:" in line:
                if inside_entry and debug:
                    print(
                        f"Error: Found new patient while processing patient: {patient_buffer[0]}"
                    )
                    # print(line)
                inside_entry = True
                patient_buffer = [line]
            # End of patient
            elif inside_entry and "Eval:" in line:
                inside_entry = False
                patient_buffer.append(line)
                patient_id, chain, eval_results = parse_patient(patient_buffer)
                patients[patient_id] = {"chain": chain, "eval_results": eval_results}
                patient_buffer = []
            # Inside patient
            elif inside_entry:
                patient_buffer.append(line)
    return patients


# Used for continuous logging
def append_to_pickle_file(filename, data):
    with open(filename, "ab") as f:
        pickle.dump(data, f)


def read_from_pickle_file(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
