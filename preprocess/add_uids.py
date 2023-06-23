import json


# Load Jsonl file
def load_jsonl(path, add_uid=False):
    uid = 0
    data = []
    with open(path) as f_in:
        all_lines = f_in.readlines()
        for line in all_lines:
            ins = json.loads(line)
            if add_uid:
                if 'uid' not in ins:
                    ins['uid'] = uid
                    uid += 1
            data.append(ins)
    return data


fname_in_n_out = 'fewrel.jsonl'
data_unlabel_and_label = load_jsonl(fname_in_n_out, add_uid=True)

# Write jsonl file
with open(fname_in_n_out, 'w') as f_out:
    for ins in data_unlabel_and_label:
        json.dump(ins, f_out)
        f_out.write('\n')
