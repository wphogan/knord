import os
import time

import requests


def read_rel_ids_file(fnames):
    rel_ids = []
    for fname in fnames:
        with open(fname) as f_in:
            lines = f_in.readlines()
            for line in lines:
                rel_ids.append(line.strip())
    return rel_ids


def main():
    # Load FewRel relation ids
    rel_ids = read_rel_ids_file(['few_rel_novel_class_ids.txt', 'few_rel_seen_class_ids.txt'])

    rel_counts = {'P710': '632695', 'P137': '566748', 'P674': '136266', 'P466': '38111', 'P136': '1671000',
                  'P306': '42452', 'P127': '493832', 'P400': '115873', 'P974': '72731', 'P1346': '260645',
                  'P460': '296124', 'P86': '235116', 'P118': '193933', 'P264': '353608', 'P750': '175290',
                  'P58': '198600', 'P3450': '85545', 'P105': '3628487', 'P276': '2833412', 'P101': '819431',
                  'P407': '16537473', 'P1001': '973537', 'P800': '111525', 'P131': '11819538', 'P177': '29646',
                  'P364': '374018', 'P2094': '244475', 'P361': '4577559', 'P641': '2010730', 'P59': '7374358',
                  'P413': '412121', 'P206': '84675', 'P412': '35322', 'P155': '1271900', 'P26': '754937',
                  'P410': '166688', 'P25': '637855', 'P463': '483291', 'P40': '1644343', 'P921': '25737346',
                  'P931': '12414', 'P4552': '64753', 'P140': '475680', 'P1923': '190586', 'P150': '1133315',
                  'P6': '28409', 'P27': '4777589', 'P449': '101480', 'P1435': '2239892', 'P175': '524960',
                  'P1344': '761549', 'P39': '1305223', 'P527': '2119425', 'P740': '46942', 'P706': '86080',
                  'P84': '83502', 'P495': '1534946', 'P123': '502250', 'P57': '355086', 'P22': '1008703',
                  'P178': '52809', 'P241': '133603', 'P403': '115301', 'P1411': '54201', 'P135': '62887',
                  'P991': '27048', 'P156': '1263534', 'P176': '151282', 'P31': '105465153', 'P1877': '13084',
                  'P102': '512905', 'P1408': '16735', 'P159': '469128', 'P3373': '434872', 'P1303': '214286',
                  'P17': '15704053', 'P106': '10391419', 'P551': '313958', 'P937': '400746', 'P355': '99501'}

    # Wikidata query service
    url = 'https://query.wikidata.org/sparql'

    for r_id in rel_ids:
        if r_id in rel_counts:
            continue

        query = f'SELECT (COUNT(*) AS ?count) WHERE  {{?item wdt:{r_id} ?value}}'
        time.sleep(1)
        r = requests.get(url, params={'format': 'json', 'query': query})
        data = r.json()
        try:
            rel_counts[r_id] = data['results']['bindings'][0]['count']['value']
        except KeyError:
            print('Key Error!!')
            rel_counts[r_id] = 0
    print(rel_counts)


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    main()
    print('\nCompelted file: ', os.path.basename(__file__))
