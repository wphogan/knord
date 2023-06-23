import copy
import json
import os
import os.path
import time

import requests as req
from SPARQLWrapper import SPARQLWrapper, JSON
from bs4 import BeautifulSoup

from utils import parent_id_loop_fixes


def web_scrape(qid):
    parent_id, parent_name = False, False
    resp = req.get(f"https://www.wikidata.org/wiki/{qid}")
    html = resp.text
    soup = BeautifulSoup(html, features="lxml")
    lines = [i.get_text() for i in soup.find_all('div', {'class': 'wikibase-listview'})]
    lines_html = [i for i in
                  soup.find_all('div', {'class': 'wikibase-snakview-value wikibase-snakview-variation-valuesnak'})]
    try:
        if len(lines_html):
            parent_id = lines_html[0].contents[0].attrs['title']

        if len(lines):
            words = [w for w in lines[0].split('\n') if len(w)]
            # print(f'{t_id}: {key} --> {words[1]} ({t_id_link_title})') # t_id_link_title --> t_id to next level
            rel_names = ['subclass of', 'instance of', 'said to be the same as']
            for rel_name in rel_names:
                if rel_name in words:
                    loc_parent_name = 1 + words.index(rel_name)
                    parent_name = words[loc_parent_name]
                    return parent_id, parent_name



    except KeyError:
        print(f'{qid}: unk (title key error)')
        return parent_id, parent_name
    except AttributeError:
        print(f'{qid}: unk (Attribute Error)')
        return parent_id, parent_name

    return parent_id, parent_name


def save_dict(fname, data_dict):
    fname_out = f"{fname}.json"
    with open(fname_out, 'w') as fout:
        json_dumps_str = json.dumps(data_dict, indent=4)
        print(json_dumps_str, file=fout)
    print('saved file: ', fname_out)


def load_instance_from_txt(fname):
    all_instances = []
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write('')
    with open(fname) as f:
        all_lines = f.readlines()
    for line in all_lines:
        all_instances.append(line.strip())
    return all_instances


def load_instances_from_jsonl(fname):
    collected_dict = {}
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write('')

    with open(fname) as f:
        all_lines = f.readlines()
    for line in all_lines:
        dict_ins = json.loads(line)

        for k, v in dict_ins.items():
            if k not in collected_dict:
                collected_dict[k] = v
            else:
                assert v == collected_dict[k]
    return collected_dict


def save_jsonl_data(qid, parent_id, parent_name, f_out_id2parent, f_out_id2name):
    id2parent = {qid: parent_id}
    id2name = {parent_id: parent_name}
    json.dump(id2parent, f_out_id2parent)
    json.dump(id2name, f_out_id2name)
    f_out_id2parent.write('\n')
    f_out_id2name.write('\n')


def extract_qid(url):
    if '/Q' not in url:
        return 'unk'
    else:
        s = url.split('/Q')
        return f'Q{s[-1]}'


def run_query(qid, rel_type):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql",
                           agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
    results = {'results': {'bindings': []}}
    parent_id, parent_name = False, False
    rel_id = 'P279' if rel_type == 'subclass' else 'P31'
    rel_id = 'P460' if rel_type == 'same_as' else rel_id
    sparql.setQuery(
        f'''SELECT ?item ?itemLabel WHERE {{ wd:{qid} wdt:{rel_id} ?item SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }} }}''')
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if len(results['results']['bindings']):
        try:
            parent_url = results['results']['bindings'][0]['item']['value']
            parent_id = extract_qid(parent_url)
            parent_name = results['results']['bindings'][0]['itemLabel']['value']
        except KeyError:
            print(f'Key error looking for {rel_id}, on qid {qid}.')

    return parent_id, parent_name


def gather_dangling_parents(id2p):
    dangling_parents = []
    for qid, parent_id in id2p.items():
        if parent_id not in id2p:
            dangling_parents.append(parent_id)
    print(f'Found {len(dangling_parents)} dangling parents.')
    return dangling_parents


def gather_error_ids(fname):
    error_ids = set()
    with open(fname) as f_in:
        lines = f_in.readlines()
        for line in lines:
            error_ids.add(line.strip())

    return list(error_ids)


def main():
    fname_id2parent = "raw_id2parent.jsonl"
    fname_id2name = "raw_id2name.jsonl"
    fname_error_ids = "raw_error_ids.txt"
    id2parent = load_instances_from_jsonl(fname_id2parent)
    id2parent = parent_id_loop_fixes(id2parent)

    # RUN SETTINGS
    ids_to_resolve = gather_dangling_parents(id2parent)
    is_debug = False
    # Get an accurate count of ids to resolve on the first loop
    if not is_debug:
        for qid, parent, in id2parent.items():
            if qid in ids_to_resolve:
                ids_to_resolve.remove(qid)
    # Main loop
    n_success = 0
    with open(fname_id2parent, 'a') as f_out_id2parent, \
            open(fname_id2name, 'a') as f_out_id2name, \
            open(fname_error_ids, 'a') as f_out_error_ids:
        while len(ids_to_resolve):
            print(f'Resolving {len(ids_to_resolve)} qids....')
            new_ids_to_resolve = []

            for i, qid in enumerate(ids_to_resolve):
                # If id already resolved, skip
                if qid in id2parent and not is_debug:
                    if i % 1000 == 0:
                        print(f'Processing {i}: {qid}')
                    continue
                time.sleep(0.1)
                if i % 10 == 0:
                    print(f'Processing {i}: {qid}')

                # "subclass of"
                parent_id, parent_name = run_query(qid, 'subclass')
                if parent_id:
                    if qid not in id2parent:
                        new_ids_to_resolve.append(parent_id)
                        id2parent[qid] = parent_id
                        save_jsonl_data(qid, parent_id, parent_name, f_out_id2parent, f_out_id2name)
                        n_success += 1
                    continue

                # "instance of"
                parent_id, parent_name = run_query(qid, 'instance')
                if parent_id:
                    if qid not in id2parent:
                        new_ids_to_resolve.append(parent_id)
                        id2parent[qid] = parent_id
                        save_jsonl_data(qid, parent_id, parent_name, f_out_id2parent, f_out_id2name)
                        n_success += 1
                    continue

                # "same as"
                parent_id, parent_name = run_query(qid, 'same_as')
                if parent_id:
                    if qid not in id2parent:
                        new_ids_to_resolve.append(parent_id)
                        id2parent[qid] = parent_id
                        save_jsonl_data(qid, parent_id, parent_name, f_out_id2parent, f_out_id2name)
                        n_success += 1
                    continue

                # last resort: web scrape
                parent_id, parent_name = web_scrape(qid)
                if parent_id:
                    if qid not in id2parent:
                        new_ids_to_resolve.append(parent_id)
                        id2parent[qid] = parent_id
                        save_jsonl_data(qid, parent_id, parent_name, f_out_id2parent, f_out_id2name)
                        n_success += 1
                    continue

                f_out_error_ids.write(f'{qid}\n')
            ids_to_resolve = copy.deepcopy(new_ids_to_resolve)
        print(f'n_success: {n_success}')


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    main()
    print('\nCompelted file: ', os.path.basename(__file__))
