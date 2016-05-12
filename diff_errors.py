import os
import json
from utilities import writeToFile, deleteFiles

outf = './data/diff_errors.txt'
deleteFiles([outf])


def compare_errors(o_errors, n_errors):
    o_indices = [e['pos'] for e in o_errors]
    n_indices = [e['pos'] for e in n_errors]
    for o_idx in o_indices:
        if o_idx in n_indices:
            o_err = o_errors[o_indices.index(o_idx)]
            n_err = n_errors[n_indices.index(o_idx)]
            diff = {'token': o_err['token'], 'norm': o_err['norm'],
                    'pos': o_err['pos']}
            flag = False
            if o_err['out'] != n_err['out']:
                diff['out_old'] = o_err['out']
                diff['out_new'] = n_err['out']
                flag = True
            if o_err['class'] != n_err['class']:
                diff['class_old'] = o_err['class']
                diff['class_new'] = n_err['class']
                flag = True
            if flag:
                yield diff
        else:
            diff = o_errors[o_indices.index(o_idx)]
            diff.update({'src': 'old'})
            yield diff
    for n_idx in n_indices:
        if n_idx not in o_indices:
            diff = n_errors[n_indices.index(n_idx)]
            diff.update({'src': 'new'})
            yield diff


def diff(old_errors, new_errors):
    old_indices = [s['index'] for s in old_errors]
    new_indices = [s['index'] for s in new_errors]
    diff_errors = {}
    for n_error in new_errors:
        diff_errors[n_error['index']] = []
        if n_error['index'] in old_indices:
            o_error = old_errors[old_indices.index(n_error['index'])]
            for diff in compare_errors(o_error['errors'], n_error['errors']):
                out_str = '\n'.join(['{}: {}'.format(k, v) for k, v in diff.items()])
                writeToFile(outf, out_str + '\n\n')
                diff_errors[n_error['index']].append(diff)
        else:
            for err in n_error['errors']:
                err.update({'src': 'new'})
                out_str = '\n'.join(['{}: {}'.format(k, v) for k, v in err.items()])
                writeToFile(outf, out_str + '\n\n')
                diff_errors[n_error['index']].append(err)
    for o_error in old_errors:
        if o_error['index'] not in new_indices:
            diff_errors[o_error['index']] = []
            for err in o_error['errors']:
                err.update({'src': 'new'})
                out_str = '\n'.join(['{}: {}'.format(k, v) for k, v in err.items()])
                writeToFile(outf, out_str + '\n\n')
                diff_errors[o_error['index']].append(err)
    return diff_errors


def perform_diff(old_file=None, new_file=None):
    if not old_file:
        old_file = './data/norm_errors.0.json'
    if not new_file:
        new_file = './data/norm_errors.json'
    if not os.path.exists(old_file):
        print 'Nothing to compare against!!'
        return

    with open(old_file) as ifi:
        old_errors = json.load(ifi)

    with open(new_file) as ifi:
        new_errors = json.load(ifi)

    all_diffs = diff(old_errors, new_errors)

    rem = []
    for k, v in all_diffs.items():
        if len(v) == 0:
            rem.append(k)
    for r in rem:
        all_diffs.pop(r)

    print 'Number of differences : ', len(all_diffs)

    with open('./data/diff_errors.json', 'w') as ofi:
        json.dump(all_diffs, ofi)

    too_old_file = './data/norm_errors.1.json'
    os.rename(old_file, too_old_file)
    os.rename(new_file, old_file)


if __name__ == '__main__':
    perform_diff()
