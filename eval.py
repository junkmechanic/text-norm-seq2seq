from __future__ import print_function
import multiprocessing
# from datetime import datetime
from utilities import loadJSON, saveJSON


PNUM = 62


def evaluateStats(samples):
    correct_norm = 0.0
    total_norm = 0.0
    total_nsw = 0.0
    for tweet in samples:
        input, pred, goal = (tweet['input'], tweet['prediction'],
                             tweet['output'])
        for i in range(len(input)):
            if pred[i].lower() != input[i].lower() and \
                    goal[i].lower() == pred[i].lower() and goal[i].strip:
                correct_norm += 1
            if goal[i].lower() != input[i].lower() and goal[i].strip():
                total_nsw += 1
            if pred[i].lower() != input[i].lower() and pred[i].strip():
                total_norm += 1
    precision = correct_norm / total_norm
    recall = correct_norm / total_nsw
    print('T:{} N:{} C:{}'.format(total_nsw, total_norm, correct_norm))
    if precision != 0 and recall != 0:
        f_measure = (2 * precision * recall) / (precision + recall)
        print("precision: {:.4f}".format(precision))
        print("recall:    {:.4f}".format(recall))
        print("F1:        {:.4f}".format(f_measure))
    else:
        print("precision: {:.4f}".format(precision))
        print("recall:    {:.4f}".format(recall))


def classifyErrors(sample):
    current = multiprocessing.current_process()
    try:
        input, pred, goal = (sample['input'], sample['prediction'],
                             sample['output'])
        sample_props = {
            'index': sample['index'],
            'in': input,
            'goal': goal,
            'pred': pred,
            'flags': sample['flags'],
            'errors': [],
            'R2W': 0, 'W2R': 0, 'W2W_C': 0, 'W2W_NC': 0
        }
        for i in range(len(input)):
            new_error = {
                'token': input[i],
                'norm': goal[i],
                'out': pred[i],
                'pos': str(i)
            }
            include = True
            if goal[i].lower() == input[i].lower():
                if pred[i].lower() != input[i].lower():
                    sample_props['R2W'] += 1
                    new_error['class'] = 'R2W'
                else:
                    include = False
            if goal[i].lower() != input[i].lower() and goal[i].strip:
                if pred[i].lower() == input[i].lower():
                    sample_props['W2W_NC'] += 1
                    new_error['class'] = 'W2W_NC'
                elif pred[i].lower() == goal[i].lower():
                    sample_props['W2R'] += 1
                    new_error['class'] = 'W2R'
                else:
                    sample_props['W2W_C'] += 1
                    new_error['class'] = 'W2W_C'
            if include:
                sample_props['errors'].append(new_error)
        # print "{} : {} : analyzed sample {}".format(
        #     datetime.now().ctime(),
        #     current.name,
        #     sample['index']
        # )
        return sample_props
    except Exception as e:
        import traceback
        print("Error from Process {}".format(current.name))
        print(traceback.format_exc())
        return {"error": str(e)}


def reportErrors(samples):
    pool = multiprocessing.Pool(processes=PNUM)
    stats = {'R2W': 0, 'W2R': 0, 'W2W_C': 0, 'W2W_NC': 0}
    all_errors = pool.map(classifyErrors, samples)

    stats = {key: reduce(lambda x, y: x + y, [smpl[key] for smpl in
                                              all_errors])
             for key in stats}
    print(stats)
    saveJSON(all_errors, './data/norm_errors.json')


def evaluate(samples=None):
    if not samples:
        samples = loadJSON('./data/test_out.json')
    evaluateStats(samples)
    reportErrors(samples)
