import numpy as np
import tensorflow as tf
from collections import namedtuple
from eval import evaluate
from trans_norm import create_model, _buckets, set_vocab_size
from data_utils import context_window, convert_format, valid_token, \
    build_aspell, sentence_to_token_ids, initialize_vocabulary, EOS_ID
from utilities import loadJSON, saveJSON


def normalize(samples):
    ngram = 3
    sep = ' _S_ '
    aspell = build_aspell()

    # Load seq2seq model
    sess = tf.Session()
    # set_vocab_size('./data/vocab.en', 'en')
    # set_vocab_size('./data/vocab.fr', 'fr')
    model = create_model(sess, True)
    model.batch_size = 1
    en_vocab, _ = initialize_vocabulary('./data/vocab.en')
    _, rev_fr_vocab = initialize_vocabulary('./data/vocab.fr')

    def predict_word(in_win):
        token_ids = sentence_to_token_ids(sep.join(convert_format(win)),
                                          en_vocab)
        bucket_id = min([b for b in xrange(len(_buckets))
                         if _buckets[b][0] > len(token_ids)])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if EOS_ID in outputs:
            outputs = outputs[:outputs.index(EOS_ID)]
        out_token = ''.join([rev_fr_vocab[out] for out in outputs])
        return out_token.replace('_', ' ')

    count = 0
    total, ignored, ruled, predicted = (0, 0, 0, 0)
    print 'total no. of samples : ', len(samples)
    for sample in samples:
        sample['prediction'] = []
        sample['flags'] = []
        for win in context_window(sample['input'], ngram):
            total += 1
            token = win[ngram // 2].lower()
            if not valid_token(token):
                sample['prediction'].append(token)
                sample['flags'].append('ignored')
                ignored += 1
                continue
            if token in aspell:
                sample['prediction'].append(token)
                sample['flags'].append('vocab')
                ignored += 1
            else:
                # Use the seq2seq model to predict the out token
                sample['prediction'].append(predict_word(win))
                sample['flags'].append('predicted')
                predicted += 1
        count += 1
        if count % 100 == 0:
            print 'completed {} samples'.format(count)
    sess.close()
    print 'Total: {}\nIgnored: {}\nRuled: {}\nPredicted: {}'.format(
        total, ignored, ruled, predicted
    )


if __name__ == '__main__':
    samples = loadJSON('./data/test_truth.json')
    normalize(samples)
    saveJSON(samples, './data/test_out_only_oov.json')
    evaluate(samples, './data/norm_errors_only_oov.json')
