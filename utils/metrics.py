''' This file contains all functions used to calculate evaluation metrics '''
import torch
import torch.nn.functional as F
import collections

import seq2seq.Constants as Constants


def cal_performance(pred, gold, smoothing=False, mmi_factor=1.0):
    '''
    Calculate accuracy and loss with
    1) label smoothing if specified
    2) maximal mutual information (MMI) if specified
    '''
    if mmi_factor > 0:
        #- Calculate CE loss with MMI objective
        pred_session, pred_no_session = torch.split(pred, int(pred.shape[0]/2), dim=0)
        loss = cal_mmi_loss(pred_session, pred_no_session, gold, smoothing=smoothing, mmi_factor=mmi_factor)
        pred = (pred_session - pred_no_session).max(1)[1]
    else:
        #- Calculate CE loss with MLE objective
        loss = cal_mle_loss(pred, gold, smoothing)
        pred = pred.max(1)[1]
    
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct
    
def cal_mmi_loss(pred_session, pred_no_session, gold, smoothing=True, mmi_factor=1.0):
    '''
    Calculate MMI objective, apply label smoothing if needed

    MMI objective:
        r* = argmax_r {log P(r|r_) - lamb * log P(r)}
    where r is the session-infused response,
          r_ is the session-dry response,
          lamb is the weighting factor (lamb=0.0 is MLE)
    '''
    gold = gold.contiguous().view(-1)
    one_hot = torch.zeros_like(pred_session).scatter(1, gold.view(-1, 1), 1)

    if smoothing:
        eps = 0.1
        n_class = pred_session.size(1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

    ses_output_sftmx = F.log_softmax(pred_session, dim=1)
    no_ses_outout_sftmx = F.log_softmax(pred_no_session, dim=1)
    final_sftmax = ses_output_sftmx - mmi_factor * no_ses_outout_sftmx

    non_pad_mask = gold.ne(Constants.PAD)
    loss = -(one_hot * final_sftmax).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).sum()

    return loss

def cal_mle_loss(pred, gold, smoothing):
    '''
    Calculate cross entropy loss, apply label smoothing if needed
    
    MLE objective:
        r* = argmax_r {log P(r|r_)}
    where r is the session-infused response,
          r_ is the session-dry response
    '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

def cal_bleu_score(pred, gold, max_order=4):
    ''' Calculates n-gram matches per batch '''
    def _get_ngrams(segment, n):
        ''' Extracts all n-grams up to a given maximum order from an input segment '''
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    #- Set up stats tracking
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    prediction_length = 0

    assert len(pred) == len(gold)

    #- Loop through elements in corpus
    for prediction, reference in zip(pred, gold):
        reference_length += len(reference)
        prediction_length += len(prediction)

        #- Collect n-grams
        reference_ngram_counts = _get_ngrams(reference, max_order)
        prediction_ngram_counts = _get_ngrams(prediction, max_order)
        overlap = prediction_ngram_counts & reference_ngram_counts

        #- Accumulate exact n-gram matches
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]

        #- Generate total possible n-gram matches
        for order in range(1, max_order+1):
            possible_matches = len(prediction) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    #- Calculate precision
    precisions = [0] * max_order
    for i in range(max_order):
        if smoothing:
            precisions[i] = (matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0

    #- Calculate geometric mean of precisions (log-average)
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    #- Calcule brevity penalty
    ratio = float(prediction_length) / reference_length
    if ratio >= 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return bleu, precisions, bp, ratio
