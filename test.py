''' Translate input text with trained model '''
import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import pickle

from utils.dataset import paired_collate_fn, TranslationDataset
from seq2seq import Constants
from seq2seq.Translator import Translator


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True, help='Path to model .chkpt file')
    parser.add_argument('-test_file', required=True, help='Test pickle file for validation')
    parser.add_argument('-output', default='outputs.txt', help='Path to output the predictions (each line will be the decoded sequence')
    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-n_best', type=int, default=1, help='If verbose is set, will output the n_best decoded sentences')
    parser.add_argument('-max_order', type=int, default=4, help='Maximum n-gram order for calculating BLEU score')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    #- Prepare Translator
    translator = Translator(opt)
    print('[Info] Model opts: {}'.format(translator.model_opt))

    #- Prepare DataLoader
    test_data = torch.load(opt.test_file)
    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=test_data['dict']['src'],
            tgt_word2idx=test_data['dict']['tgt'],
            src_insts=test_data['test']['src'],
            tgt_insts=test_data['test']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        drop_last=True,
        collate_fn=paired_collate_fn)

    #- Evaluate and calculate BLEU score
    references = []
    predictions = []

    print('[Info] Evaluate on test set.')
    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Testing)', leave=False):
            src_seq, src_pos, tgt_seq, _ = batch
            all_hyp, all_scores = translator.translate_batch(src_seq, src_pos) # structure: List[batch, seq, pos]
            for pred_inst, gold_inst in zip(all_hyp, tgt_seq):
                for pred_seq, gold_seq in zip(pred_inst, gold_inst):
                    pred_seq = pred_seq[0]
                    gold_seq = gold_seq[1:-1].masked_select(gold_seq.ne(Constants.PAD))
                    gold_seq = [i.item() for i in gold_seq]
                    predictions.append(pred_seq)
                    references.append(gold_seq)
                    pred_seq = ' '.join([test_loader.dataset.tgt_idx2word[word] for word in pred_seq])
                    f.write(pred_seq + '\n')
                f.write('-\n')

    print('[Info] Calculate BLEU score.')
    bleu, precisions, bp, ratio = cal_bleu_score(predictions, references, max_order=opt.max_order)
    print('Test set BLEU score: {}'.format(bleu))

if __name__ == "__main__":
    main()
