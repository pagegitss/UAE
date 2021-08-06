"""Evaluate estimators (Naru or others) on queries."""
import argparse
import collections
import glob
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import torch
import json
import common
import datasets
import estimators as estimators_lib
import made

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device', DEVICE)

parser = argparse.ArgumentParser()

parser.add_argument('--inference-opts',
                    action='store_true',
                    help='Tracing optimization for better latency.')

parser.add_argument('--random-workload',  action='store_true',  help='is random workload file?')
parser.add_argument('--dataset', type=str, default='dmv', help='Dataset.')

parser.add_argument('--err-csv',
                    type=str,
                    default='results.csv',
                    help='Save result csv to what path?')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
parser.add_argument('--psample',
                    type=int,
                    default=1000,
                    help='# of progressive samples to use per query.')
parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Turn on wildcard skipping.  Requires checkpoints be trained with '\
    'column masking.')
parser.add_argument('--order',
                    nargs='+',
                    type=int,
                    help='Use a specific order?')


# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=128,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=2, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
parser.add_argument(
    '--inv-order',
    action='store_true',
    help='Set this flag iff using MADE and specifying --order. Flag --order'\
    'lists natural indices, e.g., [0 2 1] means variable 2 appears second.'\
    'MADE, however, is implemented to take in an argument the inverse '\
    'semantics (element i indicates the position of variable i).  Transformer'\
    ' does not have this issue and thus should not have this flag on.')
parser.add_argument(
    '--input-encoding',
    type=str,
    default='binary',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='one_hot',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
    'then input encoding should be set to embed as well.')


args = parser.parse_args()


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def MakeTable():
    assert args.dataset in ['dmv', 'census', 'cup98']
    if args.dataset == 'dmv':
        table = datasets.LoadDmv()
    elif args.dataset == 'census':
        table = datasets.LoadCensus()
    elif args.dataset == 'cup98':
        table = datasets.LoadCup98()

    oracle_est = estimators_lib.Oracle(table)

    return table, None, oracle_est


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)

def Query(estimators,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None):
    assert query is not None
    cols, ops, vals = query

    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # Actual.
    card = oracle_est.Query(cols, ops,
                            vals) if oracle_card is None else oracle_card
    # if card == 0:
    #     return

    pprint('Q(', end='')
    for c, o, v in zip(cols, ops, vals):
        pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
    pprint('): ', end='')

    pprint('\n  actual {} ({:.3f}%) '.format(card,
                                             card / table.cardinality * 100),
           end='')

    for est in estimators:
        est_card = est.Query(cols, ops, vals)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
    pprint()

def QueryTwosided(estimators,
              do_print=True,
              oracle_card=None,
              query=None,
              table=None,
              oracle_est=None):
    assert query is not None
    wildcard_indicator, valid_i_list = query

    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # Actual.
    card = oracle_card
    # if card == 0:
    #     return

    pprint('\n  actual {} ({:.3f}%) '.format(card,
                                             card / table.cardinality * 100),
           end='')

    for est in estimators:
        est_card = est.QueryTwosided(len(wildcard_indicator), wildcard_indicator, valid_i_list)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
    pprint()


def ReportEsts(estimators):
    v = -1
    for est in estimators:
        print(est.name, 'max', np.max(est.errs), '99th',
              np.quantile(est.errs, 0.99), '95th', np.quantile(est.errs, 0.95),
              'median', np.quantile(est.errs, 0.5), 'mean', np.mean(est.errs))
        v = max(v, np.max(est.errs))
    return v


def RunNwithQueries(table,
         estimators,
         query_list,
         rng=None,
         log_every=50,
         oracle_cards=None,
         oracle_est=None):
    if rng is None:
        rng = np.random.RandomState(1234)


    last_time = None
    num = len(query_list)

    columns_list = []
    operators_list = []
    vals_list = []
    for query in query_list:
        cols = query[0]
        ops = query[1]
        vals = query[2]
        columns_list.append(cols)
        operators_list.append(ops)
        vals_list.append(vals)
    wildcard_indicator, valid_i_list = estimators[0].ProcessQuery(args.dataset,
                                                                  columns_list,
                                                                  operators_list,
                                                                  vals_list)
    valid_i_list = np.array(valid_i_list)
    for i in range(num):
        do_print = False
        if i % log_every == 0:
            if last_time is not None:
                print('{:.1f} queries/sec'.format(log_every /
                                                  (time.time() - last_time)))
            do_print = True
            print('Query {}:'.format(i), end=' ')
            last_time = time.time()

        QueryTwosided(estimators,
                    do_print,
                    oracle_card=oracle_cards[i]
                    if oracle_cards is not None and i < len(oracle_cards) else None,
                    query=(wildcard_indicator[i], valid_i_list[i]),
                    table=table,
                    oracle_est=oracle_est)
        max_err = ReportEsts(estimators)
    return False




def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results = pd.DataFrame()
    for est in estimators:
        data = {
            'est': [est.name] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        results = results.append(pd.DataFrame(data))
    if return_df:
        return results
    results.to_csv(path, index=False)



def Main():

    all_ckpts = glob.glob('./models/{}'.format(args.glob))



    if args.blacklist:
        all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]

    selected_ckpts = all_ckpts

    if not args.random_workload:
        file_str = "test_queries/" + args.dataset + "-test-in.txt"
    else:
        file_str = "test_queries/" + args.dataset + "-test-random.txt"


    with open(file_str, 'r', encoding="utf8") as f:
        workload_stats = json.load(f)
    card_list = workload_stats['card_list']
    query_list = workload_stats['query_list']
    oracle_cards = [float(card) for card in card_list]

    print('ckpts', selected_ckpts)

    # OK to load tables now
    table, train_data, oracle_est = MakeTable()
    cols_to_train = table.columns

    Ckpt = collections.namedtuple(
        'Ckpt', 'epoch path loaded_model seed')
    parsed_ckpts = []

    for s in selected_ckpts:
        order = None
        if args.order is not None:
            order = list(args.order)

        model = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=0,
            fixed_ordering=order,
        )

        assert order is None or len(order) == model.nin, order
        ReportModel(model)
        print('Loading ckpt:', s)
        model.load_state_dict(torch.load(s, map_location='cuda:1'))
        model.eval()


        parsed_ckpts.append(
            Ckpt(path=s,
                 epoch=None,
                 loaded_model=model,
                 seed=0))

    # Estimators to run.
    estimators = [
        estimators_lib.ProgressiveSampling(c.loaded_model,
                                           table,
                                           args.psample,
                                           device=DEVICE,
                                           shortcircuit=args.column_masking)
        for c in parsed_ckpts
    ]
    for est, ckpt in zip(estimators, parsed_ckpts):
        est.name = str(est) + '_{}'.format(ckpt.seed)

    if args.inference_opts:
        print('Tracing forward_with_encoded_input()...')
        for est in estimators:
            encoded_input = est.model.EncodeInput(
                torch.zeros(args.psample, est.model.nin, device=DEVICE))

            # NOTE: this line works with torch 1.0.1.post2 (but not 1.2).
            # The 1.2 version changes the API to
            # torch.jit.script(est.model) and requires an annotation --
            # which was found to be slower.
            est.traced_fwd = torch.jit.trace(
                est.model.forward_with_encoded_input, encoded_input)

    if len(estimators):
        RunNwithQueries(table,
                        estimators,
                        query_list,
                        rng=np.random.RandomState(1234),
                        log_every=50,
                        oracle_cards=oracle_cards,
                        oracle_est=oracle_est)
    err_csv = 'result_' + args.dataset + str(args.psample) + '.csv'
    SaveEstimators(err_csv, estimators)
    print('...Done, result:', err_csv)


if __name__ == '__main__':

    Main()
