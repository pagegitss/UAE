"""Model training of UAE."""

import argparse
import os
import time
import collections
import glob
import pickle
import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import estimators as estimators_lib
import common
import datasets
import made
import pandas as pd
import json
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device', DEVICE)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda-num',
                    type=int,
                    default=None,
                    help='the number of cuda used')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
# for UAE
parser.add_argument('--psample',
                    type=int,
                    default=200,
                    help='# of differentiable progressive samples to use per query.')
parser.add_argument('--workload-size',
                    type=int,
                    default=20000,
                    help='Number of queries to train for.')
parser.add_argument('--tau',
                    type=float,
                    default=1.0,
                    help='tau in gumbel-softmax.')
parser.add_argument('--q-weight',
                    type=float,
                    default=1e-4,
                    help='weight of the query model.')
parser.add_argument('--fade-in-beta',
                    type=float,
                    default=0.,
                    help='beta of the fade-in process.')
parser.add_argument('--run-uaeq',
                    action='store_true',
                    help='whether train query-driven uae-q?')

# Training.
parser.add_argument('--dataset', type=str, default='dmv', help='Dataset.')
parser.add_argument('--num-gpus', type=int, default=0, help='#gpus.')
parser.add_argument('--bs', type=int, default=1024, help='Batch size of data.')
parser.add_argument('--q-bs', type=int, default=100, help='Batch size of queries. Work when running run-q')
parser.add_argument(
    '--warmups',
    type=int,
    default=0,
    help='Learning rate warmup steps.  Crucial for Transformer.')
parser.add_argument(
    '--data-model-warmups',
    type=int,
    default=0,
    help='warm-up steps for the data-driven model')
parser.add_argument('--epochs',
                    type=int,
                    default=20,
                    help='Number of epochs to train for.')
parser.add_argument('--constant-lr',
                    type=float,
                    default=None,
                    help='Constant LR? this is crucial for training single-table UAE')
parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Column masking training, which permits wildcard skipping'\
    ' at querying time.')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=128,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=4, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
parser.add_argument(
    '--inv-order',
    action='store_true',
    help='Set this flag iff using MADE and specifying --order. Flag --order '\
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


# Ordering.
parser.add_argument('--num-orderings',
                    type=int,
                    default=1,
                    help='Number of orderings.')
parser.add_argument(
    '--order',
    nargs='+',
    type=int,
    required=False,
    help=
    'Use a specific ordering.  '\
    'Format: e.g., [0 2 1] means variable 2 appears second.'
)

args = parser.parse_args()


def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    print(s)
    return ret

def QError(actual_cards, est_cards):
    # [batch_size]
    bacth_ones = torch.ones(actual_cards.shape, dtype=torch.float32, device=DEVICE)
    fixed_actual_cards = torch.where(actual_cards == 0., bacth_ones, actual_cards)
    fixed_est_cards = torch.where(est_cards == 0., bacth_ones, est_cards)

    q_error = torch.where(actual_cards>est_cards, fixed_actual_cards/fixed_est_cards,
                          fixed_est_cards/fixed_actual_cards)

    return q_error

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

def RunEpoch(split,
             model,
             estimator,
             valid_i_list,
             wildcard_indicator,
             card_list,
             opt,
             n_cols,
             train_data,
             val_data=None,
             batch_size=100,
             q_bs=5,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    q_train_idx = list(range(valid_i_list.shape[0]))
    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    np.random.shuffle(q_train_idx)
    for step, (idx, xb) in enumerate(loader):
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                        .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()

        if step * q_bs >= valid_i_list.shape[0]:
            q_idx = q_train_idx[step*q_bs-valid_i_list.shape[0]: (step+1)*q_bs-valid_i_list.shape[0]]
        elif (step+1) * q_bs > valid_i_list.shape[0]:
            q_idx = q_train_idx[step * q_bs:] + q_train_idx[0: (step+1)*q_bs-valid_i_list.shape[0]]
        else:
            q_idx = q_train_idx[step * q_bs: (step+1)*q_bs]

        valid_is = valid_i_list[q_idx]
        batch_wildcard = wildcard_indicator[q_idx]
        actual_cards = card_list[q_idx]


        if valid_is != None:
            est_cards = estimator.BatchQuery(n_cols, batch_wildcard, valid_is)

            if torch.isnan(est_cards).any():
                continue

            q_loss = QError(actual_cards, est_cards).mean()
        else:
            q_loss = 0.

        if args.fade_in_beta > 0.:
            q_weight = args.fade_in_beta * np.exp(-10./(epoch_num+1))
            all_loss = loss + q_weight * q_loss
        else:
            if epoch_num + 1 > args.data_model_warmups:
                q_weight = args.q_weight
                all_loss = loss + q_weight * q_loss
            else:
                q_weight = 0.
                all_loss = loss
        losses.append(all_loss.item())

        if step % log_every == 0:
            if split == 'train':
                print(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (data-model loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, lr))
                print('all loss:' + str(all_loss.item()) + '; q_weight:' + str(q_weight))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            all_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            opt.step()

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))
    if return_losses:
        return losses
    return np.mean(losses)

def RunQueryEpoch(split,
                 model,
                 estimator,
                 valid_i_list,
                 wildcard_indicator,
                 card_list,
                 cardinality,
                 opt,
                 n_cols,
                 batch_size=100,
                 upto=None,
                 epoch_num=None,
                 verbose=False,
                 log_every=10,
                 return_losses=False):

    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()

    losses = []

    q_train_idx = list(range(valid_i_list.shape[0]))
    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    np.random.shuffle(q_train_idx)
    num_batch = int(len(q_train_idx) / batch_size)

    for step in range(num_batch):
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = num_batch * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        if hasattr(model, 'update_masks'):
            # We want to update_masks even for first ever batch.
            model.update_masks()
        start_id = step * batch_size
        end_id = start_id + batch_size
        idx = q_train_idx[start_id:end_id]
        valid_is = valid_i_list[idx]
        batch_wildcard = wildcard_indicator[idx]
        actual_cards = card_list[idx]

        est_sels = estimator.BatchQuery(n_cols, batch_wildcard, valid_is)

        if torch.isnan(est_sels).any():
            continue

        loss = QError(actual_cards, est_sels * cardinality).mean()

        losses.append(loss.item())
        opt.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        opt.step()

        if step % log_every == 0:
            print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

    print('epoch {} mean loss: {}; constant lr {}'.format(epoch_num, np.mean(losses), args.constant_lr))

    if return_losses:
        return losses
    return np.mean(losses)


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


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


def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


def TrainTask(seed=0, rng=None):
    if args.cuda_num is not None:
        torch.cuda.set_device(args.cuda_num)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    tau = args.tau
    torch.manual_seed(0)
    np.random.seed(0)
    if rng is None:
        rng = np.random.RandomState(0)

    if args.dataset == 'dmv':
        table = datasets.LoadDmv()
    elif args.dataset == 'census':
        table = datasets.LoadCensus()
    elif args.dataset == 'cup98':
        table = datasets.LoadCup98()

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns
                                            ]).size(), [2])[0]
    fixed_ordering = None

    if args.order is not None:
        print('Using passed-in order:', args.order)
        fixed_ordering = args.order

    print(table.data.info())

    table_train = table


    if args.dataset in ['dmv', 'census', 'cup98']:
        model = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            fixed_ordering=fixed_ordering,
        )
    else:
        assert False, args.dataset

    mb = ReportModel(model)
    print('Applying InitWeight()')
    model.apply(InitWeight)

    opt = torch.optim.Adam(list(model.parameters()), 2e-4)

    log_every = 200

    train_data = common.TableDataset(table_train)
    n_cols = len(table.columns)
    bs = args.bs

    # load train data

    file_str = './training_queries/{}-train.txt'.format(args.dataset)
    with open(file_str, 'r', encoding="utf8") as f:
        workload_stats = json.load(f)
    tmp_card_list = workload_stats['card_list'][0: args.workload_size]
    query_list = workload_stats['query_list'][0: args.workload_size]

    sel_list = [float(card)/table.cardinality for card in tmp_card_list]

    columns_list = []
    operators_list = []
    vals_list = []
    card_list = []
    i = 0
    for query in query_list:
        if sel_list[i] > 0:
            cols = query[0]
            ops = query[1]
            vals = query[2]
            columns_list.append(cols)
            operators_list.append(ops)
            vals_list.append(vals)
            card_list.append(sel_list[i])
            i += 1
        else:
            i += 1
            continue

    total_query_num = len(card_list)

    if args.run_uaeq:
        q_bs = args.q_bs
    else:
        num_steps = table.cardinality / bs
        q_bs = math.ceil(total_query_num / num_steps)
        q_bs = int(q_bs)

    estimator = estimators_lib.DifferentiableProgressiveSampling(model,
                                                        table,
                                                        args.psample,
                                                        batch_size=q_bs,
                                                        device=DEVICE,
                                                        shortcircuit=args.column_masking,
                                                        tau=tau)


    wildcard_indicator, valid_i_list = estimator.ProcessQuery(args.dataset, columns_list, operators_list, vals_list)

    valid_i_list = np.array(valid_i_list)
    card_list = torch.as_tensor(card_list, dtype=torch.float32)
    card_list = card_list.to(DEVICE)

    train_start = time.time()
    for epoch in range(args.epochs):
        torch.set_grad_enabled(True)
        model.train()

        if not args.run_uaeq:
            mean_epoch_train_loss = RunEpoch('train',
                                             model,
                                             estimator,
                                             valid_i_list,
                                             wildcard_indicator,
                                             card_list,
                                             opt,
                                             n_cols=n_cols,
                                             train_data=train_data,
                                             val_data=train_data,
                                             batch_size=bs,
                                             q_bs=q_bs,
                                             epoch_num=epoch,
                                             log_every=log_every,
                                             table_bits=table_bits)
        else:
            mean_epoch_train_loss = RunQueryEpoch('train',
                                                  model,
                                                  estimator,
                                                  valid_i_list,
                                                  wildcard_indicator,
                                                  card_list,
                                                  table.cardinality,
                                                  opt,
                                                  n_cols=n_cols,
                                                  batch_size=q_bs,
                                                  epoch_num=epoch,
                                                  log_every=10)

        if epoch % log_every == 0:
            print('epoch {} train loss {:.4f} nats'.format(
                epoch, mean_epoch_train_loss))
            since_start = time.time() - train_start
            print('time since start: {:.1f} secs'.format(since_start))

        if not args.run_uaeq:
            PATH = 'models/uae-{}-bs-{}-{}epochs-psample-{}-seed-{}-tau-{}-q-weight-{}-layers-{}.pt'.format(
                args.dataset, bs, epoch, args.psample, seed, tau, args.q_weight, args.layers)
        else:
            PATH = 'models/uaeq-{}-q_bs-{}-{}epochs-psample-{}-seed-{}-tau-{}-layers-{}.pt'.format(
                args.dataset, q_bs, epoch, args.psample, seed, tau,  args.layers)

        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        torch.save(model.state_dict(), PATH)
        print('Saved to:')
        print(PATH)


TrainTask()
