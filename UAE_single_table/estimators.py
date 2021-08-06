"""A suite of cardinality estimators.

In practicular, inference algorithms for autoregressive density estimators can
be found in 'ProgressiveSampling'.

Also, Differentiable Progressive Sampling can be found in BatchProgressiveSampling
"""
import bisect
import collections
import json
import operator
import time
import copy

import numpy as np
import pandas as pd
import torch

import common
import made

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "mean", np.mean(est.errs), "time_ms",
              np.mean(est.query_dur_ms))

def FillInUnqueriedColumns(table, columns, operators, vals, column_is_idx=False):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        if column_is_idx is False:
            idx = table.ColumnIndex(c.name)
        else:
            idx = c
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


def FillInUnqueriedColumnsTwosided(table, columns, operators, vals, column_type=0):
    """Allows for some columns to be unqueried (i.e., wildcard).

    column_type:
                0: Column object
                1: Column index
                2: Column name

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        if column_type == 0:
            idx = table.ColumnIndex(c.name)
        elif column_type == 1:
            idx = c
        else:
            idx = table.ColumnIndex(c)

        if os[idx] == None:
            os[idx] = [o]
        else:
            os[idx].append(o)
        if vs[idx] == None:
            vs[idx] = [v]
        else:
            vs[idx].append(v)

    return cs, os, vs

class ProgressiveSampling(CardEst):
    """Progressive sampling."""

    def __init__(
            self,
            model,
            table,
            r,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False,  # Skip sampling on wildcards?
            is_training=False
    ):
        super(ProgressiveSampling, self).__init__()
        torch.set_grad_enabled(False)
        self.model = model
        self.table = table
        self.shortcircuit = shortcircuit
        self.is_training = is_training

        if r <= 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(1, self.model.nin, device=device))


        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        if 'MADE' in str(model):
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        print('Setting masked_weight in MADE, do not retrain!')
        for p in model.parameters():
            p.detach_()
            p.requires_grad = False
        self.init_logits.detach_()

        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros)

            self.inp = self.inp.view(self.num_samples, -1)


    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        return 'psample_{}'.format(n)

    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  inp=None):
        ncols = len(columns)
        logits = self.init_logits
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []

        # Use the query to filter each column's domain.
        valid_i_list = [None] * ncols  # None means all valid.
        for i in range(ncols):
            natural_idx = ordering[i]

            # Column i.
            op = operators[natural_idx]
            if op is not None:
                # There exists a filter.
                valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                  vals[natural_idx]).astype(np.float32,
                                                            copy=False)
            else:
                continue

            # This line triggers a host -> gpu copy, showing up as a
            # hotspot in cprofile.
            valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)

        # Fill in wildcards, if enabled.
        if self.shortcircuit:
            for i in range(ncols):
                natural_idx = i if ordering is None else ordering[i]

                if operators[natural_idx] is None and natural_idx != ncols - 1:
                    if natural_idx == 0:
                        self.model.EncodeInput(
                            None,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]],
                            is_onehot=False)
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                                 1]
                        r = self.model.input_bins_encoded_cumsum[natural_idx]
                        self.model.EncodeInput(None,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r],
                                               is_onehot=False)

        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]

            # If wildcard enabled, 'logits' wasn't assigned last iter.
            if not self.shortcircuit or operators[natural_idx] is not None:
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1)

                valid_i = valid_i_list[i]
                if valid_i is not None:
                    probs_i *= valid_i

                probs_i_summed = probs_i.sum(1)

                masked_probs.append(probs_i_summed)

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                probs_i = probs_i.masked_fill_(paths_vanished, 1.0)

            if i < ncols - 1:
                # Num samples to draw for column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples if num_samples else int(
                        self.r * self.dom_sizes[natural_idx])

                if self.shortcircuit and operators[natural_idx] is None:
                    data_to_encode = None
                else:
                    samples_i = torch.multinomial(
                        probs_i, num_samples=num_i,
                        replacement=True)  # [bs, num_i]

                    data_to_encode = samples_i.view(-1, 1) #[batch size, 1]

                # Encode input: i.e., put sampled vars into input buffer.
                if data_to_encode is not None:  # Wildcards are encoded already.

                    if natural_idx == 0:
                        self.model.EncodeInput(
                            data_to_encode,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]],
                            is_onehot=False)
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx
                                                                 - 1]
                        r = self.model.input_bins_encoded_cumsum[
                            natural_idx]
                        self.model.EncodeInput(data_to_encode,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r],
                                               is_onehot=False)

                # Actual forward pass.
                next_natural_idx = i + 1 if ordering is None else ordering[i +
                                                                           1]
                if self.shortcircuit and operators[next_natural_idx] is None:
                    # If next variable in line is wildcard, then don't do
                    # this forward pass.  Var 'logits' won't be accessed.
                    continue

                if hasattr(self.model, 'do_forward'):
                    # With a specific ordering.
                    logits = self.model.do_forward(inp, ordering)
                else:
                    if self.traced_fwd is not None:
                        logits = self.traced_fwd(inp)
                    else:
                        logits = self.model.forward_with_encoded_input(inp)

        # Doing this convoluted scheme because m_p[0] is a scalar, and
        # we want the corret shape to broadcast.
        p = masked_probs[1]
        for ls in masked_probs[2:]:
            p *= ls
        p *= masked_probs[0]

        return p.mean().item()

    def _sample_n_twosided(self,
                  num_samples,
                  ordering,
                  wildcard_indicator,
                  valid_i_list,
                  inp=None):
        ncols = len(wildcard_indicator)
        logits = self.init_logits
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []

        #model_feature_nout = None

        # Fill in wildcards, if enabled.
        if self.shortcircuit:
            for i in range(ncols):
                natural_idx = i if ordering is None else ordering[i]

                if wildcard_indicator[natural_idx] is None and natural_idx != ncols - 1:
                    if natural_idx == 0:
                        self.model.EncodeInput(
                            None,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]],
                            is_onehot=False)
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                                 1]
                        r = self.model.input_bins_encoded_cumsum[natural_idx]
                        self.model.EncodeInput(None,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r],
                                               is_onehot=False)

        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]


            # If wildcard enabled, 'logits' wasn't assigned last iter.
            if not self.shortcircuit or wildcard_indicator[natural_idx] is not None:
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1)

                valid_i = valid_i_list[i]
                t_valid_i = torch.as_tensor(valid_i, dtype=torch.float32, device=self.device)
                if valid_i is not None:
                    probs_i *= t_valid_i

                probs_i_summed = probs_i.sum(1)

                masked_probs.append(probs_i_summed)

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                probs_i = probs_i.masked_fill_(paths_vanished, 1.0)

            if i < ncols - 1:
                # Num samples to draw for column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples if num_samples else int(
                        self.r * self.dom_sizes[natural_idx])

                if self.shortcircuit and wildcard_indicator[natural_idx] is None:
                    data_to_encode = None
                else:
                    samples_i = torch.multinomial(
                        probs_i, num_samples=num_i,
                        replacement=True)  # [bs, num_i]

                    data_to_encode = samples_i.view(-1, 1) #[batch size, 1]

                # Encode input: i.e., put sampled vars into input buffer.
                if data_to_encode is not None:  # Wildcards are encoded already.
                    if natural_idx == 0:
                        self.model.EncodeInput(
                            data_to_encode,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]],
                            is_onehot=False)
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx
                                                                 - 1]
                        r = self.model.input_bins_encoded_cumsum[
                            natural_idx]
                        self.model.EncodeInput(data_to_encode,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r],
                                               is_onehot=False)

                # Actual forward pass.
                next_natural_idx = i + 1 if ordering is None else ordering[i +
                                                                           1]
                if self.shortcircuit and wildcard_indicator[next_natural_idx] is None:
                    # If next variable in line is wildcard, then don't do
                    # this forward pass.  Var 'logits' won't be accessed.
                    continue

                if hasattr(self.model, 'do_forward'):
                    # With a specific ordering.
                    logits = self.model.do_forward(inp, ordering)
                else:
                    if self.traced_fwd is not None:
                        logits = self.traced_fwd(inp)
                    else:
                        logits = self.model.forward_with_encoded_input(inp)

        # Doing this convoluted scheme because m_p[0] is a scalar, and
        # we want the corret shape to broadcast.
        p = masked_probs[1]
        for ls in masked_probs[2:]:
            p *= ls
        p *= masked_probs[0]

        return p.mean().item()

    def Query(self, columns, operators, vals):
        # Massages queries into natural order.
        columns, operators, vals = FillInUnqueriedColumns(
            self.table, columns, operators, vals)

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * len(columns)
        for natural_idx in range(len(columns)):
            inv_ordering[ordering[natural_idx]] = natural_idx


        with torch.no_grad():
            inp_buf = self.inp.zero_()
            # Fast (?) path.
            if num_orderings == 1:
                ordering = orderings[0]
                self.OnStart()
                p = self._sample_n(
                    self.num_samples,
                    inv_ordering,
                    columns,
                    operators,
                    vals,
                    inp=inp_buf)
                self.OnEnd()
                return np.ceil(p * self.cardinality).astype(dtype=np.int32,
                                                            copy=False)

            # Num orderings > 1.
            ps = []
            self.OnStart()
            for ordering in orderings:
                p_scalar = self._sample_n(self.num_samples // num_orderings,
                                          ordering, columns, operators, vals)
                ps.append(p_scalar)
            self.OnEnd()
            return np.ceil(np.mean(ps) * self.cardinality).astype(
                dtype=np.int32, copy=False)


            # Num orderings > 1.
            ps = []
            self.OnStart()
            for ordering in orderings:
                p_scalar = self._sample_n(self.num_samples // num_orderings,
                                          ordering, columns, operators, vals)
                ps.append(p_scalar)
            self.OnEnd()
            return np.ceil(np.mean(ps) * self.cardinality).astype(
                dtype=np.int32, copy=False)

    def QueryTwosided(self,  n_cols, wildcard_indicator, valid_i_list):
        # Massages queries into natural order.

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(n_cols)
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * n_cols
        for natural_idx in range(n_cols):
            inv_ordering[ordering[natural_idx]] = natural_idx


        with torch.no_grad():
            inp_buf = self.inp.zero_()
            # Fast (?) path.
            if num_orderings == 1:
                ordering = orderings[0]
                self.OnStart()
                p = self._sample_n_twosided(
                    self.num_samples,
                    inv_ordering,
                    wildcard_indicator,
                    valid_i_list,
                    inp=inp_buf)
                self.OnEnd()
                return np.ceil(p * self.cardinality).astype(dtype=np.int32,
                                                            copy=False)

            # Num orderings > 1.
            ps = []

            self.OnStart()
            for ordering in orderings:
                p_scalar = self._sample_n_twosided(self.num_samples // num_orderings,
                                          ordering, wildcard_indicator, valid_i_list)
                ps.append(p_scalar)
            self.OnEnd()
            return np.ceil(np.mean(ps) * self.cardinality).astype(
                dtype=np.int32, copy=False)


    def ProcessQuery(self, data_name, columns_list, operators_list, vals_list):
        """Proprocess the queries.
            columns_list: list of [query_number, op_number].
            operators_list: list of [query_number, op_number].
            vals_list: list of [query_number, op_number].

        return:
            wildcard_indicator:  [query_number, column_number]
            res_list: [query_number, column_number, column_size of that column]
        """

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(self.table.columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * len(self.table.columns)
        for natural_idx in range(len(self.table.columns)):
            inv_ordering[ordering[natural_idx]] = natural_idx

        res_list = []
        query_number = len(operators_list)
        ncols = len(self.table.columns)
        wildcard_indicator = []
        for q_i in range(query_number):
            wildcard_indicator_q = [None] * ncols
            valid_i_list = [None] * ncols  # None means all valid.
            cols = columns_list[q_i]
            ops = operators_list[q_i]
            vals = vals_list[q_i]
            columns, ops, vals = FillInUnqueriedColumnsTwosided(
                self.table, cols, ops, vals, 2)
            for c_i in range(ncols):
                res_valid_i = None
                natural_idx = ordering[c_i]
                # Column i.
                op_list = ops[natural_idx]
                val_list = vals[natural_idx]
                if op_list is not None:
                    # There exists a filter or several filters.
                    wildcard_indicator_q[c_i] = 1
                    op_number = len(op_list)
                    for o_i in range(op_number):
                        op = op_list[o_i]
                        val = val_list[o_i]
                        if type(val) == float or type(val) == int:
                            if not np.isnan(val):
                                if data_name == 'dmv':
                                    if natural_idx == 6:
                                        val = np.datetime64(val)
                                else:
                                    if isinstance(type(val), type(columns[natural_idx].all_distinct_values[-1])) is False:
                                        val = type(columns[natural_idx].all_distinct_values[-1])(val)

                                valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                                  val)
                            else:
                                valid_i = [False] * len(columns[natural_idx].all_distinct_values)
                                valid_i = np.array(valid_i)
                        else:
                            if data_name == 'dmv':
                                if natural_idx == 6:
                                    val = np.datetime64(val)
                            else:
                                if isinstance(type(val), type(columns[natural_idx].all_distinct_values[-1])) is False:
                                    val = type(columns[natural_idx].all_distinct_values[-1])(val)

                            first_v = columns[natural_idx].all_distinct_values[0]
                            if type(first_v)==float or type(first_v)==int:
                                if np.isnan(first_v):
                                    valid_i = OPS[op](columns[natural_idx].all_distinct_values[1:],
                                                      val)
                                    valid_i = np.insert(valid_i, 0, False)
                                else:
                                    valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                                      val)
                            else:
                                valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                                  val)
                        if res_valid_i is None:
                            res_valid_i = valid_i
                        else:
                            res_valid_i &= valid_i
                else:
                    res_valid_i = [True] * len(columns[natural_idx].all_distinct_values)
                    res_valid_i = np.array(res_valid_i)
                valid_i_list[c_i] = res_valid_i.astype(np.float32, copy=False)
            res_list.append(valid_i_list)
            wildcard_indicator.append(wildcard_indicator_q)
        return wildcard_indicator, res_list

class DifferentiableProgressiveSampling(CardEst):
    """ Differentiable Progressive sampling."""

    def __init__(
            self,
            model,
            table,
            r,
            batch_size=200,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False,  # Skip sampling on wildcards?
            is_training=True,
            tau=1.
    ):
        super(DifferentiableProgressiveSampling, self).__init__()
        self.model = model
        self.table = table
        self.shortcircuit = shortcircuit
        self.batch_size = batch_size
        self.is_training = is_training
        self.tau = tau
        if r <= 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        if self.is_training is False:
            with torch.no_grad():
                self.init_logits = self.model(
                    torch.zeros(self.batch_size, self.model.nin, device=device))
        else:
            self.init_logits = self.model(
                torch.zeros(self.batch_size, self.model.nin, device=device))

        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput



        if is_training is False:
            if 'MADE' in str(model):
                for layer in model.net:
                    if type(layer) == made.MaskedLinear:
                        if layer.masked_weight is None:
                            layer.masked_weight = layer.mask * layer.weight
                            print('Setting masked_weight in MADE, do not retrain!')
            for p in model.parameters():
                p.detach_()
                p.requires_grad = False
            self.init_logits.detach_()

        if self.is_training is False:
            with torch.no_grad():
                self.kZeros = torch.zeros(self.batch_size,
                                          self.num_samples,
                                          self.model.nin,
                                          device=self.device)
                self.inp = self.traced_encode_input(self.kZeros.view(-1,self.model.nin))

                self.inp = self.inp.view(self.batch_size, self.num_samples, -1)
        else:
            self.kZeros = torch.zeros(self.batch_size,
                                      self.num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros.view(-1,self.model.nin))

            self.inp = self.inp.view(self.batch_size, self.num_samples, -1)

    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        return 'psample_{}'.format(n)

    def _sample_n(self,
                  num_samples,
                  ordering,
                  n_cols,
                  wildcard_indicator=None,
                  valid_i_list=None,  # batch_size list of [column_number, size_of_each_column],
                  inp=None):
        '''
            wildcard_indicator: [batch_size, column_number] as torch.tensor.
            valid_i_list: batch_size list of [column_number, size_of_each_column], should not contains None.
        '''

        ncols =  n_cols
        logits = self.init_logits
        if inp is None:
            inp = self.inp.view(self.batch_size*self.num_samples, -1)
        else:
            inp = inp.view(self.batch_size * self.num_samples, -1)
        masked_probs = []

        bs = wildcard_indicator.shape[0]
        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits

        inp_list = [None] * ncols

        for i in range(ncols):
            # logits: [bs, domain_size]
            natural_idx = i if ordering is None else ordering[i]

            # If wildcard enabled, 'logits' wasn't assigned last iter.
            if not self.shortcircuit:
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1) # [bs, col_colsize] if i==0;
                                                                       # [bs*num_samples, col_colsize] if i>0;

                valid_i = valid_i_list[:, i]
                valid_i = np.array([np.array(xi) for xi in valid_i]) # [bs, col_colsize]
                if i > 0:
                    valid_i = np.repeat(valid_i, self.num_samples, axis=0) # [bs*num_samples, col_colsize]
                t_valid_i = torch.as_tensor(valid_i, dtype=torch.float32, device=self.device)  # [bs, col_size]
                probs_i = probs_i * t_valid_i

                probs_i_summed = probs_i.sum(1)
                masked_probs.append(probs_i_summed)
            else:
                unnorm_prob_i = self.model.logits_for_col(natural_idx, logits, is_training=True)
                probs_i = torch.softmax(unnorm_prob_i, dim=-1)

                valid_i = valid_i_list[:, i]
                valid_i = np.array([np.array(xi) for xi in valid_i])
                if i > 0:
                    valid_i = np.repeat(valid_i, self.num_samples, axis=0) # [bs*num_samples, col_colsize]
                t_valid_i = torch.as_tensor(valid_i, dtype=torch.float32, device=self.device) # [bs, col_size]

                probs_i = probs_i*t_valid_i

                probs_i_summed = probs_i.sum(1)
                if i == 0:
                    ones_probs_i_summed = torch.ones(bs, device=self.device)
                    probs_i_summed = torch.where(wildcard_indicator[:, i] == 0, ones_probs_i_summed, probs_i_summed)
                    probs_i_summed = probs_i_summed.view(-1, 1) # [bs, 1]
                else:
                    probs_i_summed = probs_i_summed.view(bs, -1) # [bs, num_samples]
                    ones_probs_i_summed = torch.ones(bs, probs_i_summed.shape[1], device=self.device)
                    indicator_i = wildcard_indicator[:, i].view(-1, 1) # [bs, 1]
                    # indicator_i.repeat(1, num_samples)
                    probs_i_summed = torch.where(indicator_i == 0, ones_probs_i_summed, probs_i_summed) # [bs, num_samples]
                masked_probs.append(probs_i_summed)

            if i < ncols - 1:
                # Num samples to draw for column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples if num_samples else int(
                        self.r * self.dom_sizes[natural_idx])

                if self.is_training is False:
                    samples_i = torch.multinomial(
                        probs_i, num_samples=num_i,
                        replacement=True)  # [bs, num_i]

                else:
                    logits_i = self.model.logits_for_col(natural_idx, logits)# [bs, col_colsize] if i==0;
                                                                       # [bs*num_samples, col_colsize] if i>0;

                    tensor_valid_i = torch.as_tensor(valid_i, dtype=torch.uint8, device=self.device)
                    inf_tensor = [float('-inf')] * logits_i.shape[1]
                    inf_tensor = torch.as_tensor(inf_tensor, dtype=torch.float32, device=self.device)
                    logits_i = torch.where(tensor_valid_i==True, logits_i, inf_tensor)


                    if i == 0:
                        if torch.__version__ == '1.0.1':
                            logits_i = logits_i.transpose(0, 1).repeat(self.num_samples, 1).transpose(0, 1) \
                                .reshape(-1, logits_i.shape[-1])
                        else:
                            logits_i = torch.repeat_interleave(logits_i, self.num_samples, dim=0)

                    one_hot_samples_i = torch.nn.functional.gumbel_softmax(logits_i, tau=self.tau, hard=True)


                if self.shortcircuit:
                    if natural_idx == 0:
                        wildcard_inp_candidate = self.model.EncodeInput(
                            None,
                            natural_col=0,
                            is_onehot=self.is_training) # [1, encode_size of i]

                        #wildcard_inp_candidate.repeat(self.batch_size * self.num_samples, 1)

                        normal_inp_candidate = self.model.EncodeInput(
                                one_hot_samples_i,
                                natural_col=0,
                                is_onehot=self.is_training) # [bs*num_sample, encode_size of i]
                        indicator_i = wildcard_indicator[:, i].view(-1, 1) # [bs, 1]

                        if torch.__version__ == '1.0.1':
                            indicator_i = indicator_i.transpose(0, 1).repeat(self.num_samples, 1).transpose(0, 1)\
                                            .reshape(-1, indicator_i.shape[-1])
                        else:
                            indicator_i = torch.repeat_interleave(indicator_i, self.num_samples,
                                                                      dim=0)

                        normal_candidate = torch.where(indicator_i == 0, wildcard_inp_candidate,
                                                       normal_inp_candidate)
                        inp_list[natural_idx] = normal_candidate
                    else:
                        normal_inp_candidate = self.model.EncodeInput(one_hot_samples_i,
                                               natural_col=natural_idx,
                                               is_onehot=self.is_training)
                        wildcard_inp_candidate = self.model.EncodeInput(
                            None,
                            natural_col=natural_idx,
                            is_onehot=self.is_training)
                        indicator_i = wildcard_indicator[:, i].view(-1, 1)  # [bs, 1]

                        if torch.__version__ == '1.0.1':
                            indicator_i = indicator_i.transpose(0, 1).repeat(self.num_samples, 1).transpose(0, 1)\
                                            .reshape(-1, indicator_i.shape[-1])
                        else:
                            indicator_i = torch.repeat_interleave(indicator_i, self.num_samples,
                                                                      dim=0)
                        normal_candidate = torch.where(indicator_i == 0, wildcard_inp_candidate,
                                                       normal_inp_candidate)
                        inp_list[natural_idx] = normal_candidate
                else:
                    # No variable skipping version has not implemented.
                    # Interested readers might implement it simply based on the variable skipping version
                    raise NotImplementedError

                # Actual forward pass.
                inp_res = []
                count_id = 0
                for col_inp in inp_list:
                    if col_inp is None:
                        if count_id == 0:
                            inp_res.append(inp[:, :self.model.input_bins_encoded_cumsum[0]])
                        else:
                            l = self.model.input_bins_encoded_cumsum[count_id
                                                                     - 1]
                            r = self.model.input_bins_encoded_cumsum[
                                count_id]
                            inp_res.append(inp[:, l:r])
                    else:
                        inp_res.append(col_inp)
                    count_id += 1

                inp_res = torch.cat(inp_res, dim=1)

                if hasattr(self.model, 'do_forward'):
                    # With a specific ordering.
                    logits = self.model.do_forward(inp_res, ordering)
                else:
                    if self.traced_fwd is not None:
                        logits = self.traced_fwd(inp_res)
                    else:
                        logits = self.model.forward_with_encoded_input(inp_res)

        # Doing this convoluted scheme because m_p[0] is a scalar, and
        # we want the correct shape to broadcast.
        p = masked_probs[1]  # [bs, num_samples]
        for ls in masked_probs[2:]:
            p *= ls

        p_0 = masked_probs[0] # [bs, 1]
        p *= p_0

        return p.mean(dim=1)  # [batch_size], [batch_size, -1]

    def BatchQuery(self, n_cols, wildcard_indicator, valid_i_list):
        # Massages queries into natural order.

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(n_cols)
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * n_cols
        for natural_idx in range(n_cols):
            inv_ordering[ordering[natural_idx]] = natural_idx


        # Fast (?) path.
        if num_orderings == 1:
            ordering = orderings[0]
            self.OnStart()
            batch_estimatied_p = self._sample_n(
                self.num_samples,
                inv_ordering,
                n_cols,
                wildcard_indicator,
                valid_i_list,
                )
            self.OnEnd()
            return batch_estimatied_p

        # Num orderings > 1.
        ps = []

        self.OnStart()
        for ordering in orderings:
            batch_estimatied_p = self._sample_n(self.num_samples // num_orderings,
                                    ordering, n_cols, wildcard_indicator, valid_i_list)
            ps.append(batch_estimatied_p)
        self.OnEnd()
        return ps

    def ProcessQuery(self, data_name, columns_list, operators_list, vals_list):
        """Proprocess the queries.
            columns_list: list of [query_number, op_number].
            operators_list: list of [query_number, op_number].
            vals_list: list of [query_number, op_number].

        return:
            wildcard_indicator:  [query_number, column_number]
            res_list: [query_number, column_number, column_size of that column]
        """

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(self.table.columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * len(self.table.columns)
        for natural_idx in range(len(self.table.columns)):
            inv_ordering[ordering[natural_idx]] = natural_idx

        res_list = []
        query_number = len(operators_list)
        ncols = len(self.table.columns)
        wildcard_indicator = np.zeros((query_number, ncols), dtype=np.int32)
        for q_i in range(query_number):
            valid_i_list = [None] * ncols  # None means all valid.
            cols = columns_list[q_i]
            ops = operators_list[q_i]
            vals = vals_list[q_i]
            columns, ops, vals = FillInUnqueriedColumnsTwosided(
                self.table, cols, ops, vals, 2)
            for c_i in range(ncols):
                res_valid_i = None
                natural_idx = ordering[c_i]
                # Column i.
                op_list = ops[natural_idx]
                val_list = vals[natural_idx]
                if op_list is not None:
                    # There exists a filter or several filters.
                    wildcard_indicator[q_i][c_i] = 1
                    op_number = len(op_list)
                    for o_i in range(op_number):
                        op = op_list[o_i]
                        val = val_list[o_i]
                        if type(val) == float or type(val) == int:
                            if not np.isnan(val):
                                if data_name in ['dmv', 'dmv-tiny']:
                                    if natural_idx == 6:
                                        val = np.datetime64(val)
                                else:
                                    if isinstance(type(val),
                                                  type(columns[natural_idx].all_distinct_values[-1])) is False:
                                        val = type(columns[natural_idx].all_distinct_values[-1])(val)

                                valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                                      val)
                            else:
                                valid_i = [False] * len(columns[natural_idx].all_distinct_values)
                                valid_i = np.array(valid_i)
                        else:
                            if data_name in ['dmv', 'dmv-tiny']:
                                if natural_idx == 6:
                                    val = np.datetime64(val)
                            else:
                                if isinstance(type(val), type(columns[natural_idx].all_distinct_values[-1])) is False:
                                    val = type(columns[natural_idx].all_distinct_values[-1])(val)

                            first_v = columns[natural_idx].all_distinct_values[0]
                            if type(first_v) == float or type(first_v) == int:
                                if np.isnan(first_v):
                                    valid_i = OPS[op](columns[natural_idx].all_distinct_values[1:],
                                                      val)
                                    valid_i = np.insert(valid_i, 0, False)
                                else:
                                    valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                                      val)
                            else:
                                valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                                  val)
                        if res_valid_i is None:
                            res_valid_i = valid_i
                        else:
                            res_valid_i &= valid_i
                else:
                    res_valid_i = [True] * len(columns[natural_idx].all_distinct_values)
                    res_valid_i = np.array(res_valid_i)
                valid_i_list[c_i] = res_valid_i.astype(np.float32, copy=False)
            res_list.append(valid_i_list)
            wildcard_indicator = torch.as_tensor(wildcard_indicator, dtype=torch.int32, device=self.device)
        return wildcard_indicator, res_list

class Oracle(CardEst):
    """Returns true cardinalities."""

    def __init__(self, table, limit_first_n=None):
        super(Oracle, self).__init__()
        self.table = table
        self.limit_first_n = limit_first_n

    def __str__(self):
        return 'oracle'

    def Query(self, columns, operators, vals, return_masks=False, return_crad_and_masks=False):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()

        bools = None
        for c, o, v in zip(columns, operators, vals):
            if self.limit_first_n is None:
                if self.table.name in ['Adult', 'Census']: 
                    inds = [False] * self.table.cardinality
                    inds = np.array(inds)
                    is_nan = pd.isnull(c.data)
                    if np.any(is_nan):
                        inds[~is_nan] = OPS[o](c.data[~is_nan], v)
                    else:
                        inds = OPS[o](c.data, v)
                else:
                    inds = OPS[o](c.data, v)
            else:
                # For data shifts experiment.
                inds = OPS[o](c.data[:self.limit_first_n], v)

            if bools is None:
                bools = inds
            else:
                bools &= inds
        c = bools.sum()
        self.OnEnd()
        if return_masks:
            return bools
        elif return_crad_and_masks:
            return c, bools
        return c




