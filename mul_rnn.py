
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope as vs


# 重写多层双向循环网络

def stack_bidirectional_dynamic_rnn_revised(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    time_major=False,
                                    scope=None):
                                    
    if not cells_fw:
        raise ValueError("Must specify at least one fw cell for BidirectionalRNN.")
    if not cells_bw:
        raise ValueError("Must specify at least one bw cell for BidirectionalRNN.")
    if not isinstance(cells_fw, list):
        raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
    if not isinstance(cells_bw, list):
        raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
    if len(cells_fw) != len(cells_bw):
        raise ValueError("Forward and Backward cells must have the same depth.")
    if (initial_states_fw is not None and
        (not isinstance(initial_states_fw, list) or
        len(initial_states_fw) != len(cells_fw))):
        raise ValueError(
            "initial_states_fw must be a list of state tensors (one per layer).")
    if (initial_states_bw is not None and
        (not isinstance(initial_states_bw, list) or
        len(initial_states_bw) != len(cells_bw))):
        raise ValueError(
            "initial_states_bw must be a list of state tensors (one per layer).")

    states_fw = []
    states_bw = []
    prev_layer = inputs
    F_outputs = []
    B_outputs = []

    with vs.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with vs.variable_scope("cell_%d" % i):
                outputs, (state_fw, state_bw) = rnn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype,
                    time_major=time_major)
                # Concat the outputs to create the new input.
                prev_layer = array_ops.concat(outputs, 2)
                F_outputs.append(outputs[0])
                B_outputs.append(outputs[1])
            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return F_outputs,B_outputs