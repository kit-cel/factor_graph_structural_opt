# The following file implements the sum-product algorithm (SPA) on factor graphs in PyTorch.
# The SPA is fully differentiable and can be used for back-propagation.
# The SPA works with arbitrary factor graph structures and different degrees for each FN and VN,
# where the structure is defined by the biadjacency_matrix of the graph and the potentials of the FNs.

# For an efficient computation with VNs and CNs of different degrees,
# the function "convert_biadjacency_to_connections" takes the biadjacency_matrix and groups FNs and VNs of the same
# degree so that one tensor calculation for all FNs/VNs of same degree can be used.

# The functions which are relevant to the paper are
# FactorGraph.sum_product_algorithm(), FactorGraph.fn_update(), FactorGraph.vn_update()

import math

import torch as t
from dataclasses import dataclass
from typing import Optional

@dataclass
class FNGroup:
    degree: int # Degree of all FNs which belong to this group

    # Start and end idx define an index range of the FNs that have the degree 'degree' and whose potentials are stored by this object
    start_idx: int
    end_idx: int

    # Potential of the FNs of degree 'degree': [batch, fn_idx (local), x1, x2, ..., xn] where the xi are the possible values of the VNs
    log_potential: Optional[t.Tensor]

def convert_biadjacency_to_connections(biadjacency_matrix: t.Tensor, fn_start_slots: t.Tensor = None) \
        -> tuple[t.Tensor, t.Tensor, int, int, list[FNGroup], t.Tensor, t.Tensor]:
    """
    Converts biadjacency_matrix in an efficient structure to compute the SPA with PyTorch
    :param biadjacency_matrix: tensor (vn, fn)
    :param fn_start_slots: defines for each FN which slot_idx the first "1" entry in the biadjacency_matrix of it should have
    :return: tuple(vn_connections, fn_connections, vn_degree_max: int, fn_degree_max: int, fn_sorted_to_original_index_map, potentials)
        vn_connections: tensor "(vn, vn_slot, (fn, fn_slot)"
        fn_connections: tensor "(fn, fn_slot, (vn, fn_slot)"
        fn_sorted_to_original_index_map: tensor: List of indices which can be used to invert the sorting of the FNs: map[new_fn_idx] = old_fn_idx
        potentials: List containing the potentials as tensors.
    """

    #-------Sort FNs by degree (ascending)-------
    fn_degrees_sorted, fn_index_map_new_to_old = t.sort(t.count_nonzero(biadjacency_matrix, 0))
    biadjacency_matrix_sorted = biadjacency_matrix[:, fn_index_map_new_to_old]
    if fn_start_slots is not None:
        fn_start_slots_sorted = fn_start_slots[fn_index_map_new_to_old]

    fn_index_map_old_to_new = t.zeros(biadjacency_matrix.shape[1], dtype=t.long)
    fn_index_map_old_to_new[fn_index_map_new_to_old] = t.arange(biadjacency_matrix.shape[1])

    fn_degree_max = fn_degrees_sorted[-1]

    #-------Create PotentialData without potentials-------
    new_log_potentials = []
    for fn_degree in range(1, fn_degree_max + 1):
        new_fn_indices_with_degree = t.nonzero(fn_degrees_sorted == fn_degree)
        if len(new_fn_indices_with_degree) != 0:
            start_idx = new_fn_indices_with_degree[0].item()
            end_idx = new_fn_indices_with_degree[-1].item()

            new_log_potentials.append(FNGroup(fn_degree, start_idx, end_idx, None))

    vn_degree_max = t.count_nonzero(biadjacency_matrix_sorted, 1).max()

    #-------Create connections-------
    vn_connections = -t.ones(biadjacency_matrix_sorted.shape[0], vn_degree_max, 2, dtype=t.long)
    """
    tensor (vn_idx, vn_slot, (fn_idx, fn_slot)): Stores for each VN (vn_idx) and its slot (vn_slot): 
    vn_connections[vn_idx, vn_slot, 0] = connected FN fn_idx
    vn_connections[vn_idx, vn_slot, 1] = FN slot of fn_idx at which the VN vn_idx with the slot vn_slot is connected to
    """

    fn_connections = -t.ones(biadjacency_matrix_sorted.shape[1], fn_degree_max, 2, dtype=t.long)
    """
    tensor (fn_idx, fn_slot, (vn_idx, vn_slot)): Stores for each FN (fn_idx) and its slot (fn_slot): 
    fn_connections[fn_idx, fn_slot, 0] = connected VN vn_idx
    fn_connections[fn_idx, fn_slot, 1] = VN slot of vn_idx at which the FN fn_idx with the slot fn_slot is connected to
    """

    if fn_start_slots is None:
        fn_next_free_slot = t.zeros(biadjacency_matrix_sorted.shape[1], dtype=t.long) # stores the next free slot for each FN
    else:
        fn_next_free_slot = t.clone(fn_start_slots_sorted)

    for vn_idx in range(biadjacency_matrix_sorted.shape[0]):
        # Finds the indices of the FNs which are connected with vn_idx
        connected_fn_indices = t.nonzero(biadjacency_matrix_sorted[vn_idx, :])[:, 0]

        vn_slots = t.arange(connected_fn_indices.shape[0])
        # Stores the connected FNs
        vn_connections[vn_idx, vn_slots, 0] = connected_fn_indices
        # Stores the slot of each FN at which the current VN is connected with. This is the next free slot of the corresponding FNs because
        # this slot is connected to this VN (see below)
        vn_connections[vn_idx, vn_slots, 1] = fn_next_free_slot[connected_fn_indices]

        # Stores for each of the connected FN that it is connected to the current VN at the next free slot
        fn_connections[connected_fn_indices, fn_next_free_slot[connected_fn_indices], 0] = vn_idx
        # Stores for each of the connected FN the VN Slot of the current VN to which it is connected to.
        fn_connections[connected_fn_indices, fn_next_free_slot[connected_fn_indices], 1] = vn_slots

        # Increases the next free slot for the FNs which have got a new connection
        fn_next_free_slot[connected_fn_indices] += 1
        fn_next_free_slot = t.remainder(fn_next_free_slot, fn_degrees_sorted)

    return vn_connections, fn_connections, vn_degree_max, fn_degree_max, \
        new_log_potentials, fn_index_map_old_to_new, fn_index_map_new_to_old

def convert_messages(xn_messages: t.Tensor, yn_messages_connections: t.Tensor) -> t.Tensor:
    """
    Rearranges the messages stored in fn perspective, into messages stored in vn perspective or vice versa
    In the following x and y stands for x=fn and y=vn if converted from fn to vn perspective and for x=vn and y=fn otherwise.
    :param xn_messages: tensor (batch, xn_idx, xn_slot, message_values) (cells that correspond to no valid slot can have an arbitrary value)
    :param yn_messages_connections: tensor (yn, yn_slot, (xn, xn_slot)) (entry "-1" represents no connection)
    :return: yn_messages: tensor (batch, yn_idx, yn_slot, message_values) (cells that are not used are filled with zeros)
    """
    zero = t.zeros(1, device=xn_messages.device)
    return t.where(yn_messages_connections[None, :, :, 0, None] >= 0,
                   xn_messages[:, yn_messages_connections[:, :, 0], yn_messages_connections[:, :, 1], :], zero)

def jacobian_log(tensor: t.Tensor, dim: int) -> t.Tensor:
    """
    Executes the jacobian logarithm along the dimension "dim"
    :param tensor: Tensor
    :param dim: Dimension on which the jacobian log is calculated and which is reduced
    :return: Result of the jacobian log, without dimension 'dim'
    """

    maximum = t.max(tensor, dim, keepdim=True)[0]
    return maximum.squeeze(dim) + t.log(t.sum(t.exp(tensor - maximum), dim))

def jacobian_log_multi_dim(tensor: t.Tensor, dim: int, ignore_first_dim: int = 0) -> t.Tensor:
    """
    Executes the jacobian log by summing over all dimensions except 'dim' and the first 'ignore_first_dim' dimensions
    ln(sum_{i in {set of all indices of the dimensions except dim}} exp(tensor[i]))
    :param tensor: Tensor
    :param dim: Dimension which is not reduced. ('dim = 0' is the first dimension after the ignored dimensions)
    :param ignore_first_dim: The first 'ignore_first_dim' dimensions are also not reduced. (For instance, batch dimension).
    :return: tensor: Result of jacobinan log. Only the dimension 'dim' and the ignored dimensions are left
    """
    if tensor.dim() == ignore_first_dim + 1:
        assert dim == 0
        # Here no dimension would be reduced in the sum below and no summation is applied.
        # So we just apply log(exp(.)) on a tensor which is the identity function:
        return tensor
    else:
        slice_tuple = tuple(range(ignore_first_dim, ignore_first_dim + dim)) \
                      + tuple(range(ignore_first_dim + dim + 1, tensor.dim()))
        maximum = t.amax(tensor, slice_tuple, keepdim=True)

        return maximum.squeeze() + t.log(t.sum(t.exp(tensor - maximum), slice_tuple))

def unsqueeze_tuple(idx: int, n: int) -> tuple:
    """
    Returns a tuple that can be used to unsqueeze a tensor of dim [a] to [1, ..., 1, a, 1..., 1]
    which has n dimensions and dim [a] is at index 'idx'
    :param idx: Index at which the original dimension should occur
    :param n: Number of dimensions of the new tensor
    :return: tuple representing the unsqueeze slicing
    """

    return (None,) * idx + (slice(None),) + (None,) * (n - idx - 1)

def create_indexing(n: int, *indices) -> list:
    """
    Returns a list with "slice(None)" at the "indices" and "None"s elsewhere
    Can be used for indexing to rearrange the dimensions of a tensor to the new dimensions "indices" and add dimensions
    of size one at the other indices
    :param n: Number of dimensions
    :param indices: List indices with "slice(None)"
    :return: List for indexing
    """
    indexing = [None] * n
    for idx in indices:
        indexing[idx] = slice(None)
    return indexing

def tensor_sum(tensor: t.Tensor) -> t.Tensor:
    """
    Computes a tensor sum.
    :param tensor: [d1, ..., dn, m, n] The second last dimension [m] is used to distinguish the tensors which are summed [n]. (see tensor_sum_list)
    The last dimension represents the values that are summed. And the other dimensions [d1, ..., dn]  are untouched.
    :return: tensor: [d1, ..., dn, n, ..., n], where there are m dimensions of size n
    """

    result = t.zeros(tensor.shape[:-2] + (tensor.shape[-1],) * tensor.shape[-2], device=tensor.device)

    for idx in range(tensor.shape[-2]):
        result += tensor[(tensor.dim() - 2) * (slice(None),) + (idx,) + unsqueeze_tuple(idx, tensor.shape[-2])]

    return result

def normalize_log_messages(messages: t.Tensor):
    """
    Normalizes log-Messages such that the sum of their corresponding probabilities is 1
    Normalization is done over the last dim. The other dim. aren't changed.
    :param messages: Tensor (..., message_values)
    :return: tensor (..., message_values) Normalized message vector
    """

    messages -= jacobian_log(messages, -1)[..., None]


class FactorGraph:
    vn_connections: t.Tensor
    """
    tensor (vn_idx, vn_slot, (fn_idx, fn_slot)): Stores for each VN (vn_idx) and its slot (vn_slot): 
    vn_connections[vn_idx, vn_slot, 0] = connected FN fn_idx
    vn_connections[vn_idx, vn_slot, 1] = FN slot of fn_idx at which the VN vn_idx with the slot vn_slot is connected to
    """

    fn_connections: t.Tensor
    """
    tensor (fn_idx, fn_slot, (vn_idx, vn_slot)): Stores for each FN (fn_idx) and its slot (fn_slot): 
    fn_connections[fn_idx, fn_slot, 0] = connected VN vn_idx
    fn_connections[fn_idx, fn_slot, 1] = VN slot of vn_idx at which the FN fn_idx with the slot fn_slot is connected to
    """

    fn_groups: list[FNGroup] # List containing the groups that group FNs of the same degree
    fn_index_map_old_to_new: t.Tensor # fn_index_map_old_to_new[old_fn_idx] = new_fn_idx
    fn_index_map_new_to_old: t.Tensor # fn_index_map_new_to_old[new_fn_idx] = old_fn_idx
    vn_degree_max: int # maximal degree a vn in this graph can have
    fn_degree_max: int # maximal degree a vn in this graph can have
    number_of_values: int # Number of possible values for each VN
    batch_size = -1 # batch_size of the loaded potentials. Negative if no potentials have been loaded yet.
    device: t.device # PyTorch device

    use_nbp_weights = False # Determines if the weights for NBP are used in the update functions.
    nbp_weights_vn_outgoing_messages = None # Weights for NBP
    nbp_weights_fn_outgoing_messages = None # Weights for NBP

    def __init__(self, biadjacency_matrix: t.Tensor, number_of_values: int, device: t.device = t.device('cpu'),
                 no_update_normalization=False, fn_start_slots: t.Tensor = None):
        """
        :param device: Torch device
        :param biadjacency_matrix: tensor(vn, fn)
        :param log_potentials: List of all FN potentials. Each potential is a tensor of the form [x1, ..., xn] where n is the degree of the respective FN
        :param number_of_values: Number of possible values each VN can have
        :param fn_start_slots: PyTorch Tensor or None: Defines for each FN to which slot_idx the first "1" entry in the
        biadjacency_matrix belongs to.
        """
        vn_connections, fn_connections, self.vn_degree_max, self.fn_degree_max, self.fn_groups, \
            fn_index_map_old_to_new, fn_index_map_new_to_old = convert_biadjacency_to_connections(biadjacency_matrix, fn_start_slots)

        self.vn_connections = vn_connections.to(device)
        self.fn_connections = fn_connections.to(device)
        self.fn_index_map_old_to_new = fn_index_map_old_to_new.to(device)
        self.fn_index_map_new_to_old = fn_index_map_new_to_old.to(device)

        self.number_of_values = number_of_values
        self.device = device

        self.no_update_normalization = no_update_normalization

    def sum_product_algorithm(self, spa_iterations: int, normalize_beliefs=False, log_prior_beliefs=None) -> t.Tensor:
        """
        Executes the sum product algorithm on the factor graph
        :param spa_iterations: Number of iterations
        :param log_prior_beliefs: If None, no prior beliefs are used. Otherwise, prior beliefs for all VNs can be set using a tensor t
         with indices (batch, vn_idx, message_values)
        :return: Logarithmic belief for each VN: tensor with indices (batch, vn_idx, message_values)
        """

        assert self.batch_size > 0 # Checks if potentials have been loaded yet

        # Init vn_messages with prior beliefs if existing and with constant values
        # (maximal uncertainty about the beliefs) otherwise
        if log_prior_beliefs is not None:
            vn_outgoing_messages = log_prior_beliefs.clone()[:, :, None, :].expand(-1, -1, self.vn_degree_max, -1)
        else:
            vn_outgoing_messages = t.full((self.batch_size, self.vn_connections.shape[0], self.vn_degree_max, self.number_of_values), -math.log(self.number_of_values), device=self.device)

        fn_incoming_messages = self.convert_messages_vn_to_fn_view(vn_outgoing_messages)

        # Loop over SPA iterations.
        for spa_iteration in range(spa_iterations):
            vn_incoming_messages = self.fn_update(fn_incoming_messages, spa_iteration)
            fn_incoming_messages, log_vn_beliefs = self.vn_update(vn_incoming_messages, spa_iteration)

        if normalize_beliefs:
            normalize_log_messages(log_vn_beliefs)

        return log_vn_beliefs


    def vn_update(self, vn_incoming_messages: t.Tensor, spa_iteration: int) -> tuple[t.Tensor, t.Tensor]:
        """
        Performs VN update of sum-product algorithm
        :param vn_incoming_messages: tensor (batch, vn_idx, vn_slot, message_values) (If a slot is not used the entry is zero)
        :param spa_iteration: Current spa iteration starting with 0 for the first one.
        :return: tuple[fn_incoming_messages, vn_beliefs]:
        fn_incoming_messages: tensor (batch, fn_idx, fn_slot, message_values) (If a slot is not used the entry is zero)
        log_vn_beliefs: tensor (batch, vn_idx, message_values) (not normalized) log belief/ log a posteriori probability of each VN
        """

        # VN-Update: First line sums for each VN all incoming messages.
        # The second line uses indexing to calculate all extrinsic sums for the outgoing messages
        vn_messages_sum = t.sum(vn_incoming_messages, 2, keepdim=False)
        vn_outgoing_messages = vn_messages_sum[:, :, None, ...] - vn_incoming_messages

        # To avoid an overflow, the messages are either normalized or constant.
        if not self.no_update_normalization:
            normalize_log_messages(vn_outgoing_messages)
        else:
            vn_outgoing_messages = vn_outgoing_messages - t.max(vn_outgoing_messages, -1, keepdim=True)[0]

        # Apply NBP weights
        if self.use_nbp_weights:
            vn_outgoing_messages *= self.nbp_weights_vn_outgoing_messages[spa_iteration, None, :, :, None]

        # Warning: For VNs whose degree is less than the maximum degree, vn_outgoing_messages has non-zero entries that correspond to no real message
        # This is no problem because "convert_messages" just ignores these entries
        return self.convert_messages_vn_to_fn_view(vn_outgoing_messages), vn_messages_sum

    def fn_update(self, fn_incoming_messages: t.Tensor, spa_iteration: int) -> t.Tensor:
        """
        Performs FN update of sum-product algorithm
        :param fn_incoming_messages: tensor (batch, fn_idx, fn_slot, message_values) (If a slot is not used the entry is zero)
        :param spa_iteration: Current spa iteration starting with 0 for the first one.
        :return: vn_incoming_messages: tensor (batch, vn_idx, vn_slot, message_values) (If a slot is not used the entry is zero)
        """

        fn_outgoing_messages = t.zeros_like(fn_incoming_messages)
        for fn_group in self.fn_groups:
            # Extracts incoming messages for all FNs belonging to the current fn_group:
            # tensor(batch, fn_idx, fn_slot, message_values)
            incoming_messages_fn_group = fn_incoming_messages[
                :, fn_group.start_idx: (fn_group.end_idx + 1), :fn_group.degree, :]

            # tensor(batch, fn_idx, fn_slot, message_values)
            outgoing_messages_fn_group = t.zeros_like(incoming_messages_fn_group)

            # For each FN, all incoming messages are summed and stored in tensor_sum_all_incoming_messages
            # In the following loop, the extrinsic sum needed for the FN-update is calculated by removing the respective incoming message

            # tensor (batch, fn_idx, message_values, ..., message_values) where 'message_values' occurs potential_data.degree times
            tensor_sum_all_incoming_messages = tensor_sum(incoming_messages_fn_group)
            for outgoing_slot_idx in range(fn_group.degree):
                tensor_sum_extrinsic_incoming_messages = tensor_sum_all_incoming_messages \
                    - incoming_messages_fn_group[(slice(None), slice(None))
                    + (outgoing_slot_idx,) + unsqueeze_tuple(outgoing_slot_idx, fn_group.degree)]

                # Calculate FN update using jacobian logarithm.
                outgoing_messages_fn_group[:, :, outgoing_slot_idx, :] = \
                    jacobian_log_multi_dim(fn_group.log_potential[0] + tensor_sum_extrinsic_incoming_messages, outgoing_slot_idx, 2)

            # Normalizes the messages. In theory, this does not change the result but avoids overflows.
            if not self.no_update_normalization:
                normalize_log_messages(fn_outgoing_messages[:, fn_group.start_idx: (fn_group.end_idx + 1), :fn_group.degree, :])

            # Multiply outgoing messages with nbp weights.
            if not self.use_nbp_weights or fn_group.degree == 1:
                fn_outgoing_messages[:, fn_group.start_idx: (fn_group.end_idx + 1), :fn_group.degree, :] \
                    = outgoing_messages_fn_group
            else:
                fn_outgoing_messages[:, fn_group.start_idx: (fn_group.end_idx + 1), :fn_group.degree, :]\
                    = outgoing_messages_fn_group * self.nbp_weights_fn_outgoing_messages[
                    spa_iteration, None, fn_group.start_idx: (fn_group.end_idx + 1), :fn_group.degree, None]

        return self.convert_messages_fn_to_vn_view(fn_outgoing_messages)

    def set_nbp_weights(self, nbp_weights_vn_outgoing_messages: t.Tensor, nbp_weights_fn_outgoing_messages: t.Tensor):
        """
        If set, the weights are multiplied by the outgoing messages of the VNs and FNs
        :param nbp_weights_vn_outgoing_messages: tensor (bp_iteration, vn_idx, vn_slot):
        :param nbp_weights_fn_outgoing_messages: tensor (bp_iteration, fn_idx, fn_slot)
        """

        self.nbp_weights_vn_outgoing_messages = nbp_weights_vn_outgoing_messages
        self.nbp_weights_fn_outgoing_messages = nbp_weights_fn_outgoing_messages
        self.use_nbp_weights = True

    def convert_messages_vn_to_fn_view(self, vn_messages: t.Tensor) -> t.Tensor:
        """
        Converts messages in VN perspective into messages in FN perspective
        :param vn_messages: tensor (batch, vn_idx, vn_slot, message_values) (cells that correspond to no valid slot can have an arbitrary value)
        :return: fn_messages: tensor (batch, fn_idx, fn_slot, message_values) (cells that are not used are filled with zeros)
        """
        return convert_messages(vn_messages, self.fn_connections)

    def convert_messages_fn_to_vn_view(self, fn_messages: t.Tensor) -> t.Tensor:
        """
        Converts messages in FN perspective into messages in VN perspective
        :param vn_messages_outgoing: tensor (batch, vn_idx, vn_slot, message_values) (cells that correspond to no valid slot can have an arbitrary value)
        :return: fn_messages: tensor (batch, fn_idx, fn_slot, message_values) (cells that are not used are filled with zeros)
        """
        return convert_messages(fn_messages, self.vn_connections)

    def convert_fn_message_to_original_biadjacency(self, fn_data: t.Tensor):
        """
        Reorders the fn_data so that their indices match with the fn indices of the original biadjancy matrix
        :param fn_data: tensor (batch, fn_idx, ...)
        :return: fn_data with original indexing of the FNs
        """

        return fn_data[:, self.fn_index_map_old_to_new, ...]

    def load_potentials(self, log_potentials: list[t.tensor], batch_size: int):
        """
        Sets the potentials of the FNs
        :param log_potentials: List of all FN potentials. Each potential is a tensor
        of the form [batch, x1, ..., xn] where n is the degree of the respective FN
        or  [batch,iter, x1, ..., xn] if potentials_for_each_iteration == True
        :param number_of_values: Number of possible values for each VN.
        :param batch_size: int
        :return: potentials: List containing the potentials as tensors.
        """

        self.batch_size = batch_size

        assert len(log_potentials) == self.fn_connections.shape[0]

        for log_potential_data in self.fn_groups: # potential_data: stores potentials of FNs of same degree
            log_potential_data.log_potential = t.zeros(1, batch_size, log_potential_data.end_idx
                - log_potential_data.start_idx + 1, *(log_potential_data.degree * [self.number_of_values]), device=self.device)

            for log_potential_idx, old_fn_idx in enumerate(self.fn_index_map_new_to_old[log_potential_data.start_idx:(log_potential_data.end_idx + 1)]):
                assert log_potentials[old_fn_idx].shape[1:] == log_potential_data.log_potential.shape[3:]

                log_potential_data.log_potential[:, :, log_potential_idx, ...] = log_potentials[old_fn_idx][None, ...]
