from __future__ import annotations
import code
import dataclasses
import math

import torch
import tqdm
from dblm.core import graph
from dblm.core.interfaces import pgm
from dblm.core.interfaces import inferencer
from dblm.core.modeling import constants, probability_tables, switching_tables

import collections

@dataclasses.dataclass
class InferenceResults:

    query_marginals: list[probability_tables.LogLinearProbabilityTable]
    messages_to_variables: list[list[pgm.PotentialTable | pgm.ProbabilityTable | None]] | None = None
    messages_to_factors: list[list[pgm.PotentialTable | None]] | None = None

def conditional_factor_variables(factor_variables, factor_functions, observation, materialize_switch=False):
    if len(observation) == 0:
        return factor_variables
    return [tuple(var for var in factor_vars if var not in observation) for factor_vars in factor_variables]


def conditional_factor_functions(factor_variables, factor_functions, observation, materialize_switch=False):
    new_factor_functions = []
    for factor_vars, factor_function in zip(factor_variables, factor_functions):
        factor_function : pgm.PotentialTable
        local_observation: dict[int,int] | dict[int, torch.Tensor] = {local_name: observation[global_name] for local_name, global_name in enumerate(factor_vars) if global_name in observation} # type:ignore
        if len(local_observation) > 0:
            if not materialize_switch and isinstance(factor_function, switching_tables.SwitchingTable):
                new_factor_functions.append(factor_function.condition_on(local_observation, bp=True, cartesian=False))
            else:
                new_factor_functions.append(factor_function.condition_on(local_observation))
        else:
            if (not materialize_switch) and isinstance(factor_function, switching_tables.SwitchingTable):
                new_factor_function = factor_function.to_bp_table()
                if len(observation) > 0:
                    new_factor_function = new_factor_function.expand_batch_dimensions(tuple(next(observation.values().__iter__()).size()))
                new_factor_functions.append(new_factor_function)
            elif len(observation) == 0 or isinstance(next(observation.values().__iter__()), int):
                new_factor_functions.append(factor_function)
            elif len(observation) > 0 and isinstance(next(observation.values().__iter__()), torch.Tensor):
                new_factor_functions.append(factor_function.expand_batch_dimensions(tuple(next(observation.values().__iter__()).size()))) # type:ignore
            else:
                raise ValueError(f"Unkonwn case: len(observation)={len(observation)}, "
                                 f"observation_type=={type(next(observation.values().__iter__()))}, "
                                 f"materialize_switch={materialize_switch}, "
                                 f"factor_function={factor_function}")
    return new_factor_functions

class FactorGraphBeliefPropagation(inferencer.MarginalInferencer):

    def __init__(self) -> None:
        super().__init__()
        # define message passing schedule

    def inference(self, model: pgm.FactorGraphModel, observation: dict[int, int] | dict[int, torch.Tensor], query: list[int], iterations=10, return_messages=False, renormalize=True, materialize_switch=False, messages_to_factors=None, messages_to_variables=None, true_parallel=False, allow_query_observation=False):
        if any(query_var in observation for query_var in query) and not allow_query_observation:
            raise ValueError("query is part of observation, did you have a typo?")
        factor_variables = conditional_factor_variables(model.factor_variables(), model.factor_functions(), observation, materialize_switch)
        factor_functions = conditional_factor_functions(model.factor_variables(), model.factor_functions(), observation, materialize_switch)
        factor_graph: graph.FactorGraph = model.graph()
        messages_to_factors = [None] * factor_graph.num_edges if messages_to_factors is None else messages_to_factors
        messages_to_variables = [None] * factor_graph.num_edges if messages_to_variables is None else messages_to_variables
        messages_to_factors_book = []
        messages_to_variables_book = []
        for _ in tqdm.tqdm(range(iterations),leave=False):
            if true_parallel:
                messages_to_factors, messages_to_variables = self._variable_to_factor(model, observation, messages_to_variables), self._factor_to_variable(model, observation,factor_variables, factor_functions, messages_to_factors) # type:ignore
            else:
                messages_to_factors = self._variable_to_factor(model, observation, messages_to_variables)
                messages_to_variables = self._factor_to_variable(model, observation,factor_variables, factor_functions, messages_to_factors) # type:ignore
            # code.interact(local=locals())
            if renormalize:
                messages_to_variables = [message.renormalize() if message is not None else None for message in messages_to_variables]
            messages_to_factors_book.append(messages_to_factors)
            messages_to_variables_book.append(messages_to_variables)
        query_factors: dict[int,list[pgm.PotentialTable]] = collections.defaultdict(list)
        query_set = set(query)
        for edge, message in zip(factor_graph.edges, messages_to_variables): # type:ignore
            if edge.variable_id in query_set and message is not None:
                query_factors[edge.variable_id].append(message)
        results = []
        for var in query:
            if var in observation:
                if not isinstance(observation[var], torch.Tensor):
                    raise NotImplementedError("allow_query_observation not implemented for nontensor observations")
                table = probability_tables.LogLinearProbabilityTable([model.nvals[var]], [], constants.TensorInitializer.CONSTANT).to(factor_functions[0].device).expand_batch_dimensions(tuple(observation[var].size())) # type:ignore
                table._logits.data.fill_(-math.inf)
                table._logits.data = table._logits.data.scatter(dim=-1, index=observation[var][..., None], src=torch.zeros_like(observation[var][..., None], dtype=table.logits.data.dtype)) # type:ignore
                results.append(table)
                continue
            if len(query_factors[var]) == 0:
                results.append(probability_tables.LogLinearProbabilityTable([model.nvals[var]], [], constants.TensorInitializer.CONSTANT).to(factor_functions[0].device))
            else:
                results.append(probability_tables.LogLinearProbabilityTable.joint_from_factors(1,
                                                                            [model.nvals[var]],
                                                                            [(0,)] * len(query_factors[var]),
                                                                            query_factors[var]))
        return InferenceResults(results,
                                messages_to_factors=messages_to_factors_book if return_messages else None,
                                messages_to_variables=messages_to_variables_book if return_messages else None)

    def _variable_to_factor(self, model, observation, messages_to_variables):
        var_to_edge: dict[int, list(int)] = collections.defaultdict(list) # type:ignore
        messages_to_factors = [None] * model.graph().num_edges
        for i, edge in enumerate(model.graph().edges):
            var_to_edge[edge.variable_id].append(i)
        for var, edge_ids in var_to_edge.items():
            if var in observation:
                continue # DON'T SEND  MESSAGE FROM OBSERVED VARIABLES
            else:
                for out_edge_id in edge_ids:
                    messages_to_factors[out_edge_id] = self._message_from_single_variable_to_single_factor(messages_to_variables, out_edge_id, edge_ids)
        return messages_to_factors

    def _message_from_single_variable_to_single_factor(self, messages, out_index, in_indices):
        outgoing_factors = []
        for index in in_indices:
            if index != out_index and messages[index] is not None:
                outgoing_factors.append(messages[index])
        if len(outgoing_factors) == 0:
            return None
        elif len(outgoing_factors) == 1:
            return outgoing_factors[0]
        else:
            nvars = 1
            nvals = list(outgoing_factors[0].nvals)
            return probability_tables.LogLinearPotentialTable.joint_from_factors(nvars, nvals, [(0,)] * len(outgoing_factors), outgoing_factors)

    def _factor_to_variable(self, model, observation, factor_variables, factor_functions, messages_to_factors):
        factor_to_edge: dict[int, list(int)] = collections.defaultdict(list) # type:ignore
        messages_to_variables = [None] * model.graph().num_edges
        for i, edge in enumerate(model.graph().edges):
            if edge.variable_id not in observation: # DON'T MESSAGE OBSERVED VARIABLES
                factor_to_edge[edge.factor_id].append(i)
        for factor_id, edge_ids in factor_to_edge.items():
            for out_edge_id in edge_ids:
                # if out_edge_id == 0:
                #     code.interact(local=locals())
                messages_to_variables[out_edge_id] = self._message_from_single_factor_to_single_variable(model, factor_variables[factor_id], factor_functions[factor_id], messages_to_factors, out_edge_id, edge_ids)
        return messages_to_variables

    def _message_from_single_factor_to_single_variable(self, model, factor_variable, factor_function: pgm.PotentialTable, messages, out_index, in_indices):
        if isinstance(factor_function, switching_tables.SwitchingTableForMessagePassing):
            edges = model.graph().edges
            outgoing_message_variable = factor_variable.index(edges[out_index].variable_id) # index is to get the local id
            message =  factor_function.message_to_var(outgoing_message_variable, [messages[in_index] if in_index != out_index else None for in_index in in_indices]) # type:ignore
            return message
        # all of these variable indices below are LOCAL so 0, 1, 2, ..., #vars of factor - 1
        outgoing_nvars_before_marginalization = len(factor_variable) # this is the size (number of variables) of the local potential potential function, which doesn't change from multiplying with any incoming message
        outgoing_nvals_before_marginalization = [model.nvals[i] for i in factor_variable] # this is the size (number of possibilities for each variable) of the local potential function, which doesn't change from multiplying with any incoming message
        outgoing_factor_functions = [factor_function]
        outgoing_factor_variables = [tuple(range(outgoing_nvars_before_marginalization))] # NOTE these indices are LOCAL, from the perspective of the factor, rather than from within the whole factor graph

        edges = model.graph().edges
        outgoing_message_variable = factor_variable.index(edges[out_index].variable_id) # index is to get the local id

        for index in in_indices:
            if index != out_index and messages[index] is not None:
                outgoing_factor_functions.append(messages[index])
                outgoing_factor_variables.append((factor_variable.index(edges[index].variable_id),))
        marginalization_indices = [i for i in range(outgoing_nvars_before_marginalization) if i != outgoing_message_variable]
        if len(outgoing_factor_functions) == 1:
            return factor_function.marginalize_over(marginalization_indices)
        else:
            return probability_tables.LogLinearPotentialTable.joint_from_factors(outgoing_nvars_before_marginalization,
                                                                        outgoing_nvals_before_marginalization,
                                                                        outgoing_factor_variables,
                                                                        outgoing_factor_functions).marginalize_over(marginalization_indices)
