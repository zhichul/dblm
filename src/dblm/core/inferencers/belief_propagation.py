from __future__ import annotations
import code
import dataclasses

import torch
from dblm.core import graph
from dblm.core.interfaces import pgm
from dblm.core.interfaces import inferencer
from dblm.core.modeling import probability_tables

import collections

@dataclasses.dataclass
class InferenceResults:

    query_marginals: list[probability_tables.LogLinearProbabilityTable]
    messages_to_variables: list[list[pgm.PotentialTable | None]] | None = None
    messages_to_factors: list[list[pgm.PotentialTable | None]] | None = None

class FactorGraphBeliefPropagation(inferencer.MarginalInferencer):

    def __init__(self) -> None:
        super().__init__()
        # define message passing schedule

    def inference(self, model: pgm.FactorGraphModel, observation: dict[int, int] | dict[int, torch.Tensor], query: list[int], iterations=10, return_messages=False, renormalize=True):
        if any(query_var in observation for query_var in query):
            raise ValueError("query is part of observation, did you have a typo?")
        factor_variables = model.conditional_factor_variables(observation)
        factor_functions = model.conditional_factor_functions(observation)
        factor_graph: graph.FactorGraph = model.graph()
        messages_to_factors = [None] * factor_graph.num_edges
        messages_to_variables = [None] * factor_graph.num_edges
        messages_to_factors_book = []
        messages_to_variables_book = []
        for _ in range(iterations):
            messages_to_factors, messages_to_variables = self._variable_to_factor(model, observation, messages_to_variables), \
                self._factor_to_variable(model, observation,factor_variables, factor_functions, messages_to_factors) # type:ignore
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
            if var in observation: continue # DON'T SEND MESSAGE FROM OBSERVED VARIABLES  # noqa: E701
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
                messages_to_variables[out_edge_id] = self._message_from_single_factor_to_single_variable(model, factor_variables[factor_id], factor_functions[factor_id], messages_to_factors, out_edge_id, edge_ids)
        return messages_to_variables

    def _message_from_single_factor_to_single_variable(self, model, factor_variable, factor_function: pgm.PotentialTable, messages, out_index, in_indices):
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
