from __future__ import annotations

def map_21(size1, size2, shared:dict[int, int]):
        """For two models with size1 and size2 variables each, and
        shared variables mapping from model2 to model1 index, this
        function returns a map that maps any index from the second model
        to that of the combined model, where the combination is the
        variables of the first model, followed by variables of the
        second model that are not shared.
        """
        v2_to_v1 = dict()
        next_node_id = size1
        for i in range(size2):
            if i not in shared:
                v2_to_v1[i] = next_node_id
                next_node_id += 1
            else:
                v2_to_v1[i] = shared[i]
        return v2_to_v1