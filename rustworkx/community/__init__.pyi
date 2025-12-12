# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see src/community/

from rustworkx import PyGraph
from rustworkx import PyDiGraph

from typing import Any, Callable
from collections.abc import Sequence

def louvain_communities(
    graph: PyGraph,
    /,
    weight_fn: Callable[[Any], float] | None = ...,
    resolution: float = ...,
    threshold: float = ...,
    seed: int | None = ...,
    min_community_size: int | None = ...,
    adjacency: list[list[int]] | None = ...,
) -> list[list[int]]: ...
def modularity(
    graph: PyGraph,
    communities: Sequence[Sequence[int]],
    /,
    weight_fn: Callable[[Any], float] | None = ...,
    resolution: float = ...,
) -> float: ...
def leiden_communities(
    graph: PyGraph | PyDiGraph,
    /,
    weight_fn: Callable[[Any], float] | None = ...,
    resolution: float = ...,
    seed: int | None = ...,
    min_weight: float | None = ...,
    max_iterations: int | None = ...,
    return_hierarchy: bool = ...,
    adjacency: list[list[int]] | None = ...,
) -> list[list[int]]: ...
def cpm_communities(
    graph: PyGraph,
    /,
    k: int = ...,
) -> list[list[int]]: ...
def find_maximal_cliques(
    graph: PyGraph,
    /,
) -> list[list[int]]: ...
def asyn_lpa_communities(
    graph: PyGraph | PyDiGraph,
    /,
    weight: str | None = ...,
    seed: int | None = ...,
    max_iterations: int | None = ...,
    adjacency: list[list[int]] | None = ...,
) -> list[list[int]]: ...
def asyn_lpa_communities_strongest(
    graph: PyGraph | PyDiGraph,
    /,
    weight: str | None = ...,
    seed: int | None = ...,
) -> list[list[int]]: ...
