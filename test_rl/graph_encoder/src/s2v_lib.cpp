#include "s2v_lib.h"
#include "config.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>
#include <cassert>

int Init(const int argc, const char **argv)
{
    cfg::LoadParams(argc, argv);
    return 0;
}

int n2n_construct(int num_nodes, int num_edges, int* node_degrees, int* edge_pairs, long long* idxes, Dtype* vals)
{
    int nnz = 0;    
    long long* row_ptr = idxes;
    long long* col_ptr = idxes + num_edges;

	for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = 0; j < node_degrees[i]; ++j)
        {
            assert(edge_pairs[nnz * 2 + 1] == i);
            vals[nnz] = cfg::msg_average ? 1.0 / node_degrees[i] : 1.0;
            row_ptr[nnz] = i;
            col_ptr[nnz] = edge_pairs[nnz * 2];
            nnz++;
        }
    }
    assert(nnz == num_edges);
    return 0;
}
