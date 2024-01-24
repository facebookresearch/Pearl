#ifndef S2V_LIB_H
#define S2V_LIB_H

#include "config.h"

extern "C" int Init(const int argc, const char **argv);

extern "C" int n2n_construct(int num_nodes, int num_edges, int* node_degrees, int* edge_pairs, long long* idxes, Dtype* vals);

#endif