import torch as th


def extract_root(batch, device, logits):
    g = batch.graph
    root_idx = []
    result_idx = 0
    for idx in g.batch_num_nodes:
        root_idx.append(result_idx)
        result_idx = result_idx + idx
    root_idx = th.LongTensor(root_idx).to(device)
    batch_label = th.index_select(batch.label, 0, root_idx)
    if logits.shape[0] != g.batch_size:
        logits = th.index_select(logits, 0, root_idx)
    return batch_label, logits
