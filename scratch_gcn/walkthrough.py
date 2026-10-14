"""Step-by-step walkthrough of the scratch GCN. Run: python -m scratch_gcn.walkthrough"""
import torch

from scratch_gcn.core import (GCN, MLP, GCNLayer, acc, check_manual, check_permutation, drop_edges,
                              make_dataset, manual_layer_node, prep_graph, random_graph, set_seed,
                              train_model)

set_seed(0)
x, y, ei, masks = make_dataset(seed=0)
n = x.shape[0]

print("\n=== 1. DATASET ===")
print("features:", tuple(x.shape), "| class counts:", y.bincount().tolist())
print("directed edge entries:", ei.shape[1], "| undirected edges:", ei.shape[1] // 2)
print("same-class edge fraction:", round((y[ei[0]] == y[ei[1]]).float().mean().item(), 3))
print("split sizes:", {k: int(v.sum()) for k, v in masks.items()})

print("\n=== 2. SELF-LOOPS, NORMALISATION, MESSAGE PASSING ===")
src, dst, norm = prep_graph(ei, n)
print("messages before / after adding self-loops:", ei.shape[1], "->", src.shape[0])
node = 0
nbrs = ei[0][ei[1] == node].tolist()
print(f"node {node}: neighbours {nbrs}, degree incl. self-loop = {len(nbrs) + 1}")
for s, d, w in zip(src.tolist(), dst.tolist(), norm.tolist()):
    if d == node:
        print(f"  message {s} -> {node}: weight 1/sqrt(d_i*d_j) = {w:.4f}")
layer = GCNLayer(3, 4)
print("one layer output shape:", tuple(layer(x, (src, dst, norm)).shape), "(200 nodes x 4 hidden)")

print("\n=== 3. TRAINING (GCN) and 6. MLP COMPARISON, same features and split ===")
g = prep_graph(ei, n)
set_seed(0); gcn = GCN(3, 16, 2); print("GCN best val acc/loss:", train_model(gcn, x, y, g, masks))
set_seed(0); mlp = MLP(3, 16, 2); print("MLP best val acc/loss:", train_model(mlp, x, y, None, masks))
gcn.eval(); mlp.eval()
with torch.no_grad():
    print("GCN test acc:", acc(gcn(x, g), y, masks["test"]))
    print("MLP test acc:", acc(mlp(x), y, masks["test"]))

print("\n=== 4. MANUAL CHECK of one node, one layer ===")
auto = gcn.l1(x, g)[node].detach()
manual = manual_layer_node(x, ei, gcn.l1.W, gcn.l1.b, node)
print("library-free layer :", auto[:4].tolist())
print("hand calculation   :", manual[:4].tolist())
print("match:", check_manual(gcn, x, ei, node=node))

print("\n=== 5. REORDERING CHECK ===")
print("predictions reordered correspondingly:", check_permutation(gcn, x, ei))

print("\n=== 7. DOES THE MODEL USE GRAPH STRUCTURE? ===")
with torch.no_grad():
    for f in (0.0, 0.5, 1.0):
        gd = prep_graph(drop_edges(ei, f, 0), n)
        print(f"  trained GCN, {int(f*100):3d}% of edges removed at test time: acc = {acc(gcn(x, gd), y, masks['test']):.3f}")
set_seed(0)
gr = prep_graph(random_graph(n, ei.shape[1] // 2, 0), n)
gcn_r = GCN(3, 16, 2); train_model(gcn_r, x, y, gr, masks); gcn_r.eval()
with torch.no_grad():
    print(f"  GCN retrained on a random graph (same edge count): acc = {acc(gcn_r(x, gr), y, masks['test']):.3f}")