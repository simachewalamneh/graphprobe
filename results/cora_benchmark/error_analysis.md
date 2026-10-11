# Error analysis (GCN, seed 0)

Rows: 4 most confident mistakes + 4 random mistakes.

| node | true | pred | conf | MLP pred | degree | words | neighbour true labels | neighbour predictions | nbr homophily |
|---|---|---|---|---|---|---|---|---|---|
| 2291 | 0 | 4 | 0.51 | 4 | 9 | 4 | {3: 1, 2: 1, 4: 3, 0: 4} | {4: 7, 3: 1, 0: 1} | 0.44 |
| 2484 | 3 | 4 | 0.468 | 0 | 3 | 18 | {4: 3} | {4: 3} | 0.0 |
| 1810 | 3 | 4 | 0.455 | 3 | 44 | 19 | {3: 40, 4: 1, 2: 1, 1: 2} | {3: 37, 4: 3, 1: 3, 2: 1} | 0.91 |
| 2323 | 4 | 5 | 0.42 | 5 | 2 | 5 | {5: 1, 2: 1} | {5: 2} | 0.0 |
| 2028 | 3 | 6 | 0.173 | 6 | 1 | 18 | {0: 1} | {0: 1} | 0.0 |
| 2255 | 1 | 3 | 0.178 | 3 | 1 | 20 | {1: 1} | {3: 1} | 1.0 |
| 2533 | 0 | 3 | 0.238 | 0 | 1 | 18 | {3: 1} | {3: 1} | 0.0 |
| 2356 | 0 | 4 | 0.277 | 4 | 5 | 20 | {4: 4, 0: 1} | {0: 3, 4: 2} | 0.2 |

MLP is correct on 25% of the sampled GNN errors.

Error rate by degree bucket:

             mean  count
deg_bucket              
0-1         0.253    178
2-3         0.179    436
4-5         0.122    238
6+          0.209    148

Error rate by neighbourhood homophily:

             mean  count
hom_bucket              
<=0.5       0.516    190
>0.5        0.105    810
