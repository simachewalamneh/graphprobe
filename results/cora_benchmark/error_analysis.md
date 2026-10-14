# Error analysis (GAT, seed 0)

Rows: 4 most confident mistakes + 4 random mistakes.

| node | true | pred | conf | MLP pred | degree | words | neighbour true labels | neighbour predictions | nbr homophily |
|---|---|---|---|---|---|---|---|---|---|
| 1764 | 5 | 2 | 0.168 | 5 | 2 | 21 | {5: 1, 2: 1} | {2: 2} | 0.5 |
| 2168 | 1 | 2 | 0.163 | 2 | 8 | 21 | {1: 2, 2: 6} | {2: 8} | 0.25 |
| 2484 | 3 | 4 | 0.16 | 2 | 3 | 18 | {4: 3} | {4: 3} | 0.0 |
| 2607 | 3 | 2 | 0.16 | 3 | 1 | 5 | {2: 1} | {2: 1} | 0.0 |
| 2032 | 3 | 4 | 0.154 | 3 | 1 | 19 | {4: 1} | {4: 1} | 0.0 |
| 2208 | 1 | 2 | 0.151 | 5 | 1 | 23 | {2: 1} | {2: 1} | 0.0 |
| 2534 | 3 | 4 | 0.154 | 3 | 5 | 4 | {3: 5} | {4: 3, 3: 2} | 1.0 |
| 2326 | 1 | 5 | 0.147 | 5 | 4 | 21 | {5: 2, 1: 2} | {5: 2, 1: 1, 3: 1} | 0.5 |

MLP is correct on 50% of the sampled GNN errors.

Error rate by degree bucket:

             mean  count
deg_bucket              
0-1         0.309    178
2-3         0.264    436
4-5         0.197    238
6+          0.243    148

Error rate by neighbourhood homophily:

             mean  count
hom_bucket              
<=0.5       0.611    190
>0.5        0.169    810
