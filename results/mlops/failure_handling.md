| case | expected | observed |
|---|---|---|
| isolated target node | 200: prediction from own features; isolated=true + warning | 200: - OK |
| duplicate target ids | 422: duplicate_target_ids | 422: duplicate_target_ids OK |
| invalid edge endpoint | 422: invalid_edge_endpoint | 422: invalid_edge_endpoint OK |
| negative edge endpoint | 422: invalid_edge_endpoint | 422: invalid_edge_endpoint OK |
| wrong feature dimension | 422: bad_feature_dim | 422: bad_feature_dim OK |
| unknown target id | 404: unknown_target_id | 404: unknown_target_id OK |
| empty target list | 422: empty_targets | 422: empty_targets OK |
| duplicate + self-loop edges | 200: cleaned silently | 200: - OK |
| no approved model / empty registry | 503: model_unavailable | 503: model_unavailable |
| corrupted weights (checksum mismatch) | 503: model_unavailable | 503: model_unavailable |
