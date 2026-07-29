Models: 12 across 2 groups ['bart', 'nobart']
Common records across all models: 300
Records skipped due to offset mismatch (summed across all model pairs): 0

Within-group pairs (MAE): n=30, mean=0.000493, std=0.000108
Cross-group pairs (MAE):  n=36, mean=0.000676, std=0.000046
Within-group pairs by group (MAE):
  bart: n=15, mean=0.000595, std=0.000038
  nobart: n=15, mean=0.000390, std=0.000031

Within-group pairs (RMSE): n=30, mean=0.000970, std=0.000229
Cross-group pairs (RMSE):  n=36, mean=0.001280, std=0.000154
Within-group pairs by group (RMSE):
  bart: n=15, mean=0.001172, std=0.000122
  nobart: n=15, mean=0.000768, std=0.000093

Within-group pairs (MedAE): n=30, mean=0.000284, std=0.000065
Cross-group pairs (MedAE):  n=36, mean=0.000383, std=0.000021
Within-group pairs by group (MedAE):
  bart: n=15, mean=0.000347, std=0.000013
  nobart: n=15, mean=0.000220, std=0.000013

Within-group pairs (Spearman): n=30, mean=0.435815, std=0.020844
Cross-group pairs (Spearman):  n=36, mean=0.186895, std=0.044331
Within-group pairs by group (Spearman):
  bart: n=15, mean=0.424989, std=0.016502
  nobart: n=15, mean=0.446641, std=0.019032

Within-group pairs (Jaccard@5): n=30, mean=0.075410, std=0.016368
Cross-group pairs (Jaccard@5):  n=36, mean=0.028485, std=0.010247
Within-group pairs by group (Jaccard@5):
  bart: n=15, mean=0.084422, std=0.014461
  nobart: n=15, mean=0.066397, std=0.012815

Group-label permutation test on MAE (exact, 924 label assignments enumerated):
  observed mean(cross) - mean(within) = 0.000183
  p-value = 0.002165 (significant at alpha=0.05; minimum achievable p-value at this group size = 0.002165)

Group-label permutation test on Spearman (exact, 924 label assignments enumerated):
  observed mean(cross) - mean(within) = -0.248920
  p-value = 0.002165 (significant at alpha=0.05; minimum achievable p-value at this group size = 0.002165)

Group-label permutation test on Jaccard@5 (exact, 924 label assignments enumerated):
  observed mean(cross) - mean(within) = -0.046925
  p-value = 0.002165 (significant at alpha=0.05; minimum achievable p-value at this group size = 0.002165)

Heatmap saved to mds/rat_bench/bart_vs_nobart_heatmap.png
t-SNE plot saved to mds/rat_bench/bart_vs_nobart_tsne.png
Full results saved to mds/rat_bench/bart_vs_nobart_stats.json