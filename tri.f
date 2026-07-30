yay@login1:~/dp$ for file in $(ls logs/2_rat_bench_*tri_training/*); do echo $file; grep "TRI test debug" $file; done
logs/2_rat_bench_nobart_tri_training/3217266_1.err
logs/2_rat_bench_nobart_tri_training/3217266_1.out
TRI test debug mrr=0.914964 acc=0.863333 total=300
logs/2_rat_bench_nobart_tri_training/3217387_0.err
logs/2_rat_bench_nobart_tri_training/3217387_0.out
TRI test debug mrr=0.890374 acc=0.840000 total=300
logs/2_rat_bench_nobart_tri_training/3228263_4.err
logs/2_rat_bench_nobart_tri_training/3228263_4.out
TRI test debug mrr=0.917309 acc=0.873333 total=300
logs/2_rat_bench_nobart_tri_training/3228333_0.err
logs/2_rat_bench_nobart_tri_training/3228333_0.out
TRI test debug mrr=0.959164 acc=0.933333 total=300
logs/2_rat_bench_nobart_tri_training/3228334_1.err
logs/2_rat_bench_nobart_tri_training/3228334_1.out
TRI test debug mrr=0.916331 acc=0.886667 total=300
logs/2_rat_bench_nobart_tri_training/3228336_2.err
logs/2_rat_bench_nobart_tri_training/3228336_2.out
TRI test debug mrr=0.907796 acc=0.866667 total=300
logs/2_rat_bench_nobart_tri_training/3228338_3.err
logs/2_rat_bench_nobart_tri_training/3228338_3.out
TRI test debug mrr=0.967619 acc=0.943333 total=300
logs/2_rat_bench_tri_training/3217267_1.err
logs/2_rat_bench_tri_training/3217267_1.out
TRI test debug mrr=0.951447 acc=0.930000 total=300
logs/2_rat_bench_tri_training/3217416_0.err
logs/2_rat_bench_tri_training/3217416_0.out
TRI test debug mrr=0.935002 acc=0.910000 total=300
logs/2_rat_bench_tri_training/3228262_4.err
logs/2_rat_bench_tri_training/3228262_4.out
TRI test debug mrr=0.931514 acc=0.900000 total=300
logs/2_rat_bench_tri_training/3228265_0.err
logs/2_rat_bench_tri_training/3228265_0.out
TRI test debug mrr=0.937170 acc=0.906667 total=300
logs/2_rat_bench_tri_training/3228328_1.err
logs/2_rat_bench_tri_training/3228328_1.out
TRI test debug mrr=0.938917 acc=0.906667 total=300
logs/2_rat_bench_tri_training/3228329_2.err
logs/2_rat_bench_tri_training/3228329_2.out
TRI test debug mrr=0.953560 acc=0.933333 total=300
logs/2_rat_bench_tri_training/3228330_3.err
logs/2_rat_bench_tri_training/3228330_3.out
TRI test debug mrr=0.848277 acc=0.783333 total=300


➜  dp git:(master) ✗ python risk_drift_diff_attackers.py groups --group bart data/rat_bench/tri_risk/shap_model_1.jsonl data/rat_bench/tri_risk/shap_model_2.jsonl data/rat_bench/tri_risk/shap_model_3.jsonl data/rat_bench/tri_risk/shap_model_4.jsonl data/rat_bench/tri_risk/shap_model_5.jsonl --group nobart data/rat_bench/tri_risk/shap_nobart_model_1.jsonl data/rat_bench/tri_risk/shap_nobart_model_2.jsonl data/rat_bench/tri_risk/shap_nobart_model_3.jsonl data/rat_bench/tri_risk/shap_nobart_model_4.jsonl data/rat_bench/tri_risk/shap_nobart_model_5.jsonl --alpha 0.05 --permutations 10000 --seed 42 --heatmap-out mds/rat_bench/bart_vs_nobart_heatmap.png --json-out mds/rat_bench/bart_vs_nobart_stats.json
Models: 10 across 2 groups ['bart', 'nobart']
Common records across all models: 300
Records skipped due to offset mismatch (summed across all model pairs): 0

Within-group pairs (MAE): n=20, mean=0.000448, std=0.000101
Cross-group pairs (MAE):  n=25, mean=0.000560, std=0.000061

Group-label permutation test (10000 permutations, seed=42):
  observed mean(cross) - mean(within) = 0.000112
  p-value = 0.009000 (significant at alpha=0.05)

Welch's t-test on within vs cross distances (independence assumption approximate):
  t-statistic = 4.2313, p-value = 0.000207 (significant at alpha=0.05)
  Cohen's d = 1.3023, post hoc power = 0.9888

Heatmap saved to mds/rat_bench/bart_vs_nobart_heatmap.png
Full results saved to mds/rat_bench/bart_vs_nobart_stats.json