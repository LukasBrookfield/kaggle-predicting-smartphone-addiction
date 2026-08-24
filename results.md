| Model | CV score | CV length | Notes |
|---|---|---|---|
CatBoost Baseline | 0.9627888433905024 | full | Previous best model. |
XGBoost Baseline | 0.95764 | quick | Screening reference, capped at 100 trees so underfit. |
XGBoost + total_screen | 0.95743 | quick | Below baseline; 0.00101 fold spread swamps the effect. |
XGBoost + d_plus_2s | 0.95733 | quick | Worst of the three sum features; reject. |
XGBoost + avg_screen | 0.95752 | quick | Identical fold AUCs is coincidence, still below baseline. |
XGBoost + social_media_ratio | 0.95766 | quick | Dead even with baseline (+0.00002); no effect. |
XGBoost + portion_of_day_on_screen | 0.95760 | quick | No effect (-0.00004). |
XGBoost + screen_to_work_ratio | 0.95785 | quick | Tightest folds of the sweep (std 0.00004) but gain under noise. |
XGBoost + leisure_ratio | 0.95804 | quick | Best of the 10, but +0.00040 is below the best-of-10 noise bar. |
XGBoost + screen_share_of_free | 0.95755 | quick | Largest fold spread (0.00127) in the sweep; unreliable. |
XGBoost + mins_per_open | 0.95731 | quick | Consistently the worst ratio feature. |
XGBoost + notif_per_screen_hour | 0.95747 | quick | No gain; notifications are a weak signal on this dataset. |
XGBoost + iterative-imputed cols | 0.95824 | quick | Only screening winner (+0.00060); confirmed on full CV below. |
XGBoost + original data | 0.95758 | quick | Original rows in train folds only; folds disagree on sign, so noise. |
XGBoost + original data + is_original flag | 0.95709 | quick | Flag proxies "no NaNs" and is constant at test time; hurts. |
XGBoost full (1000 trees, lr 0.1) | 0.96391 | full | best_iter hit the 1000-tree cap, so this is a floor not a converged score. |
XGBoost full + iterative-imputed cols | 0.96456 | full | +0.00065 on all 5 folds, p=0.0006; keep, and port to CatBoost. |
