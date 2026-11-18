# CV Bullets

- Implemented a PyTorch CTR training pipeline that combines DCN-V2-style cross
  layers with DIN, DIEN, or BST sequence-interest encoders.
- Built bounded-memory behaviour-sequence processing with 50-event truncation
  and 262,144-bucket hashing, alongside train-partition categorical mappings.
- Evaluated retained CTR experiments with stratified validation, ROC-AUC, and
  PR-AUC; the best recorded AUC was 0.7417 and the best recorded PR-AUC was
  0.0792 in separate configurations.
- Reported a no-cross-tower DIN comparison (validation ROC-AUC 0.7412,
  PR-AUC 0.0779) showing that, in single-seed runs, the DCN-V2 cross tower
  added no clear gain over the sequence encoder alone.

Do not describe these results as a production deployment, external test score,
or competition placement without additional evidence.
