# 🚀 Project Progress Log: Hybrid Defect Detection

This log tracks key milestones and results for the Hybrid CNN-ViT Defect Detection Project.

## 18–19 Nov (Day 3)
- Created **ViT_Training** notebook
- Fixed DataModule structural and loading errors to use full dataset.
- Loaded **ViT-Small** with frozen backbone and trained for **7 epochs** (Model 2 created)
- Saved model weights to Drive (`vit_day3.ckpt`)
- Ran `trainer.test()` for unbiased metrics (Test Acc: 81.47%, Class 1 F1: 0.0)
- Logged accuracy/loss/class-wise results to Drive (`test_results_vit_day3.json`)
- Uploaded notebook and DataModule fix to GitHub
