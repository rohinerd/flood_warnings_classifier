# Flood Warnings Classifier

A classifier that predicts severe flood warnings in England from the Environment Agency's historic warnings log. It was built as a learning project for ML and MLOps, and it reports its results honestly, including where the models fail.

## Problem and audience

The Environment Agency issues three levels of flood warning: Flood Alert, Flood Warning, and Severe Flood Warning. Severe warnings carry a danger-to-life risk and are rare. Being able to flag them early, or to understand which patterns come before them, would help emergency planners and anyone who needs to judge flood risk in their area.

This project is for people learning applied machine learning and MLOps, and for reviewers who want to see a complete, evaluated workflow on a real public dataset, including a result that doesn't work.

[ROC curve for the weekly warning-count model](results/roc_weekly_count.png)

ROC curve for the weekly warning-count model on the 2020 to 2026 test years. The curve falls below the diagonal (AUC 0.33), so the feature ranks severe warnings below ordinary ones on this split.
