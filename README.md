# Flood Warnings Classifier

A classifier that tests whether historic warnings can predict severe flood warnings in England, and reports where the models fail. It was built as a learning project for ML and MLOps.

## Problem and audience

The Environment Agency issues three levels of flood warning: Flood Alert, Flood Warning, and Severe Flood Warning. Severe warnings carry a danger-to-life risk and are rare. Being able to flag them early, or to understand which patterns come before them, would help emergency planners and anyone who needs to judge flood risk in their area.

This project is for people learning applied machine learning and MLOps, and for reviewers who want to see a complete, evaluated workflow on a real public dataset, including a result that doesn't work.

## Method

### Target and split

The target is whether a new warning is severe (1) or another level (0). Only new Flood Alert, Flood Warning, and Severe Flood Warning rows are kept, after normalising case. Training uses warnings up to 2019, and testing uses 2020 onwards, which gives 57 severe warnings in the test years, the most of any cut-off tested.

### Baseline

A model that always predicts "not severe" scores 99.9% accuracy but a severe F1 of 0.0. That's the reference any real model has to beat.

### Handling update rows

The log contains two kinds of row: new warnings and update rows (Update Flood Alert, Update Flood Warning, Update Severe Flood Warning). Update rows record changes to an existing warning, so one flood event can appear several times.

For the current target, update rows are excluded. An update is written after the original warning, so an "Update Severe Flood Warning" row reveals the label the model is trying to predict. Training on it would give the model hindsight and inflate the score.

The data itself is valuable, and this exclusion is a deliberate test choice rather than a judgement that updates are unimportant. Deliberately withholding them tests whether the model can predict severity from information available when a warning is first issued. Earlier updates can still be used as input features, computed only from data before each date, which is the next experiment.

## Known weaknesses

![ROC curve for the weekly warning-count model](results/roc_weekly_count.png)

ROC curve for the weekly warning-count model on the 2020 to 2026 test years. The curve falls below the diagonal (AUC 0.33), so the feature ranks severe warnings below ordinary ones on this split.

- **No feature beats chance so far.** Area and month, area history, and the weekly warning count all scored at or below an AUC of 0.5 on the 2020 to 2026 test years.
- **Few severe examples in the test set.** 57 severe warnings is enough to show a pattern, but not to support a firm conclusion. A single case can move the scores noticeably.
- **Missing drivers.** Severity depends mainly on rainfall and river levels, which aren't in this dataset, so predicting severity from warning history alone is a hard problem.
