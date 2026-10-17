# Model artifacts

The existing `.pt` files in this directory are exploratory local CPU training
artifacts from before the Kaggle-only training requirement. They are not the
final model and must not be used for deployment or reported as Kaggle results.

The best current market-parameterized candidate comes from Kaggle run v2 and is
stored as `market_parametric_pinn_kaggle.pt`. Its chronological holdout and
selection evidence are in `reports/kaggle_training_verification.json` and
`reports/kaggle_model_selection.json`. The later maturity-balanced Kaggle run
was rejected because it performed worse. The candidate is not yet an accepted
production model because its acceptance tolerance was selected after observing
the holdout.
