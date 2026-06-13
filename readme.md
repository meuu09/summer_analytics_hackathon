# Summer Analytics 2026 — E-Commerce Conversion Prediction

**Hackathon:** Summer Analytics 2026, Week 2  
**Task:** Binary classification — predict user conversion  
**Metric:** F1 Score  
**CV F1 Score:** ~0.565

## Approach
- Merged train + public_test (13,000 labeled rows)
- 12 engineered features (engagement, loyalty, log transforms)
- Logistic Regression with class_weight='balanced', C=0.1
- Stratified 5-Fold CV with threshold tuning (~0.43)

## Files
- `notebook.ipynb` — full reproducible pipeline
- `submission.csv` — predictions for private_test.csv
- `report.pdf` — one-page methodology summary