# ML Assignment 2: Bagging, Boosting and Bias-Variance on Olivetti Faces

**Name:** Sneha   
**Seed:** 242010

## Setup
Tested with Python 3.9.6.

```bash
git clone https://github.com/SnehaLakhlan/ML_Assignment_2.git
cd ML-ASS_2
pip install -r requirements.txt
```

## Run
```bash
python3 main.py
```

This runs `data_setup.py`, `question_1.py`, `question_2.py` and `question_3.py` in order and saves their outputs to `logs/`. The dataset downloads automatically on the first run.

## Files
- `data_setup.py`: loads the dataset, creates the 6/2/2 split per identity, and builds the masked test set
- `question_1.py`: decision tree baseline and bagging
- `question_2.py`: SAMME AdaBoost and the bagging vs boosting comparison
- `question_3.py`: bias and variance over 30 resampled training sets
- `data/`: saved split (`split_indices.npz`, `split_indices.csv`)
- `configs/`: selected parameters from Q1 and Q2, and the Q3 resampled sets
- `logs/`: final logs
- `figures/`: all figures used in the report

## Reproducibility
- Split: `numpy.random.default_rng(242010)`
- Bagging with B trees: `numpy.random.default_rng([242010, B])`
- Q3 resampling: `numpy.random.default_rng([242010, 3])`
- All decision trees: `random_state=242010`