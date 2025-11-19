from typing import Literal

type ScoreOptions = Literal[
    'F',
    'F2',
    'G',
    'recall',
    'precision',
    'mcc',
    'aps'
]

type PredProb = Num[nda, 'features']
type ProbThres = Num[nda, '*#batch']
type PredThres = Bool[nda, '*#batch features']
