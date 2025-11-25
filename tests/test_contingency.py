import numpy as np
from hypothesis import given, settings, example
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import pytest
from typing import get_args
# from jaxtyping import install_import_hook

# with install_import_hook("contingency", "beartype.beartype"):
#     from contingency import Contingent

from contingency import Contingent
from contingency.contingent import ScoreOptions

# import warnings
# warnings.filterwarnings("error", category=UserWarning)


@st.composite  
def make_shapes(draw):
    return draw(hnp.array_shapes(max_dims=2, min_dims=1, min_side=5))


@st.composite
def make_bools(draw, shape=(1,5)):
    arr = draw(hnp.arrays(
        bool,
        shape,
        elements=st.just(True), fill=st.just(False),
    ))
    return arr


@st.composite
def make_true_pred(draw):
    shape = draw(make_shapes())
    y_true = draw(make_bools(shape=shape[-1]))
    y_pred = draw(make_bools(shape=shape))
    return (y_true, y_pred)

@example((np.array([0,1,0,1]),np.array([0,1,1,0]))).xfail() #ints aren't bools
@example((
    np.array([False, True, False]),
    np.array([False, True]))
).xfail()  #non-matching shapes
@given(make_true_pred())
def test_scoring(y_Y):
    y_true, y_pred = y_Y
    M = Contingent(y_true, y_pred)  # jaxtyped
    assert M.mcc.dtype == 'float'   # collapsed to scalar
    # assert (M.F <= M.G).all()      # harm. mean <= geom. mean


@st.composite
def make_true_prob(draw):
    shape = draw(hnp.array_shapes(max_dims=1, min_dims=1, min_side=5))
    y_true = draw(make_bools(shape=shape))
    y_pred = draw(hnp.arrays(float, shape, elements=st.floats(0,1)))
    return (y_true, y_pred)

@example((np.array([0,1,0,1]),None)) # when the algorithm's got nothin' 
@given(make_true_prob())
def test_from_scalar(y_Y):
    y_true, y_pred = y_Y
    if y_pred is None:
        with pytest.warns(UserWarning, match="`None` value recieved, passing the buck..."):
            M = Contingent.from_scalar(y_true, y_pred) 
        assert M is None
    else:
        M = Contingent.from_scalar(y_true, y_pred) 
        assert M.mcc.dtype == 'float'   # collapsed to scalar
        np.testing.assert_array_less(M.F, M.G+1e-5)      # harm. mean <= geom. mean



@given(
    make_true_prob(),
    st.sampled_from(get_args(ScoreOptions))
)
def test_expected(y_Y, mode):
    y_true, y_pred = y_Y
    M = Contingent.from_scalar(y_true, y_pred) 

    assert isinstance(M.expected(mode), float)
