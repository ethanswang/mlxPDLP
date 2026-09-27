"""CSR indices must be validated before any narrowing conversion."""

import numpy as np
import pytest

import mlxpdlp


@pytest.fixture(params=["solver", "plan"])
def make_model(request):
    p = mlxpdlp.Parameters()
    p.presolve = False
    p.verbose = False

    def make(row_ptr, col_indices, n=2, m=1):
        args = dict(num_variables=n, num_constraints=m, row_ptr=row_ptr,
                    col_indices=col_indices, values=np.ones(col_indices.size),
                    parameters=p, device="cpu")
        if request.param == "solver":
            return mlxpdlp.Solver(**args, objective=np.ones(2))
        return mlxpdlp.SharedMatrixPlan(**args)

    return make


@pytest.mark.parametrize("field", ["row_ptr", "col_indices"])
@pytest.mark.parametrize("dtype,value", [
    (np.int64, 2**31),
    (np.int64, 2**32 + 1),  # Wraps to a plausible index/offset.
    (np.int64, 2**63 - 1),
    (np.int64, -1),
    (np.int64, -(2**32) + 1),
    (np.int64, -(2**63)),
    (np.uint32, 2**32 - 1),
    (np.uint64, 2**32 + 1),
    (np.uint64, 2**63 + 1),
    (np.uint64, 2**64 - 1),
])
def test_sparse_indices_reject_out_of_range(make_model, field, dtype, value):
    row_ptr = np.array([0, 1], dtype=np.int64)
    col_indices = np.array([0], dtype=np.int64)
    if field == "row_ptr":
        row_ptr = np.array([0, value], dtype=dtype)
    else:
        col_indices = np.array([value], dtype=dtype)
    with pytest.raises(ValueError, match=field + ".*INT32_MAX"):
        make_model(row_ptr, col_indices)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16,
                                  np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("strided", [False, True])
def test_sparse_indices_accept_integer_dtypes(make_model, dtype, strided):
    row_ptr = np.array([0, 2], dtype=dtype)
    col_indices = np.array([0, 1], dtype=dtype)
    if strided:
        row_ptr = np.repeat(row_ptr, 2)[::2]
        col_indices = np.repeat(col_indices, 2)[::2]
    row_ptr.setflags(write=False)
    col_indices.setflags(write=False)
    assert make_model(row_ptr, col_indices) is not None


@pytest.mark.parametrize("field", ["row_ptr", "col_indices"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128, np.bool_])
def test_sparse_indices_require_integer_dtype(make_model, field, dtype):
    arrays = dict(row_ptr=np.array([0, 1], dtype=np.int32),
                  col_indices=np.array([0], dtype=np.int32))
    arrays[field] = arrays[field].astype(dtype)
    with pytest.raises(TypeError, match=field + ".*integer"):
        make_model(**arrays)


@pytest.mark.parametrize("dimension", ["n", "m"])
def test_sparse_dimensions_reject_overflow(make_model, dimension):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        make_model(np.array([0, 1], dtype=np.int32),
                   np.array([0], dtype=np.int32), **{dimension: 2**32 + 1})
