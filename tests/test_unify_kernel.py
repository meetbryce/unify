from unify_kernel import UnifyKernel
import pytest

@pytest.fixture
def kernel():
    return UnifyKernel()

def test_autocomplete(kernel):
    unify = kernel.unify_runner
    assert unify is not None

    def check_query(code, expected):
        res = kernel.do_complete(code, len(code))
        assert "matches" in res
        assert set(res["matches"]) == set(expected)

    check_query("show ", ["tables", "schemas", "columns from "])
    check_query("show tables from ", unify._list_schemas())

