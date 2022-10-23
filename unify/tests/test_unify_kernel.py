from unify_kernel import UnifyKernel
import pytest

@pytest.fixture
def kernel() -> UnifyKernel:
    return UnifyKernel()


def test_basic_kernel(kernel: UnifyKernel):
    res = kernel.do_execute("show schemas", silent=True)
    assert res['status'] == 'ok'

    res = kernel.do_execute("select * from information_schema.tables", silent=True)
    assert res['status'] == 'ok'


@pytest.mark.skip("autocomplete needs to be rebuilt")
def test_autocomplete(kernel):
    unify = kernel.unify_runner
    assert unify is not None

    def check_query(code, expected):
        res = kernel.do_complete(code, len(code))
        assert "matches" in res
        assert set(res["matches"]) == set(expected)

    check_query("show ", ["tables", "schemas", "columns from "])
    check_query("show tables from ", unify._list_schemas())
    check_query("show tables from ", unify._list_schemas())
    check_query("show tables from in", unify._list_schemas("in"))
    sch1 = unify._list_schemas()['schema_name'][0]
    
    # FIXME: Autocomplete needs new grammar definitions
    #st_tables_filtered(sch1, ""))
    #table1 = unify._list_tables_filtered(sch1, "")[0][0:3]
    #check_query(f"show columns from {sch1}.{table1}", unify._list_tables_filtered(sch1, table1))



