import pytest
from rest_schema import PagingHelper, PageAndCountPager, OffsetAndCountPager, NullPager

def test_pagers():
    assert isinstance(PagingHelper.get_pager(None), NullPager)
    assert isinstance(PagingHelper.get_pager({}), NullPager)
    
    with pytest.raises(RuntimeError):
        PagingHelper.get_pager(
            {"strategy": "pageAndCount", "page_size":25}
        )

    pager = PagingHelper.get_pager(
            {"strategy": "pageAndCount", "page_size":25, "page_param": "page", "count_param" : "count"}
        )

    assert isinstance(pager, PageAndCountPager)
    assert pager.page_size == 25

    pager = PagingHelper.get_pager(
        {"strategy": "offsetAndCount", "offset_param":"start", "count_param":"count", "page_size":100}
    )
    assert isinstance(pager, OffsetAndCountPager)
    assert pager.page_size == 100

    pager = PagingHelper.get_pager(
        {"strategy": "offsetAndCount", "offset_param":"start", "count_param":"count"}
    )
    assert pager.page_size == 1
