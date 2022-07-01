import pytest
from rest_schema import PagingHelper, PageAndCountPager, OffsetAndCountPager, NullPager

def test_pagers():
    assert isinstance(PagingHelper.get_pager(None), NullPager)
    assert isinstance(PagingHelper.get_pager({}), NullPager)
    
    with pytest.raises(RuntimeError):
        PagingHelper.get_pager(
            {"strategy": "pageAndCount", "page_size":25}
        )

    pager1 = PagingHelper.get_pager(
            {"strategy": "pageAndCount", "page_size":25, "page_param": "page", "count_param" : "count"}
        )

    assert isinstance(pager1, PageAndCountPager)
    assert pager1.page_size == 25

    pager2 = PagingHelper.get_pager(
        {"strategy": "offsetAndCount", "offset_param":"start", "count_param":"count", "page_size":100}
    )
    assert isinstance(pager2, OffsetAndCountPager)
    assert pager2.page_size == 100

    pager = PagingHelper.get_pager(
        {"strategy": "offsetAndCount", "offset_param":"start", "count_param":"count"}
    )
    assert pager.page_size == 1

    # Use the pager as intended
    for page in range(3):
        params = pager1.get_request_params(page)        
        assert params["page"] == page
        assert params["count"] == pager.page_size

