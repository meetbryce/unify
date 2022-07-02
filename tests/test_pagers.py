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
        {"strategy": "offsetAndCount", "offset_param":"start", "count_param":"count", "page_size":75}
    )
    assert isinstance(pager2, OffsetAndCountPager)
    assert pager2.page_size == 75

    pager = PagingHelper.get_pager(
        {"strategy": "offsetAndCount", "offset_param":"start", "count_param":"count"}
    )
    assert pager.page_size == 1
    pager = None

    # Use the pager as intended
    for page in range(3):
        params = pager1.get_request_params()        
        assert params["page"] == (page+1)
        assert params["count"] == pager1.page_size
        pager1.next_page(pager1.page_size)

    assert pager1.next_page(pager1.page_size-1) == False 
    
    for page in range(3):
        params = pager2.get_request_params()        
        assert params["start"] == (page * pager2.page_size)
        assert params["count"] == pager2.page_size
        pager2.next_page(pager2.page_size)

    assert pager2.next_page(pager2.page_size-1) == False 
