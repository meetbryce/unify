import pytest
from unify.rest_adapter import (
    PagingHelper, 
    PageAndCountPager, 
    OffsetAndCountPager, 
    NullPager, 
    PagerTokenPager
)

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

    pager3 = PagingHelper.get_pager(
        {"strategy": "offsetAndCount", "offset_param":"start", "count_param":"count"}
    )
    assert pager3.page_size == 1

    ####
    pager = None

    # Use the pager as intended
    for page in range(3):
        params = pager1.get_request_params()        
        assert params["page"] == (page+1)
        assert params["count"] == pager1.page_size
        pager1.next_page(pager1.page_size, [])

    assert pager1.next_page(pager1.page_size-1, []) == False 
    
    for page in range(3):
        params = pager2.get_request_params()        
        assert params["start"] == (page * pager2.page_size)
        assert params["count"] == pager2.page_size
        pager2.next_page(pager2.page_size, [])

    assert pager2.next_page(pager2.page_size-1, []) == False 

def test_token_pager():
    # Token pager needs the actual result to work

    pager = PagingHelper.get_pager(
        {"strategy": "pagerToken", 
        "pager_token_path": "results[*].subd.nextPage", 
        "count_param":"limit",
        "token_param":"next",
        "page_size":25
        }
    )
    assert isinstance(pager, PagerTokenPager)
    assert pager.page_size == 25

    page1 = {
        "results": [
            {
                "fields": {"field1": "one", "field2": "two"},
                "subd": {"nextPage": "page2"}
            },
            {
                "fields": {"field1": "one", "field2": "two"},
            },
        ]
    }
    page2 = {
        "results": [
            {
                "fields": {"field1": "one", "field2": "two"},
            }
        ]
    }

    assert pager.get_request_params() == {"limit": 25}
    assert pager.next_page(25, page1) == True
    print(pager.get_request_params())
    assert pager.get_request_params() == {"next": "page2", "limit": 25}
    assert pager.next_page(25, page2) == False

    

