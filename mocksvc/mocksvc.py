import os
import requests_mock
import json
import random

class MockSvc:
    @classmethod
    def setup_mocksvc_api(cls, mock):
        mock.get("https://mocksvc.com/api/ping", text='pong')

        # Load json relative to current file
        fpath1 = os.path.join(os.path.dirname(__file__), "json_data", "page1-100.json")
        rows100 = open(fpath1).read()
        fpath2 = os.path.join(os.path.dirname(__file__), "json_data", "page2-27.json")
        rows27 = open(fpath2).read()

        # auth matches: scott@example.com:abc123
        auth_header = {"Authorization": "Basic c2NvdHRAZXhhbXBsZS5jb206YWJjMTIz"}

        mock.get("https://mocksvc.com/api/repos_27", text=rows27, request_headers=auth_header)
        mock.get("https://mocksvc.com/api/repos_100", text=rows100, request_headers=auth_header)
        mock.get("https://mocksvc.com/api/repos_100?page=2", text="[]", request_headers=auth_header)

        records = json.loads(rows100)

        mock.get(f"https://mocksvc.com/api/repos_1100", text=rows100, request_headers=auth_header)

        for page in range(1, 12):
            # randomize the id's because they are supposed to be unique (but we are re-using
            # the same 100 records)
            for r in records:
                r['id'] = random.randint(0, 999999)
            body = json.dumps(records)
            if page == 11:
                body = rows27
            mock.get(f"https://mocksvc.com/api/repos_1100?page={page}&count=100", 
                text=body, request_headers=auth_header)
                    
        for page in range(1, 4):
            body = rows27
            if page == 3:
                body = "[]"
        mock.get(f"https://mocksvc.com/api/repos_81?page={page}&count=27", 
                text=body, request_headers=auth_header)


    