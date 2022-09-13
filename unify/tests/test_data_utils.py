from unify.data_utils import dict_deep_transform, interp_dollar_values

def test_dict_transform():
    d1 = {"foo": "bar",
            "nested": {
                "n1": "val1",
                "n2": "val2",
                "list1": [
                    {"nn1": "val11"},
                    {"nn2": "val22"}
                ]
            },
            "alist": ["val5", "val6"]
        }

    comp = {"foo": "bar",
            "nested": {
                "n1": "ZZ1",
                "n2": "ZZ2",
                "list1": [
                    {"nn1": "ZZ11"},
                    {"nn2": "ZZ22"}
                ]
            },
            "alist": ["ZZ5", "ZZ6"]
        }

    d2 = dict_deep_transform(d1, lambda k, v: v.replace("val","ZZ"))

    assert d2 == comp

    d3 = dict_deep_transform(d1, key_xform=lambda k, v: k.upper())

    assert d3 == {"FOO": "bar",
            "NESTED": {
                "N1": "val1",
                "N2": "val2",
                "LIST1": [
                    {"NN1": "val11"},
                    {"NN2": "val22"}
                ]
            },
            "ALIST": ["val5", "val6"]
        }

def test_interp_values():
    d1 = {"foo": "${bar}",
            "nested": {
                "n1": "george ${last_name}",
                "n2": "${dog} runs",
                "list1": [
                    {"nn1": "${p1} and ${p2}"},
                ]
            },
            "alist": ["${color}", "${day}"]
        }
    d2 = interp_dollar_values(d1, 
        {"bar": "soap",
         "last_name": "bush",
         "dog": "Spot",
         "p1": "Rick", "p2": "Morty",
         "color": "green",
         "day": "monday"        
         }
    )
    assert d2 == {"foo": "soap",
            "nested": {
                "n1": "george bush",
                "n2": "Spot runs",
                "list1": [
                    {"nn1": "Rick and Morty"},
                ]
            },
            "alist": ["green", "monday"]
        }
