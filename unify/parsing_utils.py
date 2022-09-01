from gettext import find
from lark.tree import Tree

def find_subtree(rule, tree):
    found = None
    if getattr(getattr(tree, 'data', {}), 'value', None) == rule:
        found = tree
    for node in getattr(tree, 'children', []):
        if getattr(getattr(node, 'data', {}), 'value', None) == rule or \
            getattr(node, 'type', None) == rule:
            found = node
            break
        else:
            res = find_node_return_children(rule, node)
            if res:
                return res
    return found

def find_node_return_children(rule, tree):
    found = None
    if isinstance(rule, list):
        node = tree
        for idx, r in enumerate(rule):
            node = find_subtree(r, node)
            if node is None:
                break
        if idx == len(rule) - 1:
            found = node
    else:
        found = find_subtree(rule, tree)
    if isinstance(found, list):
        return found
    elif found:
        return [(child if isinstance(child, Tree) else child.value) for child in found.children]
    else:
        return None

def find_node_return_child(rule, tree):
    children = find_node_return_children(rule, tree)
    if children and len(children) > 0:
        return children[0]
    else:
        return None

def collect_strings(node):
    if isinstance(node, Tree):
        res = []
        for c in node.children:
            res.extend(collect_strings(c))
        return res
    elif isinstance(node, list):
        return node
    else:
        return [node.value]

def collect_child_string_list(rule, tree):
    st = find_subtree(rule, tree)
    if st:
        return collect_strings(st)
    else:
        return []

def collect_child_strings(rule, tree):
    st = find_subtree(rule, tree)
    if st:
        return " ".join(collect_strings(st))
    else:
        return None

def collect_child_text(rule: str, tree: Tree, full_code: str) -> str:
    st = find_subtree(rule, tree)
    if st:
        return full_code[st.meta.start_pos:st.meta.end_pos]
    else:
        return None

