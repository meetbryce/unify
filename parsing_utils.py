def find_node_return_children(rule, tree):
    found = None
    if getattr(getattr(tree, 'data', {}), 'value', None) == rule:
        found = tree
    for node in getattr(tree, 'children', []):
        if getattr(getattr(node, 'data', {}), 'value', None) == rule:
            found = node
            break
        else:
            res = find_node_return_children(rule, node)
            if res:
                return res
    if found:
        return [child.value for child in found.children]
    else:
        return None

