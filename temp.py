from deepdiff import DeepDiff

obj1 = {"a": 1, "b": 2, "c": [1, 2]}
obj2 = {"a": 1, "b": 2, "c": [1, 2, 3]}

diff = DeepDiff(obj1, obj2, ignore_order=True)
assert not diff, diff
