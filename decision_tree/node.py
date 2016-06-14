from features import get_feature_name

class Node:
    def __init__(self, type, value):
        self.__children = []
        self.__type = type
        self.__value = value

    @property
    def children(self):
        return self.__children

    @property
    def type(self):
        return self.__type

    @property
    def value(self):
        return self.__value

    def append_child(self, node):
        self.__children.append(node)

    def _get_call_name(self):
        return self.value

    def __str__(self, depth = 0):
        str = " " * 2 * depth
        if depth > 0:
            str += "|__ "
        str += self._get_call_name()
        for child in self.children:
            str += "\n" + child.__str__(depth + 1)
        return str


class FeatureNode(Node):
    def __init__(self, value):
        super().__init__("feature", value)

    def _get_call_name(self):
        return "<%s>" % get_feature_name(self.value)


class ValueNode(Node):
    def __init__(self, value):
        super().__init__("value", value)


class ResultNode(Node):
    def __init__(self, value):
        super().__init__("result", value)

    def _get_call_name(self):
        return " = %s" % self.value
