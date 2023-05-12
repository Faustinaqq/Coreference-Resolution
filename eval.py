from typing import Set, List


def flatten(l):
    """
    list的嵌套变成长list
    :param l: [[1, 2], [3, 4]]
    :return:  [1, 2, 3, 4]
    """
    return [item for sublist in l for item in sublist]


class BaseCorefMetric:
    def __init__(self, **args):
        pass

    def calculate_p(self, keys: List, responses: List) -> float:
        """
        计算精确率
        """
        raise NotImplementedError

    def calculate_r(self, keys: List, responses: List) -> float:
        """
        计算召回率
        """
        raise NotImplementedError

    def calculate_f(self, keys: List, responses: List) -> float:
        """
        计算F1
        """
        p = self.calculate_p(keys, responses)
        r = self.calculate_r(keys, responses)
        return (2 * p * r) / (p + r)


class MUC(BaseCorefMetric):
    """
    计算MUC评价指标
    ---------------
    ver: 2022-10-25
    by: changhongyu
    """

    def calculate_p(self, keys: List[Set], responses: List[Set]) -> float:
        partitions = []
        for response in responses:
            partition = 0
            un = set()
            for key in keys:
                if response.intersection(key):
                    partition += 1
                    un = un.union(key)
            partition += len(response - un)
            partitions.append(partition)
        numerator = sum([len(response) - partition for response, partition in zip(responses, partitions)])
        denominator = sum([len(response) - 1 for response in responses])
        return numerator / denominator

    def calculate_r(self, keys: List[Set], responses: List[Set]) -> float:
        return self.calculate_p(responses, keys)

# muc = MUC()
# muc.calculate_f(keys, responses)   # 0.4
class B_CUBED(BaseCorefMetric):
    """
    计算B3评价指标
    ---------------
    ver: 2022-10-25
    by: changhongyu
    """

    def calculate_p(self, keys: List, responses: List) -> float:
        numerator = sum([sum([(len(response.intersection(key)) ** 2 / len(response)) for key in keys])
                         for response in responses])
        denominator = sum([len(response) for response in responses])
        return numerator / denominator

    def calculate_r(self, keys: List, responses: List) -> float:
        return self.calculate_p(responses, keys)


class CEAF(BaseCorefMetric):
    """
    计算CEAF指标
    根据自己的理解写了Key和Response之间的匹配规则
    给定一个超参数ratio，当Response中的元素出现在Key中的比率大于ratio认为匹配上
    另外，匹配到某个Response的Key不能匹配给另一个Response
    ---------------
    ver: 2022-10-25
    by: changhongyu
    """

    def __init__(self, ratio: float = 0.5, kind: str = 'e'):
        """
        :param ratio: 超参数，判定Key和Response匹配上的阈值比率
        :param kind: 计算CEAFm指标还是CEAFe指标
        """
        super(CEAF, self).__init__()
        assert 0 < ratio <= 1, AssertionError('Input a ratio between 0 and 1.')
        assert kind in ['m', 'e'], AssertionError('Metric kind must be `e` or `m`.')
        self.ratio = ratio
        self.kind = kind
        self.pairs = None
        self.numerator = None

    def _align(self, keys: List, responses: List):
        """
        对齐Response和Key
        每个Key只能与一个Response对齐
        """
        # 所有符合条件的两两组合
        candidate_pairs = list(filter(lambda x: len(x[0].intersection(x[1])) / len(x[1]) >= self.ratio,
                                      sorted(flatten([[(key, response) for key in keys] for response in responses]),
                                             key=lambda x:
                                             len(x[0].intersection(x[1])) / len(x[1]) + (len(x[1]) + len(x[0])) * 1e-9,
                                             reverse=True)))
        # 按照匹配程度从高到低，为每一个pair中的Response选择对应的Key
        matched_pairs = []
        used_keys = []
        for pair in candidate_pairs:
            for response in responses:
                if response in pair and pair[0] not in used_keys:
                    matched_pairs.append(pair)
                    used_keys.append(pair[0])
                    break
        self.pairs = matched_pairs

    def _calculate_numerator(self):
        """
        为了避免在计算p和r的时候重复计算分子，所以把这个结果临时存储起来
        :return:
        """
        def f4(pair_):
            return 2 * (len(pair_[0].intersection(pair_[1]))) / (len(pair_[0]) + len(pair_[1]))

        assert self.pairs is not None
        if self.kind == 'm':
            self.numerator = sum([len(pair[0].intersection(pair[1])) for pair in self.pairs])
        else:
            self.numerator = sum([f4(pair) for pair in self.pairs])

    def _calculate_m(self, keys_or_responses: List) -> float:
        """
        计算CEAFm的精确率或召回率
        当传入Responses，计算的是精确率
        当传入Keys，计算的是召回率
        """
        if not self.numerator:
            self._calculate_numerator()
        return self.numerator / sum([len(k_or_r) for k_or_r in keys_or_responses])

    def _calculate_e(self, keys_or_responses: List) -> float:
        """
        计算CEAFe的精确率
        当传入Responses，计算的是精确率
        当传入Keys，计算的是召回率
        """
        if not self.numerator:
            self._calculate_numerator()
        return self.numerator / len(keys_or_responses)

    def calculate_p(self, keys: List, responses: List) -> float:
        if self.kind == 'm':
            return self._calculate_m(responses)
        elif self.kind == 'e':
            return self._calculate_e(responses)
        else:
            raise ValueError(self.kind)

    def calculate_r(self, keys: List, responses: List) -> float:
        return self.calculate_p(responses, keys)

    def calculate_f(self, keys: List, responses: List) -> float:
        """
        计算完f值后，要把pairs置为None，以便下次计算
        """
        if self.pairs is None:
            self._align(keys, responses)
        res = super(CEAF, self).calculate_f(keys, responses)
        self.pairs = None
        self.numerator = None
        return res
