from logic.compartments import Compartments


class Transitions(object):
    """Class used to represent an transitions by compartment"""
    def __init__(self, rate=0.0, probability=0.0, org=Compartments(), dest=Compartments()):
        self._rate = rate
        self._probability = probability

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate: float):
        self._rate = rate

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, probability: float):
        self._probability = probability

    @property
    def value(self):
        if self._rate >= 0.0:
            return self._probability
        else:
            return self._rate * self._probability

    def __str__(self):
        # toString()
        output = {}
        for key, var in vars(self).items():  # Iterate over the values
            output.update({key: var})
        return output.__str__()
