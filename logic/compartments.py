class Compartments(object):
    """
    Class used to represent an Compartments disease model
    :Example:
        Compartments(name='susceptible', value=0.0)
    """

    def __init__(self, name=None, value=None, result=None):
        """ Compartments constructor object.
        :param name: name of compartment.
        :type name: str
        :param value: value of compartment.
        :type value: float
        :param result: population by compartment.
        :type result: List
        :returns: Compartments object
        :rtype: object
        """
        self._name = 'default_name' if name is None else name
        self._value = 0.0 if value is None else value
        self._result = 0.0 if result is None else result

    @property
    def name(self):
        """
        Returns the name of the compartment.
        :returns: name of compartment.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """
        The name of the compartment is assigned.
        :param name: name of compartment.
        :type: str
        """
        self._name = name

    @property
    def result(self):
        """
        Returns the result of the compartment.
        :returns: population by compartment.
        :rtype: List
        """
        return self._result

    @result.setter
    def result(self, val: None):
        """
        The result of the compartment is assigned.
        :param val: result of compartment.
        :type: list
        """
        self._result = val

    @property
    def value(self):
        """
        Returns the value of the compartment.
        :returns: value by compartment.
        :rtype: float
        """
        return self._value

    @value.setter
    def value(self, val: float):
        """
        The value of the compartment is assigned.
        :param val: value of compartment.
        :type: float
        """
        self._value = val

    def __str__(self):
        """
        Returns all values of the compartment.
        :returns: to_string Compartments object
        :rtype: str
        """
        result = {}
        for key, var in vars(self).items():  # Iterate over the values
            result.update({key: var})
        return result
