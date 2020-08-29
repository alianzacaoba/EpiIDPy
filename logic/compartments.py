class Compartments(object):

    def __init__(self, name=None, value=None, result=None):
        """
            Init compartments object with parameters.

        Parameters
        ----------
            name : str
                name of compartment.
            value : float
                value of compartment.
            result : list
                population by compartment.
        Returns
        -------
            object
                object of compartment.
        """
        self._name = 'default_name' if name is None else name
        self._value = 0.0 if value is None else value
        self._result = 0.0 if result is None else result

    @property
    def name(self):
        """
        Returns the name of the compartment.

        Returns
        --------
            str
                name of compartment.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """
        The name of the compartment is assigned.

        Parameters
        ----------
            name : str
                name of compartment.
        """
        self._name = name

    @property
    def result(self):
        """
        Returns the result of the compartment.

        Returns
        --------
            list
                population by compartment.
        """
        return self._result

    @result.setter
    def result(self, val: None):
        """
        The result of the compartment is assigned.

        Parameters
        ----------
            val : list
                result of compartment.
        """
        self._result = val

    @property
    def value(self):
        """
        Returns the value of the compartment.

        Returns:
        ----------
            float
                value by compartment.
        """
        return self._value

    @value.setter
    def value(self, val: float):
        """
        The value of the compartment is assigned.

        Parameters
        ----------
            val : float
                value of compartment.
        """
        self._value = val

    def __str__(self):
        """
        Returns all values of the compartment.

        Returns
        -------
            str
                values by compartment.
        """
        result = {}
        for key, var in vars(self).items():  # Iterate over the values
            result.update({key: var})
        return result
