Susceptible-Infectious-Recovered Python (SIRPy)
==============================================
The Susceptible-Infectious-Recovered Python (SIRPy) is a framework 
designed to help scientists and public health officials create and 
use models of emerging infectious diseases. This framework uses 
mathematical models of diseases (based on differential equations) 
to simulate the development or evolution of a disease in time. 
Also, the framework is founded on an object-oriented approach in 
order to optimize the simulation processes of the models. 
The following illustrates and describes the process of creating a 
SIRPy type of model.


![](./media/BPM_SIRPy.png){:height="50%" width="50%"}

To create a new model, you must instantiate the base object 
DiseaseModel and implement the equations method and the main 
entry point main. For the function of the equation, the differential 
equations that the model represents are defined, for example:

![](./media/equations.png)

In relation to the main function, the following activities must be carried out:
1.	Create the compartments that represent the model, defining instances of Compartments type objects, for example:
        susc = Compartments(name='susceptible')
        expo = Compartments(name='exposed')
        inf = Compartments(name='infectious')
        rec = Compartments(name='recovered')
2.	Then the compartments type objects are assigned to a list.

