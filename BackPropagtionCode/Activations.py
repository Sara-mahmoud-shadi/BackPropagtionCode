
import math

def sigmoid(net,a=1):
    return (1/(1+math.exp(-a*net)))


def Hyperbolic_Tangent(net,b=1,a=1):
    return a* (math.tanh(b*net))


def gradient_sigmoid(net,a=1):
    return (a*sigmoid(net,a))*(1-sigmoid(net,a))


def gradient_Hyperbolic_Tangent(net,b=1,a=1):
    return (b/a) * (a-Hyperbolic_Tangent(net,b,a) * (a+Hyperbolic_Tangent(net,b,a)))