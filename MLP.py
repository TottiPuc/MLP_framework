# -*- coding: utf-8 -*-
import numpy as np
#Clase para representar la estructura de grafos de un nodo genérico
class Neuron(object):
    """ 
    Clase base para los nodos de la red
    Argumentos: nodos_entrada: una lista de nodos com  lineas para cada nodo
    """
    def __init__(self, nodos_entrada =[]):
        #nodos que reciben las entradas
        self.nodos_entrada = nodos_entrada
        # nodos para los cuales se pasan los valores
        self.nodos_salida = []
        #para cada nodod de entrada sera adicionado como un nodo de salida
        for n in self.nodos_entrada:
            n.nodos_salida.append(self)
        self.valor = None # con el fin de actualizar al momento de retropropagar
        
    def forward(self):
        """
        forward propagation

        calcular el valor de salida con  base en los nodos de entrada 
        y se almacena el valor o resultado en self.valor
        -------
        
        """
        pass
        
#creo un subclase que herda de la clase principal NEuron
class Input(Neuron):
    """
    una entrada generica en la red
    """
    def __init__(self):
        #un nodo de entrada no posee neurones de entrada
        #por esta razon no es necesario pasar nada pa instanciar la clase 
        Neuron.__init__(self)
        # el nodo de entrada es el unico donde el valor puede ser pasado como un argumento 
        #para la funcion forward, todos los demas nodos deben obtener el valor del nodo anterior
        #de self.nodos_entrada
        
        def forward(self,valor =None):
            # se sustityue el valor actual si un valor es pasado como parametro
            # if valor is not None:
            #     self.valor=valor
            pass
        
#subclase de Neuron para realizar calculos                
# class Add(Neuron):
#     def __init__(self,*inputs):
#         Neuron.__init__(self,inputs)
        
#     def forward(self):
#         x_value = self.nodos_entrada[0].valor
#         y_value = self.nodos_entrada[1].valor
#         self.valor = x_value + y_value
        
class Lineal(Neuron):
    """
    Representa un nodo que ejecuta una transformacion lineal
    """
    def __init__(self,X,W,b):
        # el constructo de la calse base (nodo). pesos y bias son tratados como nodos de entrada
        Neuron.__init__(self,[X,W,b])
        
    def forward(self):
        X = self.nodos_entrada[0].valor
        W = self.nodos_entrada[1].valor
        b = self.nodos_entrada[2].valor
        self.valor = np.dot(X,W) + b
        
        
class Sigmoid(Neuron):
    """
    Representa un nodo que ejecuta la funcion de activacion sigmoid
    """
    def __init__(self,nodo):
        Neuron.__init__(self,[nodo])
        
    def _sigmoid(self,x): # 
        return 1./(1. + np.exp(-x))
    
    def forward(self):
        input_value = self.nodos_entrada[0].valor
        self.valor = self._sigmoid(input_value)
        
class CostFunction(Neuron):
    def __init__(self,y,a):
        """
        Funcion de coste del error medio cuadratico que es usado como ultimo nodo de la red
        donde y es el valor original y el valor de a es la prediccion en cada pasada
        """
        Neuron.__init__(self,[y,a])
    def forward(self):
        """
        Calculo del error cuadratico medio
        conviertiendo los array (3,1 para evitar problemas )
        """
        y = self.nodos_entrada[0].valor.reshape(-1,1)
        a = self.nodos_entrada[1].valor.reshape(-1,1)
        m = self.nodos_entrada[0].valor.shape[0]
        
        diff = y - a
        self.valor = np.mean(diff**2)
                        
        
def topologia_sort(feed_dict):
    """
    Algoritmo de kahn para clasificar los nodos en orden topologico 
    
    Parameters
    ----------
    feed_dict : diccionario
        diccionario donde la clave es un nodo de entrada y el valor es el respectico feed de valor para ese nodo

    Returns: una lista de nodos ordenados L
    -------
    """ 
    
    input_nodos = [n for n in feed_dict.keys()]
    
    G = {}
    nodos = [n for n in input_nodos]
    while len(nodos) > 0:
        n = nodos.pop(0)
        if n not in G:
            G[n] = {'in':set(), 'out':set()}
        for m in n.nodos_salida:
            if m not in G:
               G[m] = {'in':set(), 'out':set()} 
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodos.append(m)
    
    L = []
    S = set(input_nodos)
    while len(S) > 0:
        n = S.pop()
        
        if isinstance(n,Input):
            n.valor = feed_dict[n]
            
        L.append(n)
        for m in n.nodos_salida:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    return L
    
    
def forward_pass(output_node, sorted_nodes):
    """
    Ejecuta una pasada para frente a  través de una lista de nodos ordenados

    Parameters
    ----------
    output_node : 
        un nodo en el grafo, debe ser el nodo de salida
    sorted_nodes : 
        una lista topologiacmente ordenada de nodos

    Returns
    -------
    el valor del nodo de salida

    """
    for n in sorted_nodes:
        print(n.valor)
        n.forward()
        
    return output_node.valor



#ejecución del grafo
    
# definimos los inputs (todo esto es solo para inicializar nodos aqui nada tiene valor)
inputs , weights, bias = Input() , Input(), Input()
y = Input()

#llamamos la funcino lineal()
f = Lineal(inputs , weights, bias)
g = Sigmoid(f)

#Función de coste
cost = CostFunction(y,g)

#atribuyendo valores a los parametros
x = np.array([[-2.,-1.],[-2,-4]])
w = np.array([[4.,-6],[3.,-2]])
b = np.array([-2.,-3])

#valores de salida originales
array_y = np.array([[-4.,-2.],[-1,-3]])

# definimos el feed_dict
feed_dict = {inputs :x , weights:w, bias:b, y: array_y}

#ordenamos las entradas para ejecución
graph = topologia_sort(feed_dict)

#generamos la salida con un forward_pass
output = forward_pass(cost,graph)

print(output)