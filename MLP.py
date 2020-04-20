# -*- coding: utf-8 -*-
import numpy as np
#Clase para representar la estructura de grafos de un nodo genérico
class Neuron(object):
    """ 
    Clase base para los nodos de la red
    Argumentos: nodos_entrada: una lista de nodos com  lineas para cada nodo
    """
    def __init__(self, nodos_entrada =[]):
        
        self.nodos_entrada = nodos_entrada      # nodos que reciben las entradas
        self.nodos_salida = []                  # nodos para los cuales se pasan los valores
        self.valor = None                       # con el fin de actualizar al momento de retropropagar
        self.gradientes = {}                    # almacenara las derivadas que seran usadas para actualizar los pesos
        #para cada nodod de entrada sera adicionado como un nodo de salida
        for n in self.nodos_entrada:
            n.nodos_salida.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
    
    
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
        
    def forward(self):
        pass
    
    def backward(self):
        #un nodo de entrada no posee inputs de este modo el gradiente o derivada es cero
        self.gradientes = {self:0}
        
        #Los pesos y bias pueden ser entradas, entonces es necesario sumar el gradiente, de los gradientes de salida
        for n in self.nodos_salida:
            grad_cost = n.gradientes[self]
            self.gradientes[self] += grad_cost
        
        
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
        
    def backward(self):
        """
        Calcula el gradiente con base en los valores de salida
        """
        
        #inicializa un valor parcial para cada nodo de entrada
        self.gradientes = {n:np.zeros_like(n.valor) for n in self.nodos_entrada}
        
        # recorriendo las salidas
        # el gradiente cambiara dependiendo de cada salida, entonces los gradientes son sumados en todas las salidas
        for n in self.nodos_salida:
            #obtener parcial del coste en relacion a este nodo
            grad_cost = n.gradientes[self]
            
            #Definiendo la perdida parcial en relacion a los nodos de entrada
            self.gradientes[self.nodos_entrada[0]] += np.dot(grad_cost, self.nodos_entrada[1].valor.T)
            
            #Definiendo la perdida parcial en relacion a los pesos de este nodo
            self.gradientes[self.nodos_entrada[1]] += np.dot(self.nodos_entrada[0].valor.T,grad_cost)
            
            #Definiendo la perdida parcial en relacion a los bias de este nodo
            self.gradientes[self.nodos_entrada[2]] += np.sum(grad_cost, axis = 0 , keepdims = False)
            
        
        
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
        
    def backward(self):
    #Calcula el gradiente usando la derivada de la función sigmoide
    #inicializa un valor parcial para cada nodo de entrada con 0
        self.gradientes = {n:np.zeros_like(n.valor) for n in self.nodos_entrada}
    
        # recorriendo las salidas
        # el gradiente cambiara dependiendo de cada salida, entonces los gradientes son sumados en todas las salidas
        for n in self.nodos_salida:
            #obtener parcial del coste en relacion a este nodo
            grad_cost = n.gradientes[self]
            sigmoid = self.valor
            #Definiendo la perdida parcial en relacion a los nodos de entrada
            self.gradientes[self.nodos_entrada[0]] += sigmoid * ( 1 - sigmoid) * grad_cost
            
        
        
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
        self.m = self.nodos_entrada[0].valor.shape[0]
        
        self.diff = y - a
        self.valor = np.mean(self.diff**2)
    
    def backward(self):
        """
        Calcula el gradiente de la funcion de coste
        Este es el nodo final de la red para los nodos de salida
        """
        self.gradientes[self.nodos_entrada[0]] = ( 2 / self.m) * self.diff
        self.gradientes[self.nodos_entrada[1]] = ( -2 / self.m) * self.diff
        
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
    
    
def forward_backward(graph):
    """
    Ejecuta una pasada para adelante y una pasada para atras a través de una lista de nodos ordenados

    """
    #forward pass
    for n in graph:
        print(n.valor)
        n.forward()
        
    # backward pass
    # el valor negativo en el slice permite hacer una copia de  la mesma lista en orden inversa
    for n in graph[::-1]:
        n.backward()
    



#ejecución del grafo
    
# definimos los inputs (todo esto es solo para inicializar nodos aqui nada tiene valor)
X , W, b = Input() , Input(), Input()
y = Input()

#llamamos la funcino lineal()
f = Lineal(X , W, b)
a = Sigmoid(f)

#Función de coste
cost = CostFunction(y,a)

#atribuyendo valores a los parametros
entradas = np.array([[-1., -2.], [-1, -2]])
pesos = np.array([[2.], [3.]])
bias = np.array([-3.])
salida = np.array([1, 2])

# definimos el feed_dict
feed_dict = {X :entradas , y:salida, W:pesos, b:bias}

#ordenamos las entradas para ejecución
graph = topologia_sort(feed_dict)

#forward e backward
forward_backward(graph)

# retorna los gradientes de cada input
gradientes = [t.gradientes[t] for t in [X, y, W, b]]


print(gradientes)