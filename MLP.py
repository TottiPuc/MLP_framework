# -*- coding: utf-8 -*-

#Clase para representar la estructura de grafos de un nodo genérico
class Neuron(object):
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
        
#creo un subclase que herda de la clase principal NEuron
class Input(Neuron):
    def __init__(self):
        #un nodo de entrada no posee neurones de entrada
        #por esta razon no es necesario pasar nada pa instanciar la clase 
        Neuron.__init__(self)
        # el nodo de entrada es el unico donde el valor puede ser pasado como un argumento 
        #para la funcion forward, todos los demas nodos deben obtener el valor del nodo anterior
        #de self.nodos_entrada
        
        def forward(self,valor =None):
            # se sustityue el valor actual si un valor es pasado como parametro
            if valor is not None:
                self.valor=valor
                
#subclase de Neuron para realizar calculos                
class Add(Neuron):
    def __init__(self,*inputs):
        Neuron.__init__(self,inputs)
        
    def forward(self):
        x_value = self.nodos_entrada[0].valor
        y_value = self.nodos_entrada[1].valor
        self.valor = x_value + y_value
        
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
    
    L = {}
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
    
    
    