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
    def __init__(self,x,y):
        Neuron.__init__(self,[x,y])
    