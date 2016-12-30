from __future__ import division
import numpy as np
from sympy import *

class ThePablos:
    
    def __init__(self, l1=1, l2=1, l3=1, l4=1, r=1e-2, t=1):
        
        #Configurar dimensiones
        self.l = np.array([l1,l2,l3,l4])
        self.r = r
        
        #Posiciones locales de los centros de masa
        self.CM = np.array([[0, -self.l[0]/2.0, 0,1],\
                            [-self.l[1]/2.0,0,0,1],\
                            [-self.l[2]/2.0,0,0,1],\
                            [-self.l[3]/2.0,0,0,1] ])
        
        #Tiempo total de accion
        self.t = t
        
        #Vector de gravedad
        self.g = np.array([0,0,-10,0])
        
        #Matriz de inercia de cada barra
        Ia = (self.r**2)*self.l/2
        Ib = ( (self.r**2)*self.l/4 ) + ( (self.l**3)/3 )
        Ibars = np.array([[ Ib[0], Ia[0], Ib[0] ],\
                               [ Ia[1], Ib[1], Ib[1] ],\
                               [ Ia[2], Ib[2], Ib[2] ],\
                               [ Ia[3], Ib[3], Ib[3] ] ])
        
        #Matriz de inercia 4x4 para la matraca
        self.I = np.zeros((4,4,4))                       
        for i in range(4):
            aux1 = (-Ibars[i,0] + Ibars[i,1] + Ibars[i,2])/2
            aux2 = (Ibars[i,0] - Ibars[i,1] + Ibars[i,2])/2
            aux3 = (Ibars[i,0] + Ibars[i,1] - Ibars[i,2])/2
            self.I[i] = np.array( [[aux1, 0,0,self.l[i]*self.CM[i,0]],\
                                   [0, aux2,0,self.l[i]*self.CM[i,1]],\
                                   [0, 0,aux3,self.l[i]*self.CM[i,2]],\
                                   [self.l[i]*self.CM[i,0], self.l[i]*self.CM[i,1], self.l[i]*self.CM[i,2], self.l[i] ]])
        
    def setPath(self, theta1, theta2, theta3, theta4):
        
        #Ingresar las trayectorias y el numero de puntos en cada una
        self.N = len(theta1)    
        self.theta = np.vstack([theta1,theta2,theta3,theta4])
        
    def angularCons(self):
        
        self.c1 = np.cos( self.theta[0] )
        self.c2 = np.cos( self.theta[1] )
        self.c23 = np.cos( self.theta[1] + self.theta[2] )
        self.c234 = np.cos( self.theta[1] + self.theta[2] + self.theta[3] )
        
        self.s1 = np.sin( self.theta[0] )
        self.s2 = np.sin( self.theta[1] )
        self.s23 = np.sin( self.theta[1] + self.theta[2] )
        self.s234 = np.sin( self.theta[1] + self.theta[2] + self.theta[3] )
        
    def getEnergySim(self):
        #constantes angulares cargadas
        
        #Derivadas angulares
        ts = self.t/self.N
        
        dTheta = np.zeros((4,self.N))
        for t in range(self.N):
            for i in range(4):
                if ( t == 0 ):
                    dTheta[i,t] = (self.theta[i,t+1]-self.theta[i,t])/ts
                elif ( t == (self.N-1) ):
                    dTheta[i,t] = (self.theta[i,t]-self.theta[i,t-1])/ts
                else:
                    dTheta[i,t] = (self.theta[i,t+1]-self.theta[i,t-1])/(2*ts)
        #some matra-k
        
        v2x = -self.s1*dTheta[0]*self.l[1]*self.c2 \
                -self.c1*dTheta[1]*self.s2*self.l[1]
        v2y = self.c1*dTheta[0]*self.l[1]*self.c2 \
                -self.s1*dTheta[1]*self.s2*self.l[1]
        v2z = dTheta[1]*self.c2*self.l[1]
        
        v3x = v2x - dTheta[0]*self.l[2]*self.c23*self.s1 \
            - self.c1*self.l[2]*self.s23*( dTheta[1]+dTheta[2] )
        v3y = v2y + dTheta[0]*self.l[2]*self.c23*self.c1 \
            - self.s1*self.l[2]*self.s23*( dTheta[1]+dTheta[2] )  
        v3z = v2z + self.l[2]*self.c23*( dTheta[1]+dTheta[2] )
        
        v4x = v3x - dTheta[0]*self.l[3]*self.c234*self.s1 \
            - self.c1*self.l[3]*self.s234*( dTheta[1]+dTheta[2]+dTheta[3] )
        v4y = v3y + dTheta[0]*self.l[3]*self.c234*self.c1 \
            - self.s1*self.l[3]*self.s234*( dTheta[1]+dTheta[2]+dTheta[3] ) 
        v4z = v3z + self.l[3]*self.c234*( dTheta[1]+dTheta[2]+dTheta[3] )
        
        #suma proporcional a los cambios de energia cinetica
        aux  = sum( np.absolute(np.diff(v2x**2)) )*self.l[1]
        aux += sum( np.absolute(np.diff(v2y**2)) )*self.l[1]
        aux += sum( np.absolute(np.diff(v2z**2)) )*self.l[1]
        aux += sum( np.absolute(np.diff(v3x**2)) )*self.l[2]
        aux += sum( np.absolute(np.diff(v3y**2)) )*self.l[2]
        aux += sum( np.absolute(np.diff(v3z**2)) )*self.l[2]
        aux += sum( np.absolute(np.diff(v4x**2)) )*self.l[3]
        aux += sum( np.absolute(np.diff(v4y**2)) )*self.l[3]
        aux += sum( np.absolute(np.diff(v4z**2)) )*self.l[3]
        #aux = aux*ts
        
        return aux
    
    def simDerivatives(self):
         #Inicializar las matrices de transformacion de manera simbolica
            
        t1 = Symbol('t1')
        t2 = Symbol('t2')
        t3 = Symbol('t3')
        t4 = Symbol('t4')
        
        T_01 = Matrix([ [cos(t1), 0, sin(t1), 0],\
                        [sin(t1), 0, -cos(t1), 0],\
                        [0,1,0,self.l[0]],\
                        [0,0,0,1]])
        T_02 = Matrix([  [ cos(t1)*cos(t2), -cos(t1)*sin(t2), sin(t1), cos(t1)*self.l[1]*cos(t2) ],\
              [ sin(t1)*cos(t2), -sin(t1)*sin(t2), -cos(t1), sin(t1)*self.l[1]*cos(t2) ],\
              [ sin(t2), cos(t2), 0, self.l[0]+self.l[1]*sin(t2) ],\
              [ 0,0,0,1 ]])
        T_03 = Matrix([[cos(t1)*cos(t2+t3), -cos(t1)*sin(t2+t3), sin(t1), cos(t1)*(self.l[1]*cos(t2)+self.l[2]*cos(t2+t3))],\
                       [sin(t1)*cos(t2+t3), -sin(t1)*sin(t2+t3), -cos(t1), sin(t1)*(self.l[1]*cos(t2)+self.l[2]*cos(t2+t3))],\
                       [sin(t2+t3), cos(t2+t3), 0, self.l[0]+self.l[1]*sin(t2)+self.l[2]*sin(t2+t3)],\
                       [0,0,0,1]])
        T_04 = Matrix([[cos(t1)*cos(t2+t3+t4),-cos(t1)*sin(t2+t3+t4),sin(t1),cos(t1)*(self.l[1]*cos(t2)+self.l[2]*cos(t2+t3)+self.l[3]*cos(t2+t3+t4))],\
                       [sin(t1)*cos(t2+t3+t4),-sin(t1)*sin(t2+t3+t4),-cos(t1),sin(t1)*(self.l[1]*cos(t2)+self.l[2]*cos(t2+t3)+self.l[3]*cos(t2+t3+t4))],\
                     [sin(t2+t3+t4),cos(t2+t3+t4),0, self.l[0]+self.l[1]*sin(t2)+self.l[2]*sin(t2+t3)+ self.l[3]*sin(t2+t3+t4)],\
                       [0,0,0,1]])      
        
        #listas para recorrer las variables simbolicas y las matrices simbolicas
        tt = [t1,t2,t3,t4]
        T_0 = [T_01,T_02,T_03,T_04]
        
        
        #Primera derivada respecto a angulos
        self.Tpr = []
        for i in range(4):
            self.Tpr.append([])
            for j in range(4):
                Tprima = T_0[i].diff(tt[j])
                func = lambdify((t1,t2,t3,t4), Tprima, 'numpy')
                self.Tpr[i].append(func)
                
        #Segunda derivada respecto a angulos
        self.Tprpr = []
        for i in range(4):
            self.Tprpr.append([])
            for j in range(4):
                self.Tprpr[i].append([])
                for k in range(4):
                    Tprimaprima = T_0[i].diff(tt[j], tt[k])
                    func2 = lambdify((t1,t2,t3,t4), Tprimaprima, 'numpy')
                    self.Tprpr[i][j].append(func2)
    
    def getTorque(self):
        #some hard coding
        #hard coding incoming
        #wait for it
        #here it comes
        
        #Primera derivada respecto a angulos
        self.difT = np.zeros((self.N,4,4,4,4))
        for t in range(self.N):
            for i in range(4):
                for j in range(4):
                    func = self.Tpr[i][j]
                    self.difT[t,i,j] = func( self.theta[0,t],self.theta[1,t],self.theta[2,t],self.theta[3,t] )
        
        #Segunda derivada respecto a angulos
        self.ddifT = np.zeros((self.N,4,4,4,4,4))
        for t in range(self.N):
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        func2 = self.Tprpr[i][j][k]
                        self.ddifT[t,i,j,k] = func2( self.theta[0,t],self.theta[1,t],self.theta[2,t],self.theta[3,t] )
    
        #Termino de la gravedad
        self.GG = np.zeros((4,self.N))
        for t in range(self.N):
            for i in range(4):
                #Sumatoria
                for j in range(4):
                    self.GG[i,t] -= self.l[j]*np.dot( np.dot(self.difT[t,j,i], self.CM[j]) , self.g ) 
       
        #Termino de coriolis y centrifuga
        self.HH = np.zeros((self.N,4,4,4))
        for t in range(self.N):
            for i in range(4):
                for k in range(4):
                    for m in range(4):
                        #Sumatoria
                        first = max([i,k,m])
                        for j in range(first,4):
                            self.HH[t,i,k,m] += np.trace( np.dot( np.dot(self.ddifT[t,j,k,m], self.I[j]), self.difT[t,j,i].T ) )
                    
        #Termino de inercia
        self.MM = np.zeros((self.N,4,4))
        for t in range(self.N):
            for i in range(4):
                for k in range(4):
                    #Sumatoria
                    first = max([i,k])
                    for j in range(first,4):
                        self.MM[t,i,k] += np.trace( np.dot( np.dot(self.difT[t,j,k], self.I[j]), self.difT[t,j,i].T ) )
        
        #Derivadas angulares
        ts = self.t/self.N
        
        self.difTheta = np.zeros((self.N,4))
        for t in range(self.N):
            for i in range(4):
                if ( t == 0 ):
                    self.difTheta[t,i] = (self.theta[i,t+1]-self.theta[i,t])/ts
                elif ( t == (self.N-1) ):
                    self.difTheta[t,i] = (self.theta[i,t]-self.theta[i,t-1])/ts
                else:
                    self.difTheta[t,i] = (self.theta[i,t+1]-self.theta[i,t-1])/(2*ts)
        
        self.ddifTheta = np.zeros((self.N,4))
        for t in range(self.N):
            for i in range(4):
                if ( t == 0 ):
                    self.ddifTheta[t,i] = (self.theta[i,t+2]-2*self.theta[i,t+1]+self.theta[i,t])/(2*(ts**2))
                elif ( t == (self.N-1) ):
                    self.ddifTheta[t,i] = (self.theta[i,t]-2*self.theta[i,t-1]+self.theta[i,t-2])/(2*(ts**2))
                else:
                    self.ddifTheta[t,i] = (self.theta[i,t+1]-2*self.theta[i,t]+self.theta[i,t-1])/(ts**2)   
        
        #Torque
        
        #Sumatoria con M
        self.sumatoriaM = np.zeros((4,self.N))
        for t in range(self.N):
            for i in range(4):
                #Sumatoria
                for k in range(4):
                    self.sumatoriaM[i,t] += self.MM[t,i,k]*self.ddifTheta[t,k]    
                
        #Sumatoria con H
        self.sumatoriaH = np.zeros((4,self.N))
        for t in range(self.N):
            for i in range(4):            
                #Sumatoria
                for k in range(4):
                    for m in range(4):
                        self.sumatoriaH[i,t] += self.HH[t,i,k,m]*self.difTheta[t,k]*self.difTheta[t,m] 
               
        #Expresion final
        self.torque = self.sumatoriaM + self.sumatoriaH + self.GG
        
        return self.torque
    
    def getTorqueObj(self):
        ts = self.t/self.N
        torque = self.getTorque()
        
        #return sum(sum( ( np.diff(torque)**2 ) ))/(2*ts) #TorqueChange
        return np.sqrt( np.mean(torque**2) ) #RMS
        
    def efectorPath(self):
        
        #constantes angulares cargadas
        
        x = self.c1 * ( self.l[1]*self.c2 + self.l[2]*self.c23 + self.l[3]*self.c234 ) 
        y = self.s1 * ( self.l[1]*self.c2 + self.l[2]*self.c23 + self.l[3]*self.c234 )
        z = self.l[0] + self.l[1]*self.s2 + self.l[2]*self.s23 + self.l[3]*self.s234
        
        x.shape = (self.N,1)
        y.shape = (self.N,1)
        z.shape = (self.N,1)

        return np.hstack([x,y,z])
    
    def endPoint(self):
        #constantes angulares cargadas
        
        x = self.c1[-1] * ( self.l[1]*self.c2[-1] + self.l[2]*self.c23[-1] + self.l[3]*self.c234[-1] ) 
        y = self.s1[-1] * ( self.l[1]*self.c2[-1] + self.l[2]*self.c23[-1] + self.l[3]*self.c234[-1] )
        z = self.l[0] + self.l[1]*self.s2[-1] + self.l[2]*self.s23[-1] + self.l[3]*self.s234[-1]
        
        return np.array([x,y,z])
    
    def getExtremos(self,k):
        r1 = 0
        z1 = self.l[0]
        
        r2 = self.l[1]*self.c2[k]
        z2 = self.l[0] + self.l[1]*self.s2[k]
        
        r3 = self.l[1]*self.c2[k] + self.l[2]*self.c23[k]
        z3 = self.l[0] + self.l[1]*self.s2[k] + self.l[2]*self.s23[k]
        
        r4 = self.l[1]*self.c2[k] + self.l[2]*self.c23[k] + self.l[3]*self.c234[k]
        z4 = self.l[0] + self.l[1]*self.s2[k] + self.l[2]*self.s23[k] + self.l[3]*self.s234[k]
        
        return np.array([[r1,r2,r3,r4],[z1,z2,z3,z4]]) 
