import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import time as time
import os
import ufl_legacy as ufl

#densidad
rho=1053
#dominio
lx=0.015
ly=0.015
lz=0.015

#numero elementos
nx=100
ny=25
nz=25

#tiempo
t_0=0.0
t_1=0.03
t_pasos=150

#Funciones de rotacion
beta=0
gamma=0
alfa='(174.53293*x[0])-1.22173'

rx=fe.Expression([['1','0','0'],['0','cos('+alfa+')','-sin('+alfa+')'],['0','sin('+alfa+')','cos('+alfa+')']],degree=0)
ry=fe.Expression([['cos(beta)','0','sin(beta)'],['0','1','0'],['-sin(beta)','0','cos(beta)']],degree=0,beta=0)
rz=fe.Expression([['cos(gamma)','-sin(gamma)','0'],['sin(gamma)','cos(gamma)','0'],['0','0','1']],degree=0,gamma=0)

#Archivo
nombre='Suave100x25x25Sin-su'
fileU = fe.File('CardioOnda/'+nombre+'/U.pvd')
fileE = fe.File('CardioOnda/'+nombre+'/E.pvd')
fileS = fe.XDMFFile('ondaB/solution'+nombre+'S.xdmf')
fileUXDMF=fe.XDMFFile('CardioOnda/'+nombre+'/UX.xdmf')
fileUXDMFY=fe.XDMFFile('CardioOnda/'+nombre+'/UY.xdmf')
fileV = fe.File('CardioOnda/'+nombre+'/V.pvd')
fileRx = fe.File('CardioOnda/'+nombre+'/Rx.pvd')
fileRy = fe.File('CardioOnda/'+nombre+'/Ry.pvd')
fileRz = fe.File('CardioOnda/'+nombre+'/Rz.pvd')
fileR = fe.File('CardioOnda/'+nombre+'/R.pvd')
fileMFD = fe.File('CardioOnda/'+nombre+'/MFD.pvd')
fileCFD = fe.File('CardioOnda/'+nombre+'/CFD.pvd')
fileC = fe.File('CardioOnda/'+nombre+'/C.pvd')

#elasticidad
c1111='190639'
c2222='165509'
c3333='188477'
c1122='160553'
c1133='175014'
c2233='168465'
c2323='5996'
c1313='5429'
c1212='8185'

c1112='0'
c1113='0'
c1123='0'
c1213='0'
c1222='0'
c1223='0'
c1233='0'
c1322='0'
c1323='0'
c1333='0'
c2223='0'
c2333='0'



"""
Notas sobre la notacion de Voigt para isotropia ortotropica
1->11
2->22
3->33
4->23
5->13
6->12

c11 c12 c13         c1111 c1122 c1133
c12 c22 c23  ->     c1122 c2222 c2233
c13 c23 c33         c1133 c2233 c3333

c44  0   0          c2323   0     0
 0  c55  0   ->       0   c1313   0
 0      c66           0     0    c1212

"""

d = np.zeros((3, 3, 3, 3))

c=fe.Constant([[[[c1111,c1112,c1113],[c1112,c1122,c1123],[c1113,c1123,c1133]],
                [[c1112,c1212,c1213],[c1212,c1222,c1223],[c1213,c1223,c1233]],
                [[c1113,c1213,c1313],[c1213,c1322,c1323],[c1313,c1323,c1333]]],
                
                [[[c1112,c1212,c1213],[c1212,c1222,c1223],[c1213,c1223,c1233]],
                [[c1122,c1222,c1322],[c1222,c2222,c2223],[c1322,c2223,c2233]],
                [[c1123,c1223,c1323],[c1223,c2223,c2323],[c1323,c2323,c2333]]],
                
                [[[c1113,c1123,c1313],[c1213,c1322,c1323],[c1313,c1323,c1333]],
                 [[c1123,c1223,c1323],[c1223,c2223,c2323],[c1323,c2323,c2333]],
                 [[c1133,c1233,c1333],[c1233,c2233,c2333],[c1333,c2333,c3333]]]])

c=c/rho

r1=fe.dot(rz,ry)
r=fe.dot(r1,rx)
i,j,k,l,m,n,o,p=ufl.indices(8)
tx=fe.as_tensor(c[i,j,k,l]*r[i,m],(m,j,k,l))
tx=fe.as_tensor(tx[m,j,k,l]*r[j,n],(m,n,k,l))
tx=fe.as_tensor(tx[m,n,k,l]*r[k,o],(m,n,o,l))
tx=fe.as_tensor(tx[m,n,o,l]*r[l,p],(m,n,o,p))


#Mallado
mesh = fe.BoxMesh(fe.Point(0,0,0), fe.Point(lx,ly,lz),nx,ny,nz)
#fe.plot(mesh)
#plt.show()

#Espacios de funciones
V=fe.VectorFunctionSpace(mesh,'P',1)
u_tr=fe.TrialFunction(V)
v=fe.TestFunction(V)
#W=fe.TensorFunctionSpace(mesh,'P',1)
#H=fe.FunctionSpace(mesh,'P',1)

#funciones y clases

tol=1E-10

def frontera(x,on_boundary):
    return fe.near(x[0],0,tol) and x[1]>(3*ly/8) and x[1]<(5*ly/8) and x[2]>(3*lz/8) and x[2]<(5*lz/8)

def epsilon(u):
    return 0.5*(fe.grad(u)+fe.grad(u).T)

def sigma(u):
    i,j,k,l=ufl.indices(4)
    eps=epsilon(u)
    #si=ufl.operators.contraction(c[i,j,k,l],(i,j),eps[i,j],(i,j))
    #sdp=c[i,j,k,l]*eps[i,j]
    return fe.as_tensor(c[i,j,k,l]*eps[i,j],(k,l))
    #return si
    #return sdp



t=np.linspace(t_0,t_1,t_pasos)
dt=t_1/t_pasos



#condiciones de frontera
u_D=fe.Expression(("t<b?sin(a*t):0",
                   "0",
                   "0"),t=0.0,a=3588,b=np.pi/3588,degree=0)

bc=fe.DirichletBC(V,u_D,frontera)

#Inicialización
u=fe.Function(V)
s=fe.Function(V)
u_bar=fe.Function(V)
du=fe.Function(V)
ddu=fe.Function(V)
ddu_old=fe.Function(V)
#s=fe.Function(W)



def a_horas_minutos(tempo):
    minutos=int(tempo/60)
    segundos=tempo-minutos*60
    horas=int(minutos/60)
    minutos=minutos-horas*60
    return str(horas)+":"+str(minutos)+":"+str(segundos)

#Forma Débil
F=fe.inner(sigma(u_tr),epsilon(v))*fe.dx + 4*rho/(dt*dt)*fe.dot(u_tr-u_bar,v)*fe.dx
a, L = fe.lhs(F), fe.rhs(F)


#Integracion
tiempo=0
n=0
for ti in t:
    print("iniciando")
    if(n==0):
        """
        print("proyectando orientación")
        vectori=fe.project(rotacion(),V,solver_type="mumps")
        vectori.rename("v","v")
        print("escribiendo orientación",end=" ")
        fileV << (vectori,ti)
        print("☺")
        print("proyectando rotacion X")
        frx=fe.project(rx,W,solver_type="mumps")
        frx.rename("frx","frx")
        print("escribiendo rotación X",end=" ")
        fileRx << (frx,ti)
        print("☺")
        print("proyectando rotacion Y")
        fry=fe.project(ry,W,solver_type="mumps")
        fry.rename("fry","fry")
        print("escribiendo rotación Y",end=" ")
        fileRy << (fry,ti)
        print("☺")
        print("proyectando rotacion Z")
        frz=fe.project(rz,W,solver_type="mumps")
        frz.rename("frz","frz")
        print("escribiendo rotación Z",end=" ")
        fileRz << (frz,ti)
        print("☺")
        print("proyectando rotacion")
        fr=fe.project(r,W,solver_type="mumps")
        fr.rename("fr","fr")
        print("escribiendo rotación",end=" ")
        fileR << (fr,ti)
        print("☺")
        print("proyectando elasticidad")
        elas=fe.project(elastic(),W,solver_type="mumps")
        elas.rename("elas","elas")
        print("escribiendo elasticidad",end=" ")
        fileC << (elas,ti)
        print("☺")
        """
        marca_tiempo1=time.time()
    
    print("iteración: "+str(n))        
    
    tiempo +=dt
    u_D.t=ti
    #k=u+dt*du+0.25*dt*dt*ddu
    u_bar.assign(u+dt*du+0.25*dt*dt*ddu)
    print("resolviendo")
    fe.solve(a==L,u,bc,solver_parameters={'linear_solver':'mumps'})
    #s=fe.project(sigma(u),W,solver_type="mumps")
    #s.rename("s","s")
    #fileS.write(s, ti)
    """
    print("escribiendo S",end=" ")
    esp=fe.project(epsilon(u),W,solver_type="mumps")
    esp.rename("esp","esp")
    fileE << (esp, ti)
    print("☺")
    """
    print("escribiendo U",end=" ")
    ddu_old.assign(ddu)
    ddu.assign(4/(dt*dt)*(u-u_bar))
    du.assign(du+0.5*dt*(ddu+ddu_old))
    u.rename("su","su")
    fileUXDMF.write_checkpoint(fe.project(u,V),"su",ti,fileUXDMF.Encoding.HDF5,True)
    fileUXDMF.close()
    fileU << (u, ti)
    print("☺")
    
    n=n+1
    marca_tiempo2=time.time()
    pasado=marca_tiempo2-marca_tiempo1
    tpasado=a_horas_minutos(pasado)
    print("tiempo pasado:"+tpasado)
    faltante=(pasado)*(t_pasos-n)/n
    restante=a_horas_minutos(faltante)
    
    print("estimado restante:"+restante)

    
    #[s11,s12,s21,s22]=s.split(True)
    #s_1=fe.project(s11,H)
    #if(n%5==0):
        #s_1.rename("s_1","s_1")
    
    
    
    
    
        #h=fe.project(fe.inner(u,u),H)
        #fe.plot(s11)
        #plt.savefig(str(n)+".png")
    
    
#fileS.close()
