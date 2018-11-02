from BP import BPnetwork
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fun=lambda x:x**2
net1=BPnetwork(1,6,1,0.5)
training_sets=[]
for i in range(1000):
     training_inputs = []
     training_outputs = []
     a=random.random()
     b=fun(a)
     training_inputs.append(a)
     training_outputs.append(b)
     training_sets.append([training_inputs,training_outputs])

error_sets=[]
for i in range(100):
    random.shuffle(training_sets)
    each=training_sets[0:90]
    net1.trains(each)
    error=net1.get_etotal(training_sets)
    error_sets.append(error)
    print(i,error)


test_sets=[]
for i in range(100):
    test_inputs=[]
    test_outputs=[]
    a=random.random()
    b=fun(a)
    test_inputs.append(a)
    test_outputs.append(b)
    test_sets.append([test_inputs,test_outputs])

test_sets=[[[x],[fun(x)]] for x in np.arange(0,1,0.01)]
tout=[]
outputs=[]
sum=0
for i in range(len(test_sets)):
     res=net1.get_result(test_sets[i][0])
     outputs.append(res)
     tout.append((test_sets[i][1]))
     if res[0]-test_sets[i][1][0]<=0.02:
         sum+=1
print(sum/len(test_sets))
x=np.arange(0,1,1/100)
plt.plot(x,tout,'g-',x,outputs,'b-')
plt.show()
plt.plot(x,error_sets,'r-')
plt.show()

# y=x1+x2
def  fun1(x):
    y=0
    for i in range(len(x)):
        y+=x[i]
    return y
#随机产生训练数据集
training_sets=[]
for i in range(1000):
    training_inputs = [random.random(), random.random()]
    training_outputs = [fun1(training_inputs)]
    training_sets.append([training_inputs,training_outputs])

net2=BPnetwork(2,15,1,0.5)
error_sets=[]
for i in range(100):
    random.shuffle(training_sets)
    each=training_sets[0:100]
    net2.trains(each)
    error=net2.get_etotal(training_sets)
    error_sets.append(error)
    print(i,error)

test_sets=[[[x,x],[fun1([x,x])]] for x in np.arange(0,1,0.01) ]
tout=[]
outputs=[]
sum=0
for i in range(len(test_sets)):
     res=net2.get_result(test_sets[i][0])
     outputs.append(res)
     tout.append((test_sets[i][1]))
     if res[0]-test_sets[i][1][0]<=0.02:
             sum+=1
print(sum/len(test_sets))

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(0, 0.5, 0.01)
Y = np.arange(0, 0.5, 0.01)
X, Y = np.meshgrid(X, Y)
Z = X+ Y
ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
plt.show()

x=np.arange(0,1,1/100)
plt.plot(x,error_sets,'r-')
plt.show()

#产生0到pi的数据
# y=sin(x)
training_sets=[[[x/(np.pi)],[np.sin(x)]] for x in np.arange(0,np.pi,0.01)]

net3=BPnetwork(1,12,1,0.5)
error_sets=[]
for i in range(315):
    random.shuffle(training_sets)
    each=training_sets[0:315]
    net3.trains(each)
    error=net3.get_etotal(training_sets)
    error_sets.append(error)
    print(i,error)

test_sets=[[[x/(np.pi)],[np.sin(x)]] for x in np.arange(0,np.pi,0.01)]

tout=[]
outputs=[]
sum=0
for i in range(len(test_sets)):
     res=net3.get_result(test_sets[i][0])
     outputs.append(res)
     tout.append((test_sets[i][1]))
     if res[0]-test_sets[i][1][0]<=0.02:
         sum+=1
print(sum/len(test_sets))
x=np.arange(0,1,1/315)
plt.plot(x,tout,'g-',x,outputs,'b-')
plt.show()
plt.plot(x,error_sets,'r-')
plt.show()