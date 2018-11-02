# coding:utf-8
import random
import math
import numpy as np
import matplotlib.pyplot as plt

#定义神经元的结构
class neuron:
    def __init__(self,b):
        self.b=b
        self.weights=[]
        self.sigmoid = lambda x: 1 / (1 + math.exp(-x))

    def get_output(self,inputs):
        self.input=inputs
        sum=0
        for i in range(len(inputs)):
            sum+=self.input[i]*self.weights[i]
        sum+=self.b
        self.output=self.sigmoid(sum)
        return self.output

    #计算误差
    def get_error(self,tout):
        return 0.5*(tout-self.output)**2

    #输入层的误差
    def error_inputLayer(self,tout):
        return -(tout-self.output)*self.output*(1-self.output)

    #输出层的误差
    def  error_outputLayer(self,tout):
         return -(tout-self.output)

    #输出层到输入层的误差
    def error_outputLayer_to_inputLayer(self):
        return self.output*(1-self.output)

    #输入层对权重的误差
    def error_inputLayer_to_weight(self,i):
        return self.input[i]

#定义层的结构
class  NetLayer:
    def __init__(self,num,b):
        if b is None:
            self.b=random.random()
        else:
            self.b=b
        self.neurons=[]
        for i in range(num):
            a=neuron(self.b)
            self.neurons.append(a)

    def  layer_output(self,inputs):
        outputs=[]
        for each in self.neurons:
            output=each.get_output(inputs)
            outputs.append(output)
        return outputs

class BPnetwork:
    def __init__(self,num1,num2,num3,learing_rate,hweight=None,hb=None,oweight=None,ob=None):
        self.learing_rate=learing_rate
        self.hiddenlayer=NetLayer(num2,hb)
        self.outputlayer=NetLayer(num3,ob)
        #建立隐藏层
        cnt=0
        for  i in range(num2):
            for j in range(num1):
                if not hweight:
                    self.hiddenlayer.neurons[i].weights.append(random.random())
                else:
                    self.hiddenlayer.neurons[i].weights.append(hweight[cnt])
                cnt+=1
        cnt1=0
        for i in range(num3):
            for j in range(num2):
                if not oweight:
                    self.outputlayer.neurons[i].weights.append(random.random())
                else:
                    self.outputlayer.neurons[i].weights.append(oweight[cnt1])
                cnt1+=1
    #前向传播
    def forward(self,inputs):
         hidder_outputs=self.hiddenlayer.layer_output(inputs)
         final_outputs=self.outputlayer.layer_output(hidder_outputs)
         return final_outputs
    #反向传播
    def backward(self,train_input,train_output):
        self.forward(train_input)
        update_error_output=[0]*len(self.outputlayer.neurons)
        for i in range(len(self.outputlayer.neurons)):
            update_error_output[i]=self.outputlayer.neurons[i].error_inputLayer(train_output[i])

        update_error_hidden=[0]*len(self.hiddenlayer.neurons)
        for i in range(len(self.hiddenlayer.neurons)):
            temp=0
            for j in range(len(self.outputlayer.neurons)):
                temp+=update_error_output[j]*self.outputlayer.neurons[j].weights[i]
            update_error_hidden[i]=temp*self.hiddenlayer.neurons[i].error_outputLayer_to_inputLayer()

        #更新权值
        for i in range(len(self.outputlayer.neurons)):
            for j in range(len(self.outputlayer.neurons[i].weights)):
                delta_weight=update_error_output[i]*self.outputlayer.neurons[i].error_inputLayer_to_weight(j)
                self.outputlayer.neurons[i].weights[j]-=delta_weight*self.learing_rate

        for i in range(len(self.hiddenlayer.neurons)):
            for j in range(len(self.hiddenlayer.neurons[i].weights)):
                delta_weight=update_error_hidden[i]*self.hiddenlayer.neurons[i].error_inputLayer_to_weight(j)
                self.hiddenlayer.neurons[i].weights[j]-=delta_weight*self.learing_rate

    def trains(self,train_data):
        for i in range(len(train_data)):
            self.backward(train_data[i][0],train_data[i][1])

    def get_etotal(self,train_data):
        sum=0
        for i in range(len(train_data)):
            train_input,train_output=train_data[i]
            self.forward(train_input)
            for j in range(len(train_output)):
                # temp=self.outputlayer.neurons[j].get_output()
                # sum+=0.5*(train_output[j]-temp)**2
                sum+=self.outputlayer.neurons[j].get_error(train_output[j])
        return sum

    def get_result(self,test_data):
        hidden_output=[]
        for i in range(len(self.hiddenlayer.neurons)):
            hidden_output.append(self.hiddenlayer.neurons[i].get_output(test_data))
        outputs=[]
        for i in range(len(self.outputlayer.neurons)):
            outputs.append(self.outputlayer.neurons[i].get_output(hidden_output))
        return outputs






