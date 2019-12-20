import numpy as np
import random
from math import exp
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter("ignore")
from keras.utils import to_categorical
from biokit.viz import corrplot


class NeuralNetwork():
    #number of arguments in neuron_count_layers is number of hidden layers and values at each arguments is number of neurons at each hidden layer
    def __init__(self,input_x,output_y,*neuron_count_layers):
        self.input=np.array(input_x)
        self.output=np.array(output_y)
        #number of training samples
        self.M=self.input.shape[-1]
        self.n_features=self.input.shape[0]
        self.num_layers=len(neuron_count_layers)
        self.neurons_hiddenlayer_size=[neuron_count for neuron_count in neuron_count_layers]
        #print(self.input,self.output)
        self.weights_Layers=self.initialize_weights().copy()

    def initialize_weights(self):
        weights_Layers=[]
        epsilon=0
        #Dimensions is number of neurons in next layer * number of inputs from previous layer
        weights_Layers.append(np.random.uniform(-epsilon,epsilon,(self.neurons_hiddenlayer_size[0], self.n_features+1)))
        for layer in range(0,self.num_layers-1):
            weights_Layers.append(np.random.uniform(-epsilon,epsilon,(self.neurons_hiddenlayer_size[layer+1],self.neurons_hiddenlayer_size[layer]+1)))
        # weights_Layers.append(np.linspace(start=-epsilon, stop=epsilon, num=self.neurons_hiddenlayer_size[0]* (self.n_features + 1))\
        #                       .reshape(self.neurons_hiddenlayer_size[0],self.n_features + 1))
        # for layer in range(0, self.num_layers - 1):
        #     weights_Layers.append(np.linspace(start=-epsilon, stop=epsilon,
        #                                        num=self.neurons_hiddenlayer_size[layer + 1] * (self.neurons_hiddenlayer_size[layer] + 1)) \
        #                           .reshape(self.neurons_hiddenlayer_size[layer + 1], self.neurons_hiddenlayer_size[layer] + 1))

        # self.weights = np.linspace(start=-self.epsilon, stop=self.epsilon,
        #                            num=self.layer_shape[0] * (self.layer_shape[1] + 1)) \
        #     .reshape(self.layer_shape[0], self.layer_shape[1] + 1)

        return weights_Layers

    #Sigmoidal activation function for neuron activation 'Z'
    def sigmoid_function(self,Z):
        return 1.0/(1.0+np.exp(-Z))

    def tanh_function(self,Z):
        return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))

    #calculate sum of weights neuron activation
    def Z_vector_function(self,weights,layer_inputs):
            Z=np.matmul(weights,layer_inputs)
            return Z

    def add_bias_term(self,input):
        layerinput_bias=np.insert(input,0,1)
        return layerinput_bias

    #Forward propagation input to output(sigmoidal activation)
    def forward_propagation(self,input):
        self.activation_Z=[]
        layer_input = input
        layer_input_bias = self.add_bias_term(layer_input)
        self.activation_Z.append(layer_input_bias)
        #print(self.weights_Layers)
        for eachlayer_weight in self.weights_Layers:
            Z_vector=self.Z_vector_function(eachlayer_weight,self.add_bias_term(layer_input))
            a_vector=self.sigmoid_function(Z_vector)
            layer_input=a_vector
            layer_input_bias=self.add_bias_term(layer_input)
            self.activation_Z.append(layer_input_bias)

        #print(np.array(self.activation_Z))
        return np.array(self.activation_Z)

    #calculate the derivative of neuron output
    def sigmoid_derivative(self,layer_output):
        return layer_output*(1-layer_output)

    def tanh_derivative(self,layer_output):
        return (1-(layer_output**2))

    #sigmoidal cost function
    def sigmoidal_costFunction_persample(self,expected_output,activation_derivative_hypothesis_matrix):
        return np.sum((expected_output*np.log(activation_derivative_hypothesis_matrix)+ ((1 - expected_output) * np.log(1-activation_derivative_hypothesis_matrix))))
    #tanh cost function
    def tanh_costFunction_persample(self, expected_output, activation_derivative_hypothesis_matrix):
        return np.sum((((expected_output+1)/2) * np.log((activation_derivative_hypothesis_matrix+1)/2) + (
                    (1 - ((expected_output+1)/2)) * np.log(1 - (activation_derivative_hypothesis_matrix+1)/2))))


    #def backward_propagtion
    def back_propagate_error(self,forward_propagation_output,Expected_output):
        self.traingle=0
        self.network_deltas=[]
        self.cost_function=0
        self.loss=0
        _layer_delta = None
        #print("activtion",forward_propagation_output)
        layers_count=len(forward_propagation_output)
        #here 0 means input layer which is exclusive during range function
        for layer_index in range(layers_count-1,0,-1):
            #print("@layer ",layer_index)
            # delta @layer n
            if layer_index==layers_count-1:
                expected_output=Expected_output.reshape(-1,1)
                # current layer activation values
                activation_value_a = forward_propagation_output[layer_index][1:].reshape(-1, 1)
                final_layer_delta=np.subtract(activation_value_a,expected_output.reshape(-1,1))
                self.cost_function=self.sigmoidal_costFunction_persample(expected_output,activation_value_a)
                self.loss=np.sum(np.square(expected_output-activation_value_a))
                #add bias term to make calualting easy for upcoming delta calcualtion
                final_layer_delta=self.add_bias_term(final_layer_delta)
                #perform deep copy of layer deltas to global _layer_delta variable
                _layer_delta=final_layer_delta
                self.network_deltas.insert(0,_layer_delta)
                #print("fina",_layer_delta)
            else:

                #deltas are calculated only till layer two
                #print(forward_propagation_output[layer_index])
                #current layer activation values
                activation_value_a=forward_propagation_output[layer_index].reshape(-1,1)
                forward_layer_deltas=_layer_delta[1:].reshape(-1,1)
                #print("forward layer delta",forward_layer_deltas)

                activation_derivative=self.sigmoid_derivative(activation_value_a)

                #calculation deltas of the single layers
                hidden_layer_delta=np.matmul(self.weights_Layers[layer_index].T,forward_layer_deltas) * activation_derivative

                #copy current layer delta as forward layer deltas or previous layer delta in backpropagation
                _layer_delta=hidden_layer_delta.copy()

                #print("hidden", _layer_delta)
                self.network_deltas.insert(0,_layer_delta)

                # calculating backpropogated errors (triangle) or accumulated deltas for delta of forward layer delta and current layer activation
                ##calculating accumulated deltas from @layer_n to @layer_2
                #dividing accumilated deltas with number of training sample and again adding all accumulated deltas gives average of accumulated deltas over all training samples
                acculated_deltas = (np.repeat(activation_value_a.T, forward_layer_deltas.shape[0],axis=0) * forward_layer_deltas)
                #print("actib",activation_value_a.T)
                #print("accumaltelayer",acculated_deltas)
                self.network_gradientdelta[layer_index] += acculated_deltas

                #calculating accumulated deltas @layer_1
                if layer_index-1==0:
                    activation_value_a = forward_propagation_output[layer_index-1].reshape(-1, 1)
                    forward_layer_deltas = _layer_delta[1:].reshape(-1, 1)
                    acculated_deltas = np.repeat(activation_value_a.T, forward_layer_deltas.shape[0],
                                                 axis=0) * forward_layer_deltas
                    #self.network_gradientdelta.insert(0, acculated_deltas)
                    #print("accumulated",acculated_deltas)
                    self.network_gradientdelta[layer_index-1] += acculated_deltas



        #return np.array(self.network_gradientdelta),np.array(self.network_deltas)
        return self.cost_function,self.loss



    def update_weights(self,learning_rate,_lambda,Regularization):
        # print("intiial weight",self.weights_Layers)
        # print("accumulate delta",self.network_gradientdelta)
        old_weights=self.weights_Layers.copy()
        updated_weights=[0.0] * self.num_layers
        regularized_sum_layer_weights=0
        layer_index=0
        for layer_weights,layer_accumulated_deltas in zip(self.weights_Layers,self.network_gradientdelta):
            #print("layer weights",layer_weights)
            #print("layer accumuated delta",layer_accumulated_deltas)
            #print(layer_accumulated_deltas)
            #print(learning_rate)
            if Regularization==True:
                #performing ridge Regularization
                #calulating sum of square of weights at each layer(np.sum(layer_weights[:,1:]**2) and adding this value to another layer sum of square of weights
                regularized_sum_layer_weights+=np.sum(layer_weights[:,1:]**2)
                # penality term lambda/m * theata
                penality=_lambda*(layer_weights)/self.M
                # making first column or bias term penality zero
                penality[:,:1]=0
                updated_weights[layer_index] = layer_weights - (learning_rate * ((layer_accumulated_deltas/self.M)+penality))
            elif Regularization==False:
                updated_weights[layer_index] = layer_weights - (learning_rate*(layer_accumulated_deltas/self.M))
            layer_index+=1

        self.weights_Layers=updated_weights.copy()
        #print("updated weights",updated_weights)
        #print("copied updated weights",self.weights_Layers)
        return old_weights,self.weights_Layers,regularized_sum_layer_weights

            #layer_weights[:,1:] #fetch all columns except first
            #theta_square[:,:1]=0 #making first column or bias term penality zero


    def train_network(self,class_object,learning_rate,_lambda,Regularization,epoch):
        # class_object.input.T each row is a training sample with four features
        train_cost_model=[]
        train_loss=[]
        validation_cost_model = []
        validation_loss = []
        for _epoch in range(0,epoch):
            #print("epoch",epoch)
            self.network_gradientdelta = [0.0] * self.num_layers  # accumulated deltas
            J_theta=0
            loss=0
            #print(obj.weights_Layers)
            for each_training_sample_x, each_training_sample_y in zip(class_object.input.T, class_object.output.T):
                #print(each_training_sample_x, each_training_sample_y)

                activation_z = class_object.forward_propagation(each_training_sample_x)
                #print(each_training_sample_x,activation_z)
                cost_sample,loss_sample=class_object.back_propagate_error(activation_z, each_training_sample_y)
                J_theta+=cost_sample
                loss+=loss_sample
                #print("acc", accdeltas)
                # print("del",deltas)
            J_theta_epoch=-(J_theta / self.M)
            loss_epoch=loss/self.M
            #update weights
            #print("learning rate",learning_rate)
            old_weights,new_weights,regularized_sum_layer_weights=self.update_weights(learning_rate,_lambda,Regularization)

            _,_, _, v_cost, v_loss = class_object.Predict_NN(validation_data_x, validation_data_y, old_weights)

            #Regularized and normal models
            if Regularization==False: #non regularized
                train_cost_model.append(J_theta_epoch)
                train_loss.append(loss_epoch)
                validation_cost_model.append(v_cost)
                validation_loss.append(v_loss)
            elif Regularization==True:
                regularied_J_theta=J_theta_epoch+((_lambda/(2*self.M))*regularized_sum_layer_weights)
                V_regularied_J_theta=v_cost+((_lambda/(2*self.M))*regularized_sum_layer_weights)
                train_cost_model.append(regularied_J_theta)
                train_loss.append(loss_epoch)
                validation_cost_model.append(V_regularied_J_theta)
                validation_loss.append(v_loss)

            # print("training costfunction",_epoch, J_theta_epoch)
            # print("training loss", _epoch, loss_epoch)
            # print("validation costfunction",_epoch,v_cost)
            # print("validation loss", _epoch, v_loss)

        # print("old weights")
        # print(old_weights)
        #print("new weights")
        #print(new_weights)

        return self.weights_Layers,train_cost_model,train_loss,validation_cost_model,validation_loss

        # predict values
    def Predict_NN(self, test_input,test_expected_output,model_weights):
            test_inputs = np.array(test_input)
            test_size=test_inputs.shape[-1]
            expected_output_y = np.array(test_expected_output)
            Predicted_output = []
            predicted_label=[]
            J_theta_test_epoch=0
            loss_test_epoch=0
            for each_test_sample_x, each_test_output_y in zip(test_inputs.T, expected_output_y.T):
                layer_input=each_test_sample_x
                #print(layer_input)
                # print(model_weights)

                for eachlayer_weight in model_weights:
                    #calcuate sum of product weight * input using Z_vector_function()
                    test_Z_vector = self.Z_vector_function(eachlayer_weight, self.add_bias_term(layer_input))
                    test_a_vector = self.sigmoid_function(test_Z_vector)
                    layer_input = test_a_vector.copy()
                    #output is column matrix
                #print(layer_input)
                Predicted_output.append(layer_input)
                predicted_label.append(np.argmax(layer_input))

                final_prediction=layer_input.reshape(-1,1)
                #calculating cost function
                J_theta_test_epoch += self.sigmoidal_costFunction_persample(each_test_output_y.reshape(-1,1), final_prediction)
                loss_test_epoch += np.sum(np.square(each_test_output_y.reshape(-1,1) - final_prediction))
            #print(Predicted_output)

            # print(np.array(self.activation_Z))
            #np.max(predicted_output_values,axis=1)    return max probability of all data samples
            return np.array(Predicted_output) ,np.argmax(expected_output_y.T,axis=1),np.array(predicted_label),-(J_theta_test_epoch / test_size) , loss_test_epoch /test_size


def data_split():
    data_path = "/Users/vaishnaviv/PycharmProjects/Assignment3_NeuralNetworks/BSOM_DataSet_for_HW3.csv"
    __data_raw = pd.read_csv(data_path)
    __data = __data_raw[['all_mcqs_avg_n20', 'all_NBME_avg_n4', 'CBSE_01', 'CBSE_02', 'LEVEL']]
    # #__data = __data_raw[['all_mcqs_avg_n20', 'all_NBME_avg_n4', 'CBSE_01', 'CBSE_02','STEP_1', 'LEVEL']]
    # __data = __data_raw[['all_mcqs_avg_n20', 'all_NBME_avg_n4', 'CBSE_01', 'CBSE_02','STEP_1','SA_NBME', 'LEVEL']]
    print(__data.describe())
    __data = __data.dropna()

    'split data into training set and test set'
    data_copy = __data.copy()

    train_data = data_copy.sample(frac=0.80, random_state=seed)
    test_data = data_copy.drop(train_data.index)
    train_copy=train_data.copy()
    train_data=train_copy.sample(frac=0.80, random_state=seed)
    validation_data=train_copy.drop(train_data.index)

    train_data_level = train_data['LEVEL']
    test_data_level = test_data['LEVEL']
    validation_data_level = validation_data['LEVEL']

    train_data = train_data.drop(['LEVEL'], axis=1)
    test_data = test_data.drop(['LEVEL'], axis=1)
    validation_data = validation_data.drop(['LEVEL'], axis=1)

    'Z-normalization for scaling'
    train_data_scaled = (train_data - train_data.mean()) / train_data.std()
    train_data = pd.concat([train_data_scaled, train_data_level], axis=1)

    test_data_scaled = (test_data - test_data.mean()) / test_data.std()
    test_data = pd.concat([test_data_scaled, test_data_level], axis=1)

    validation_data_scaled = (validation_data - validation_data.mean()) / validation_data.std()
    validation_data = pd.concat([validation_data_scaled, validation_data_level], axis=1)

    'Label A,B,C,D to 0,1,2,3'
    train_data_LEVELS = np.array(train_data['LEVEL'].astype('category').cat.codes)
    test_data_LEVELS = np.array(test_data['LEVEL'].astype('category').cat.codes)
    validation_data_LEVELS = np.array(validation_data['LEVEL'].astype('category').cat.codes)
    # print(__data_LEVELS)
    print(train_data.describe())
    'split LEVELS into LEVEL_A,LEVEL_B,LEVEL_C,LEVEL_D'
    train_data = pd.get_dummies(train_data, prefix=['LEVEL'])
    test_data = pd.get_dummies(test_data, prefix=['LEVEL'])
    validation_data = pd.get_dummies(validation_data, prefix=['LEVEL'])
    # print(__data)
    # to be removed after coding
    # __data.insert(0, 'LEVEL_A', 0)
    # __data.insert(0, 'LEVEL_C', 0)
    # # # __data.insert(0, 'LEVEL_D', 0)
    # __data.insert(0, 'LEVEL_D', 0)
    print(train_data.describe())
    train_data['LEVEL'] = train_data_LEVELS
    test_data['LEVEL'] = test_data_LEVELS
    validation_data['LEVEL'] = validation_data_LEVELS


    # print(test_data.describe())
    # print(np.unique(train_data[["LEVEL"]],return_counts=True))
    # print(np.unique(test_data[["LEVEL"]], return_counts=True))

    print(np.unique(train_data[["LEVEL"]],return_counts=True))
    print(np.unique(validation_data[["LEVEL"]], return_counts=True))
    print(np.unique(test_data[["LEVEL"]], return_counts=True))

    return train_data, test_data,validation_data


def plot_epoch_Costfunction(model_cost,Regularization,_label,hidden_layers):
    plt.plot(model_cost, label=_label)
    plt.ylabel("cost/error")
    plt.xlabel("epoch")
    #plt.axvline(x=min_value, ymin=0.25, ymax=0.75)
    plt.title("Costfunction graph with #hidden nodes {} ,# hidden layers {} intial weights = 0".format(Regularization,hidden_layers))
    #plt.show()

def plot_lambda_loss(_lambda,model_loss,_label,hidden_neurons):
    plt.plot(_lambda,model_loss, label=_label)
    plt.ylabel("cost/error")
    plt.xlabel("lambda")
    plt.legend()
    plt.title("cost/error vs epoch  @lambda {} with {} hidden neurons".format(_lambda,hidden_neurons))

class Metrics:
    def __init__(self,predicted_output,true_output,predicted_scores):
            self.predicted=predicted_output
            self.true=true_output
            self.predicted_scores=predicted_scores

    def confusion_matrix(self):
        print(confusion_matrix(y_true=self.true,y_pred=self.predicted))
        print(classification_report(y_true=self.true,y_pred=self.predicted))
        print()

    def roc_auc(self):
        print("roc_auc",roc_auc_score(y_true=to_categorical(self.true),y_score=self.predicted_scores))

def data_correaltion():
    data_path = "/Users/vaishnaviv/PycharmProjects/Assignment3_NeuralNetworks/BSOM_DataSet_for_HW3.csv"
    __data_raw = pd.read_csv(data_path)
    __data = __data_raw[['all_mcqs_avg_n20', 'all_NBME_avg_n4', 'CBSE_01', 'CBSE_02','SA_NBME','STEP_1', 'LEVEL']]
    print(__data_raw.columns.tolist())
    #__data = __data_raw[['O1_PI_01', 'O1_PI_02', 'O1_PI_03', 'O1_PI_04', 'O1_PI_05', 'O1_PI_06', 'O1_PI_07', 'O1_PI_08', 'O1_PI_09', 'O1_PI_10', 'O1_PI_11', 'O1_PI_12', 'O1_PI_13', 'O2_PI_01', 'O2_PI_02', 'O2_PI_03', 'O2_PI_04', 'O2_PI_05', 'O2_PI_06', 'O2_PI_07', 'O2_PI_08', 'O2_PI_09', 'O2_PI_10', 'O2_PI_11', 'O2_PI_12', 'O2_PI_13', 'HA_PI_01', 'HA_PI_02', 'HA_PI_03', 'HA_PI_04', 'HD_PI_01', 'HD_PI_02', 'HD_PI_03', 'HD_PI_04', 'HD_PI_05', 'HD_PI_06', 'HD_PI_07', 'HD_PI_08', 'HD_PI_09', 'HD_PI_10', 'HD_PI_11', 'HD_PI_12', 'HD_PI_13', 'HD_PI_14', 'HD_PI_15', 'SA_PI_01', 'SA_PI_02', 'SA_PI_03', 'SA_PI_04', 'SA_PI_05', 'SA_PI_06', 'SA_PI_07', 'SA_PI_08', 'SA_PI_09', 'SA_PI_10', 'SA_PI_11', 'SA_PI_12', 'SA_PI_13', 'SA_PI_14', 'SA_PI_15', 'SA_PI_16', 'SA_PI_17', 'SA_PI_18', 'SA_PI_19', 'SA_PI_20', 'SA_PI_21', 'SA_PI_22', 'SA_PI_23', 'SA_PI_24', 'SA_PI_25', 'SA_PI_26', 'B2E_PI_01', 'B2E_PI_02', 'B2E_PI_03', 'B2E_PI_04', 'B2E_PI_05', 'B2E_PI_06', 'B2E_PI_07', 'B2E_PI_08', 'B2E_PI_09', 'B2E_PI_10', 'B2E_PI_11', 'B2E_PI_12', 'B2E_PI_13', 'B2E_PI_14', 'B2E_PI_15', 'B2E_PI_16', 'B2E_PI_17', 'B2E_PI_18', 'B2E_PI_19', 'B2E_PI_20', 'B2E_PI_21', 'B2E_PI_22', 'B2E_PI_23', 'B2E_PI_24', 'B2E_PI_25', 'B2E_PI_26', 'B2E_PI_27', 'B2E_PI_28', 'B2E_PI_29', 'B2E_PI_30', 'BCR_PI_01', 'BCR_PI_02', 'BCR_PI_03', 'BCR_PI_04', 'BCR_PI_05', 'BCR_PI_06', 'BCR_PI_07', 'BCR_PI_08', 'BCR_PI_09', 'BCR_PI_10', 'BCR_PI_11', 'BCR_PI_12', 'BCR_PI_13', 'BCR_PI_14', 'BCR_PI_15', 'BCR_PI_16', 'BCR_PI_17', 'BCR_PI_18', 'BCR_PI_19', 'BCR_PI_20', 'BCR_PI_21', 'BCR_PI_22', 'BCR_PI_23', 'BCR_PI_24', 'BCR_PI_25', 'BCR_PI_26', 'BCR_PI_27', 'BCR_PI_28', 'BCR_PI_29', 'BCR_PI_30', 'BCR_PI_31', 'O1_IRAT_01', 'O1_IRAT_02', 'O1_IRAT_03', 'O1_IRAT_04', 'O1_IRAT_05', 'O1_IRAT_06', 'O1_IRAT_07', 'O1_IRAT_08', 'O1_IRAT_09', 'O1_IRAT_10', 'O1_IRAT_11', 'O1_IRAT_12', 'O2_IRAT_01', 'O2_IRAT_02', 'HA_IRAT_01', 'HA_IRAT_02', 'HD_IRAT_01', 'HD_IRAT_02', 'SA_IRAT_01', 'SA_IRAT_02', 'SA_IRAT_03', 'SA_IRAT_04', 'SA_IRAT_05', 'SA_IRAT_06', 'SA_IRAT_07', 'B2E_IRAT_01', 'B2E_IRAT_02', 'B2E_IRAT_03', 'B2E_IRAT_04', 'B2E_IRAT_05', 'B2E_IRAT_06', 'BCR_IRAT_01', 'BCR_IRAT_02', 'BCR_IRAT_03', 'O1_MCQ1_IND', 'O1_MCQ1_GRP', 'O1_MCQ1_TOT', 'O1_MCQ2_IND', 'O1_MCQ2_GRP', 'O1_MCQ2_TOT', 'O1_MCQ3_IND', 'O1_MCQ3_GRP', 'O1_MCQ3_TOT', 'O2_MCQ1_IND', 'O2_MCQ1_GRP', 'O2_MCQ1_TOT', 'O2_MCQ2_IND', 'O2_MCQ2_GRP', 'O2_MCQ2_TOT', 'O2_MCQ3_IND', 'O2_MCQ3_GRP', 'O2_MCQ3_TOT', 'HD_MCQ1_IND', 'HD_MCQ1_GRP', 'HD_MCQ1_TOT', 'SA_MCQ1_IND', 'SA_MCQ1_GRP', 'SA_MCQ1_TOT', 'SA_MCQ2_IND', 'SA_MCQ2_GRP', 'SA_MCQ2_TOT', 'SA_MCQ3_IND', 'SA_MCQ3_GRP', 'SA_MCQ3_TOT', 'SA_MCQ4_IND', 'SA_MCQ4_GRP', 'SA_MCQ4_TOT', 'SA_MCQ5_IND', 'SA_MCQ5_GRP', 'SA_MCQ5_TOT', 'B2E_MCQ1_IND', 'B2E_MCQ1_GRP', 'B2E_MCQ1_TOT', 'B2E_MCQ2_IND', 'B2E_MCQ2_GRP', 'B2E_MCQ2_TOT', 'B2E_MCQ3_IND', 'B2E_MCQ3_GRP', 'B2E_MCQ3_GRP.1', 'B2E_MCQ4_IND', 'B2E_MCQ4_GRP', 'B2E_MCQ4_TOT', 'BCR_MCQ1_IND', 'BCR_MCQ1_GRP', 'BCR_MCQ1_TOT', 'BCR_MCQ2_IND', 'BCR_MCQ2_GRP', 'BCR_MCQ2_TOT', 'BCR_MCQ3_IND', 'BCR_MCQ3_GRP', 'BCR_MCQ3_TOT', 'BCR_MCQ4_IND', 'BCR_MCQ4_GRP', 'BCR_MCQ4_TOT', 'BCR_NBME_final', 'B2E_NBME_final', 'O1_O2_NBME', 'SA_NBME', 'HA_final', 'HD_final', 'all_NBME_avg_n4', 'all_mcqs_avg_n20', 'O1_PI_AVG_13', 'O2_PI_AVG_13', 'O1O2_PI_AVG_26', 'HA_PI_AVG_04', 'HD_PI_AVG_15', 'SA_PI_AVG_26', 'B2E_PI_AVG_30', 'BCR_PI_AVG_30', 'O1_IRAT_AVG_12', 'O2_IRAT_AVG_02', 'HA_IRAT_AVG_02', 'HD_IRAT_AVG_02', 'SA_IRAT_AVG_07', 'B2E_IRAT_AVG_06', 'BCR_IRAT_AVG_03', 'O1_MCQ_AVG_03', 'O2_MCQ_AVG_03', 'HD_MCQ_AVG_01', 'SA_MCQ_AVG_05', 'B2E_MCQ_AVG_04', 'BCR_MCQ_AVG_04', 'BCR_ANAT_MCQ_AVG_02', 'CBSE_01', 'CBSE_02', 'STEP_1', 'LEVEL']]
    #__data=__data.dropna()
    # corelatiodata = __data.corr(method="spearman")
    # print(corelatiodata)
    # c = corrplot.Corrplot(corelatiodata)
    # c.plot(colorbar=False, method="square", shrink=.9, rotation=45)
    # plt.show()
    __data_LEVEL=__data.LEVEL.astype("category").cat.codes
    __data=__data.drop(['LEVEL'], axis=1)
    __data['LEVEL']=__data_LEVEL
    #spearman corealtion
    print(__data)
    corelatiodata=__data.corr(method="spearman")
    print(corelatiodata['LEVEL'].sort_values())
    c=corrplot.Corrplot(corelatiodata)
    c.plot(colorbar=False,method="square",shrink=.9,rotation=45)
    plt.show()
    #pearson corealtion
    corelatiodata = __data.corr()
    c = corrplot.Corrplot(corelatiodata)
    c.plot(colorbar=False, method="square", shrink=.9, rotation=45)
    plt.show()




if __name__=="__main__":
    seed=121
    np.random.seed(seed)
    # 1a feature set1
    #data_correaltion()
    train_data, test_data ,validation_data = data_split()
    #print(train_data)
    #set 1
    train_data_x=[train_data['all_mcqs_avg_n20'],train_data['all_NBME_avg_n4'],train_data['CBSE_01'],train_data['CBSE_02']]
    train_data_y=[train_data['LEVEL_A'],train_data['LEVEL_B'],train_data['LEVEL_C'],train_data['LEVEL_D']]

    test_data_x = [test_data['all_mcqs_avg_n20'], test_data['all_NBME_avg_n4'], test_data['CBSE_01'],test_data['CBSE_02']]
    test_data_y = [test_data['LEVEL_A'], test_data['LEVEL_B'], test_data['LEVEL_C'], test_data['LEVEL_D']]

    validation_data_x = [validation_data['all_mcqs_avg_n20'], validation_data['all_NBME_avg_n4'], validation_data['CBSE_01'],
                    validation_data['CBSE_02']]
    validation_data_y = [validation_data['LEVEL_A'], validation_data['LEVEL_B'], validation_data['LEVEL_C'], validation_data['LEVEL_D']]

    #set 2:
    # train_data_x = [train_data['all_mcqs_avg_n20'], train_data['all_NBME_avg_n4'], train_data['CBSE_01'],
    #                 train_data['CBSE_02'],train_data['STEP_1']]
    # train_data_y = [train_data['LEVEL_A'], train_data['LEVEL_B'], train_data['LEVEL_C'], train_data['LEVEL_D']]
    #
    # test_data_x = [test_data['all_mcqs_avg_n20'], test_data['all_NBME_avg_n4'], test_data['CBSE_01'],
    #                test_data['CBSE_02'],test_data['STEP_1']]
    # test_data_y = [test_data['LEVEL_A'], test_data['LEVEL_B'], test_data['LEVEL_C'], test_data['LEVEL_D']]
    #
    # validation_data_x = [validation_data['all_mcqs_avg_n20'], validation_data['all_NBME_avg_n4'],
    #                      validation_data['CBSE_01'],
    #                      validation_data['CBSE_02'],validation_data['STEP_1']]
    # validation_data_y = [validation_data['LEVEL_A'], validation_data['LEVEL_B'], validation_data['LEVEL_C'],
    #                      validation_data['LEVEL_D']]

    #set 3:
    # train_data_x = [train_data['all_mcqs_avg_n20'], train_data['all_NBME_avg_n4'], train_data['CBSE_01'],
    #                 train_data['CBSE_02'],train_data['STEP_1'], train_data['SA_NBME']]
    # train_data_y = [train_data['LEVEL_A'], train_data['LEVEL_B'], train_data['LEVEL_C'], train_data['LEVEL_D']]
    #
    # test_data_x = [test_data['all_mcqs_avg_n20'], test_data['all_NBME_avg_n4'], test_data['CBSE_01'],
    #                test_data['CBSE_02'],test_data['STEP_1'], test_data['SA_NBME']]
    # test_data_y = [test_data['LEVEL_A'], test_data['LEVEL_B'], test_data['LEVEL_C'], test_data['LEVEL_D']]
    #
    # validation_data_x = [validation_data['all_mcqs_avg_n20'], validation_data['all_NBME_avg_n4'],
    #                      validation_data['CBSE_01'],
    #                      validation_data['CBSE_02'],validation_data['STEP_1'], validation_data['SA_NBME']]
    # validation_data_y = [validation_data['LEVEL_A'], validation_data['LEVEL_B'], validation_data['LEVEL_C'],
    #                      validation_data['LEVEL_D']]

    #1A
    hidden_nodes=[1]
    hidden_layers=1
    hidden_nodes = [2]
    for n_hidden_nodes in hidden_nodes:
        print("number of hidden nodes",n_hidden_nodes)
        print("number of hidden layers is {}".format(hidden_layers))
        obj=NeuralNetwork(train_data_x,train_data_y,n_hidden_nodes,4)
        print(obj.weights_Layers)
        Regularization = False
        #run the model to fix number of epoch where train and validation error is minimum possible and difference is nearly zero
        model_weights,model_cost,model_loss,validation_cost,validation_loss=obj.train_network(obj,learning_rate=0.5,_lambda=0.6,Regularization=False,epoch=4000)
        #0.0004851
        #print(model_weights)
        print("training model min loss", np.min(model_loss), np.argmin(model_loss))
        print("validation model min loss", np.min(validation_loss), np.argmin(validation_loss))
        print("training model min cost", np.min(model_cost), np.argmin(model_cost))
        print("validation model min cost", np.min(validation_cost), np.argmin(validation_cost))
        plot_epoch_Costfunction(model_cost, n_hidden_nodes, "Train_cost",hidden_layers)
        plot_epoch_Costfunction(validation_cost, n_hidden_nodes, "Validation_cost",hidden_layers)
        plt.axvline(x=np.argmin(validation_cost), ymax=np.min(validation_cost), linestyle='--', color='C1',linewidth=0.5,label='Minimum_cost_epoch')
        plt.axhline(y=np.min(validation_cost),linestyle='--',color='C3',linewidth=0.5,label='Minimum_validation_cost')
        plt.legend()
        plt.show()
        # plot_epoch_Costfunction(model_loss, Regularization, "Train_loss")
        # plot_epoch_Costfunction(validation_loss, Regularization, "Validation_loss")
        #plt.axvline(x=np.argmin(validation_loss),ymax=np.min(validation_loss), linestyle='--', color='r')
        # plt.show()
        epoch=np.argmin(validation_cost)
        del obj
        print("train model with given epoch as stop criteria")
        print(epoch)
    Regularization=False
    model_obj = NeuralNetwork(train_data_x, train_data_y, 2,4)
    print(model_obj.weights_Layers)
    _model_weights, _model_cost, _model_loss, _validation_cost, _validation_loss = model_obj.train_network(model_obj, learning_rate=0.5,
                                                                                                _lambda=0,
                                                                                                Regularization=Regularization,
                                                                                                epoch=673)
    print(_model_weights)
    print("training model min loss", np.min(_model_loss), np.argmin(_model_loss))
    print("validation model min loss", np.min(_validation_loss), np.argmin(_validation_loss))
    print("training model min cost", np.min(_model_cost), np.argmin(_model_cost))
    print("validation model min cost", np.min(_validation_cost), np.argmin(_validation_cost))
    plot_epoch_Costfunction(_model_cost,Regularization,"Train_cost",1)
    plot_epoch_Costfunction(_validation_cost,Regularization,"Validation_cost",1)
    # s
    plt.show()

    #making prediction on new data and analysing metrics
    predicted_output_values,expected_output,predicted_label,test_cost,test_loss=model_obj.Predict_NN(test_data_x,test_data_y,_model_weights)

    #print("predocted",predicted_output_values)
    print(expected_output)
    print(predicted_label)
    Metricsobj_test = Metrics(predicted_label, expected_output,predicted_output_values)
    print("Test predictions")
    Metricsobj_test.confusion_matrix()
    Metricsobj_test.roc_auc()

    predicted_output, expected_output, predicted_label,train_cost,train_loss = model_obj.Predict_NN(train_data_x,train_data_y, _model_weights)
    #print(predicted_output)
    print(expected_output)
    print(predicted_label)
    Metricsobj_train=Metrics(predicted_label, expected_output,predicted_output)
    print("TrainPrediction")
    Metricsobj_train.confusion_matrix()
    Metricsobj_train.roc_auc()

    # making prediction on new data and analysing metrics  VALIDATION data
    predicted_output_values, expected_output, predicted_label, validation_cost, validation_loss = model_obj.Predict_NN(validation_data_x,
                                                                                                           validation_data_y,
                                                                                                           _model_weights)

    # print("predocted",predicted_output_values)
    print(expected_output)
    print(predicted_label)
    Metricsobj_validate = Metrics(predicted_label, expected_output, predicted_output_values)
    print("validation predictions")
    Metricsobj_validate.confusion_matrix()
    Metricsobj_validate.roc_auc()
    # hidden_neurons = 2
    # # #plot lambda vs error
    # _lambda=[10,0.7,0.6,0.5,0.2,0.1,0,0.01]
    # error=[2.1363034588826424,1.8467649060741864,1.8274568358345802,1.8063122491393315,1.7073835246919715,1.657476816486276,1.5908250781037774,1.598546261414624]
    # train_error=[1.976255253641021,1.5707224875855434,1.5391326351978443,1.5028270452983494,1.341119453399623,1.2426421832719547,1.0198776725483012,1.0620512234293287]
    # plot_lambda_loss(_lambda,error,"Train_cost",hidden_neurons)
    # plot_lambda_loss(_lambda,train_error,"Validation_cost",hidden_neurons)
    # plt.show()
    #2 REGULAROZATION:
    model_obj = NeuralNetwork(train_data_x, train_data_y, 2, 4)
    hidden_neurons=2
    print("selecting best lambda")
    _lambdas=[0]
   #_lambdas=[10]
    train_regularized_model_cost=[]
    validation_regularized_model_cost= []
    for _lambda in _lambdas:
        Regularization = True
        model_weights, model_cost, model_loss, validation_cost, validation_loss = model_obj.train_network(model_obj, learning_rate=0.5,
                                                                                                    _lambda=_lambda,
                                                                                                    Regularization=True,
                                                                                                    epoch=2500)
        # 0.0004851
        #print(model_weights)
        print("train data model min loss @ lambda",_lambda, np.min(model_loss), np.argmin(model_loss),model_loss[-1])
        print("validation data model min loss @ lambda", _lambda, np.min(validation_loss), np.argmin(validation_loss), validation_loss[-1])
        print("train data model min cost @ lambda", _lambda, np.min(model_cost), np.argmin(model_cost), model_cost[-1])
        print("validation data model min cpst @ lambda", _lambda, np.min(validation_cost), np.argmin(validation_cost), validation_cost[-1])
        train_regularized_model_cost.append(model_cost[-1])
        validation_regularized_model_cost.append(validation_cost[-1])
        # plot_lambda_loss(_lambda,model_cost,"Train_cost",hidden_neurons)
        # plot_lambda_loss(_lambda,validation_cost,"Validation_cost",hidden_neurons)
        # plt.axhline(y=np.min(validation_cost),linestyle='--',color='C3',linewidth=0.6,label='epoch')
        # plt.axvline(x=np.argmin(validation_cost),linestyle='--', color='C0',linewidth=0.6,label='min_validation_cost')
        # plt.show()
    plot_lambda_loss(_lambdas, train_regularized_model_cost, "Train_cost", hidden_neurons)
    plot_lambda_loss(_lambdas, validation_regularized_model_cost, "Validation_cost", hidden_neurons)
    plt.show()
    #making prediction on new data and analysing metrics
    predicted_output_values,expected_output,predicted_label,test_cost,test_loss=model_obj.Predict_NN(test_data_x,test_data_y,model_weights)

    #print("predocted",predicted_output_values)
    print(expected_output)
    print(predicted_label)
    Metricsobj = Metrics(predicted_label, expected_output,predicted_output_values)
    print("Test predictions")
    Metricsobj.confusion_matrix()
    #Metricsobj.roc_auc()

    predicted_output, expected_output, predicted_label,train_cost,train_loss = model_obj.Predict_NN(train_data_x,train_data_y, model_weights)
    #print(predicted_output)
    print(expected_output)
    print(predicted_label)
    Metricsobj=Metrics(predicted_label, expected_output,predicted_output_values)
    print("TrainPrediction")
    Metricsobj.confusion_matrix()
    #Metricsobj.roc_auc()

    # making prediction on new data and analysing metrics  VALIDATION data
    predicted_output_values, expected_output, predicted_label, validation_cost, validation_loss = model_obj.Predict_NN(validation_data_x,
                                                                                                           validation_data_y,
                                                                                                           model_weights)

    # print("predocted",predicted_output_values)
    print(expected_output)
    print(predicted_label)
    Metricsobj = Metrics(predicted_label, expected_output, predicted_output_values)
    print("validation predictions")
    Metricsobj.confusion_matrix()
    # Metricsobj.roc_auc()

