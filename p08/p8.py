import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PymoNNto import *

def min_max_scaler(min_val, max_val, val):
    return (val - min_val) / (max_val - min_val)

def TimeToFirstSpikeEncoder(img, time, dt, min_val, max_val):
    shape = img.shape
    steps = int(time / dt)
    coded_img = torch.zeros(steps, *shape)
        
    img_scaled_flatten = [int(min_max_scaler(min_val, max_val, val) * steps) for val in img.flatten()]

    for i in range(steps):
        spikes = torch.tensor(list(map(lambda x: 1 if x == steps - i else 0, img_scaled_flatten))).view(shape)
        coded_img[i] = spikes
    return coded_img

def convolve_(img, f):
    filter_prime = f

    img_rows = img.shape[0]
    img_columns = img.shape[1]
    
    filter_rows = f.shape[0]
    filter_columns = f.shape[1]
    
    filtered_img = torch.zeros(img_rows - filter_rows + 1, img_columns - filter_columns + 1)
    rows = filtered_img.shape[0]
    columns = filtered_img.shape[1]
    
    for i in range(rows):
        for j in range(columns):
            filtered_img[i, j] = (filter_prime * img[i: i + filter_rows, j: j + filter_columns]).sum() 
    return filtered_img

def gaussian_dist(xs, ys, std):
    return torch.exp(-(xs ** 2 + ys ** 2) / (2 * std)) / (std * np.sqrt(2 * np.pi))

def make_axis(size):
    n = int((size - 1) / 2) 
    axis_range = torch.arange(-n, n + 1)
    xs, ys = torch.meshgrid([axis_range, axis_range])
    return (xs, ys)

def make_gaussian_filter(size, std):
    xs, ys = make_axis(size)
    return gaussian_dist(xs, ys, std)

def DoG(std_1, std_2, size):
    g1 = make_gaussian_filter(size, std_1)
    g2 = make_gaussian_filter(size, std_2)
    return (g1 - g2)

class STDP(Behaviour):

    def set_variables(self, synapses):
        self.add_tag('STDP')
        self.set_init_attrs_as_variables(synapses)
        self.pre = synapses.src
        self.posts = synapses.dst
        self.feature_map = features
        self.lr = synapses.lr
        self.f_size = len(features[0])
        
    def new_iteration(self, s):       
        for idx, winner in enumerate(s.winners):
            pre_traces = self.pre.v.reshape((img_dim,img_dim))[
                winner[1]: winner[1] + self.f_size, winner[0]: winner[0] + self.f_size]
            pre_spikes = self.pre.fired.reshape((img_dim,img_dim))[
                winner[1]: winner[1] + self.f_size, winner[0]: winner[0] + self.f_size]
            
            post_trace = self.posts.v.reshape((img_dim,img_dim))[winner[0] ,winner[1]]
            post_spike = self.posts.fired.reshape((img_dim,img_dim))[winner[0] ,winner[1]]
            
            dw = -self.lr * post_trace * pre_spikes + self.lr * pre_traces * post_spike
            dw[dw > 0] = self.lr
            dw[dw < 0] = -self.lr

            tempdw = np.concatenate((dw,np.zeros(shape=(dw.shape[0],self.f_size-dw.shape[1]))),axis=1)
            dw = np.concatenate((tempdw,np.zeros(shape=(self.f_size-dw.shape[0],self.f_size))),axis=0)
            self.feature_map[idx] += torch.tensor(dw)

class WinnerTakeAll(Behaviour):

    def set_variables(self, synapses):
        self.set_init_attrs_as_variables(synapses)
        self.feature_map = features
        self.out = synapses.src
        self.k = synapses.k
        synapses.winners = []
        self.f_size = len(features)
        
    def new_iteration(self, synapses):
        if len(synapses.winners) == self.k:
            return
        for idx, f_map in enumerate(self.feature_map):
            if idx >= len(synapses.winners):
                tmp = sorted(enumerate(self.out.v.flatten()), key=lambda x: x[1], reverse=True)
                for idx_prime, i in tmp:
                    if self.out.fired.flatten()[idx_prime] == 1:
                        tmp2 = np.unravel_index([idx_prime], (img_dim,img_dim))
                        w_position = (tmp2[0][0], tmp2[1][0])
                        if not self.is_in_inhibition_position(synapses.winners, w_position):
                            synapses.winners.append(w_position)
                            return
                            
    
    def is_in_inhibition_position(self, winners, p):
        for w in winners:
            if ((w[0] - self.f_size < p[0] < w[0] + self.f_size) and (w[1] - self.f_size < p[1] < w[1] + self.f_size)):
                return True           
        return False

def network_sim(features, coded_img, k, lr, params, kernel):
    
    class main(Behaviour):
        
        def set_variables(self, n):
            self.set_init_attrs_as_variables(n)
            n.v = n.get_neuron_vec('uniform') * n.v_rest
            n.fired = n.get_neuron_vec() < 0
            self.padding = n.padding
            self.kernel = kernel

        def new_iteration(self, n):
            n.v *= 0.9
            n.fired = n.v > n.v_threshold
            n.v += ((-(n.v - n.v_rest) + n.R * n.I) * par['dt']) / n.tau
            if np.sum(n.fired) > 0:
                n.v[n.fired] = n.v_reset

            for neuron in n['input_layer_neurons']:
                temp = coded_img[n.iteration - 1].flatten()
                n.fired = np.array([n == 1 for n in temp])

            for synapse in n.afferent_synapses['conv']:
                f_map = torch.FloatTensor(list(map(
                    lambda i: 1 if i else 0, synapse.src.fired))).reshape((img_dim, img_dim))
                padding_added_f_map = torch.zeros(img_dim + self.padding * 2, img_dim+self.padding * 2)
                padding_added_f_map[self.padding:img_dim + self.padding, self.padding:img_dim +
                                    self.padding] = f_map
                temp2 = convolve_(padding_added_f_map, self.kernel).flatten()
                n.v = np.array(temp2)

            for synapse in n.afferent_synapses['k_winner']:
                pass


    My_Network = Network()

    layer0 = NeuronGroup(net=My_Network,
                            tag='input_layer_neurons',
                            size=get_squared_dim(params['layer1_size']**2),
                            behaviour={
                                1: main(v_rest=params['layer1_rest'],
                                        v_reset=params['layer1_reset'],
                                        v_threshold=params['layer1_threshold'],
                                        R=params['layer1_R'],
                                        I=params['layer1_I'],
                                        tau=1,
                                        padding=params['padding']),
                                9: Recorder(tag='rec1', variables=['n.v', 'n.fired','n.I'])
                            })

    layer1 = NeuronGroup(net=My_Network,
                            tag='first_layer_neurons',
                            size=get_squared_dim(params['layer2_size']**2),
                            behaviour={
                                1: main(v_rest=params['layer2_rest'],
                                        v_reset=params['layer2_reset'],
                                        v_threshold=params['layer2_threshold'],
                                        R=params['layer2_R'],
                                        I=params['layer2_I'],
                                        tau=1,
                                        padding=params['padding']),
                                9: Recorder(tag='rec2', variables=['n.v', 'n.fired','n.I'])
                            })

    layer2 = [NeuronGroup(net=My_Network,
                            tag='second_layer_neurons',
                            size=get_squared_dim(params['layer3_size']**2),
                            behaviour={
                                1: main(v_rest=params['layer3_rest'],
                                        v_reset=params['layer3_reset'],
                                        v_threshold=params['layer3_threshold'],
                                        R=params['layer3_R'],
                                        I=params['layer3_I'],
                                        tau=1,
                                        padding=params['padding']),
                                9: Recorder(tag='rec3', variables=['n.v', 'n.fired','n.I'])
                            }) for i in range(len(features))]

    SynapseGroup(net=My_Network, src=layer0, dst=layer1, tag='conv')
    for l in layer2:
        SynapseGroup(net=My_Network, src=layer1, dst=l, tag='k_winner', behaviour={
            1:WinnerTakeAll(features=features,k=k),
            2: STDP(lr=lr)
        })

    My_Network.initialize()
    My_Network.simulate_iterations(params['Iterations'], measure_block_time=True)
    #return layer0, layer1, layer2
    return

def train_(coded_imgs, f):
    global features
    features = torch.clone(f)
    print('initial features:')
    for idx, feature in enumerate(features):
        print(f'feature {idx + 1}')
        plt.imshow(feature, cmap='gray')
        plt.show()

    for i in range(20):
        img = coded_imgs[i]
        #l0,l1,l2 = network_sim(features, img, feature_size)
        network_sim(features, img, params['feature_number'], params['lr'], params, DoG(par['std1'], par['std2'], par['filter_size']))

    print('features After training:')
    for idx, feature in enumerate(features):
        print(f'feature {idx + 1}')
        plt.imshow(feature, cmap='gray')
        plt.show()


img_dim = 60

par = {
    'std1' : 3,
    'std2' : 5,
    'filter_size' : 13,
    'min_val' : 0,
    'max_val' : 255,
    'time' : 50,
    'dt' : 1  
}

params = {
    'density' : ('random' , 0.1),
    'Iterations' : par['time'],
    'img_dim' : img_dim,
    
    'layer1_size' : img_dim,
    'layer2_size' : img_dim,
    'layer3_size' : img_dim,
    
    'layer1_threshold' : 0-1,
    'layer2_threshold' : 0.15,
    'layer3_threshold' : -1,
    
    'layer1_rest' : -5,
    'layer2_rest' : -5,
    'layer3_rest' : -5,
    
    'layer1_reset' : -10,
    'layer2_reset' : -10,
    'layer3_reset' : -10,
    
    'padding' : par['filter_size']//2,
    'kernel' : 'DoG',
    
    'img_dim' : 60,
    'feature_number' : 4,
    'feature_size' : 9,
    
    'layer1_R' : 1,
    'layer1_I' : 0,
    'layer2_R' : 1,
    'layer2_I' : 0,
    'layer3_R' : 1,
    'layer3_I' : 0,
    
    'lr' : 0.05
}

features = torch.rand(params['feature_number'], params['feature_size'], params['feature_size']) * 0.1 + 0.1