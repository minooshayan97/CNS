from PymoNNto import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_mean_vt(layer):
    plt.plot([np.mean(i) for i in layer['n.v',0]],color='blue')
    plt.xlabel('iterations')
    plt.ylabel('voltage')
    plt.show()


def plot_ut(layer, theta):
    plt.plot(layer['n.v', 0])
    plt.axhline(theta,color='black',linestyle='dashed')
    plt.xlabel('iterations')
    plt.ylabel('voltage')
    plt.show()

def raster_plot3D(coded_img):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    for z,step in enumerate(coded_img):
        for y,n1 in enumerate(step):
            for x,n2 in enumerate(n1):
                if coded_img[z][y][x]:
                    ax.scatter3D(x,y,z,c='black',s=2,alpha=0.5)
                    ax1.scatter3D(x,y,z,c='black',s=1,alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    ax.view_init()
    ax1.view_init(90, 90)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('t')
    plt.show()

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

def gabor(l, theta, sigma, gamma, size):
    xs, ys = make_axis(size)
    X = xs * np.cos(theta) + ys * np.sin(theta)
    Y = -xs * np.sin(theta) + ys * np.cos(theta)
    E = torch.exp(-(X ** 2 + gamma ** 2 * Y ** 2) / (2 * sigma ** 2))
    C = torch.cos((2 * np.pi / l) * X)
    return (E * C)

def convolve(img, f, mode):
    filter_prime = f if mode == 'on-center' else f * -1

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

def show_with_filter(filtered_img , filter1, mode):
    print('{} filter : '.format(mode))
    plt.imshow(filter1 if mode == 'on-center' else filter1 * -1, cmap='gray')
    plt.show()
    print('{}-filtered image: '.format(mode))
    plt.imshow(filtered_img, cmap='gray')
    plt.show()
    return

def plot_raster(l3, l2, l1, l1_size, l2_size):
    l1_fired = []
    for t in l1['n.fired', 0]:
        l1_fired_t = []
        for n in range(len(t)):
            if t[n]:
                l1_fired_t.append(n)
        l1_fired.append(l1_fired_t)
        
    l2_fired = []
    for t in l2['n.fired', 0]:
        l2_fired_t = []
        for n in range(len(t)):
            if t[n]:
                l2_fired_t.append(n + (l1_size**2))
        l2_fired.append(l2_fired_t)
        
    l3_fired = []
    for t in l3['n.fired', 0]:
        l3_fired_t = []
        for n in range(len(t)):
            if t[n]:
                l3_fired_t.append(n + (l1_size**2) + (l2_size**2))
        l3_fired.append(l3_fired_t)
        
    for i in range(len(l1_fired)):
        plt.scatter([i for j in range(len(l1_fired[i]))],l1_fired[i],color='blue',s=1)
    for i in range(len(l2_fired)):
        plt.scatter([i for j in range(len(l2_fired[i]))],l2_fired[i],color='orange',s=1)
    for i in range(len(l3_fired)):
        plt.scatter([i for j in range(len(l3_fired[i]))],l3_fired[i],color='green',s=1)
    plt.xlabel('iterations')
    plt.ylabel('neuron index')
    plt.show()

def pooling(img,size):
    rows = img.shape[0] + (-img.shape[0] % size)
    columns = img.shape[1] + (-img.shape[1] % size)
    _img = torch.zeros(rows//size , columns//size)
    
    for i in range(0,rows,size):
        for j in range(0,columns,size):
            _img[i//size, j//size] = (img[i: i + size, j: j + size]).max()
            
    return _img

def plot_layers(layer1, layer2, layer3, lsize1, lsize2, lsize3):
    print('3D raster plot plot for layer1(input) : ')
    l1 = []
    for step in layer1['n.fired'][0]:
        l1.append(step.reshape((lsize1,lsize1)))
    raster_plot3D(l1)
    print('3D raster plot plot for layer2(convolution) : ')
    l2 = []
    for step in layer2['n.fired'][0]:
        l2.append(step.reshape((lsize2, lsize2)))
    raster_plot3D(l2)
    print('3D raster plot plot for layer3 (pooling): ')
    l3 = []
    for step in layer3['n.fired'][0]:
        l3.append(step.reshape(((lsize3), (lsize3))))
    raster_plot3D(l3)

def network_setup(params, par, img_tensor):
    coded_img = TimeToFirstSpikeEncoder(img_tensor, par['time'], par['dt'], par['min_val'], par['max_val'])
    img_dim = params['img_dim']

    class main(Behaviour):

        def set_variables(self, n):
            self.set_init_attrs_as_variables(n)
            n.v = n.get_neuron_vec('uniform') * n.v_rest
            n.fired = n.get_neuron_vec() < 0
            if params['kernel'] == 'DoG':
                self.kernel = DoG(par['std1'], par['std2'], par['filter_size'])
            elif params['kernel'] == 'Gabor':
                self.kernel = gabor(par['l'], par['theta'], par['sigma'], par['gamma'], par['filter_size'])
            self.padding = n.padding
            self.pooling_size = n.pooling_size

        def new_iteration(self, n):
            n.v *= 0.9
            n.fired = n.v > n.v_threshold
            if np.sum(n.fired) > 0:
                n.v[n.fired] = n.v_reset
                
            for neuron in n['first_layer_neurons']:
                temp = coded_img[n.iteration - 1].flatten()
                n.fired = np.array([n == 1 for n in temp])
                
            for synapse in n.afferent_synapses['conv']:
                
                f_map = torch.FloatTensor(list(map(
                    lambda i: 1 if i else 0, synapse.src.fired))).reshape((img_dim, img_dim))
                
                padding_added_f_map = torch.zeros(img_dim + self.padding * 2, img_dim+self.padding * 2)
                padding_added_f_map[self.padding:img_dim + self.padding, self.padding:img_dim + self.padding] = f_map
                
                temp2 = convolve(padding_added_f_map, self.kernel, 'on-center').flatten()
                n.v = np.array(temp2)
                
            for synapse in n.afferent_synapses['pool']:
                f_map = torch.FloatTensor(list(map(
                    lambda i: 1 if i else 0, synapse.src.fired))).reshape((img_dim, img_dim))

                temp3 = pooling(f_map, self.pooling_size).flatten()
                n.v = np.array(temp3)
                
            
    My_Network = Network()

    layer1 = NeuronGroup(net=My_Network,
                         tag='first_layer_neurons',
                         size=get_squared_dim(params['layer1_size']**2),
                         behaviour={
                             1: main(v_rest=params['layer1_rest'],
                                         v_reset=params['layer1_reset'],
                                         v_threshold=params['layer1_threshold'],pooling_size=params['pooling_size'],padding=params['padding']),
                             9: Recorder(tag='rec1', variables=['n.v', 'n.fired'])
                         })

    layer2 = NeuronGroup(net=My_Network,
                         tag='second_layer_neurons',
                         size=get_squared_dim(params['layer2_size']**2),
                         behaviour={
                             1: main(v_rest=params['layer2_rest'],
                                         v_reset=params['layer2_reset'],
                                         v_threshold=params['layer2_threshold'],pooling_size=params['pooling_size'],padding=params['padding']),
                             9: Recorder(tag='rec2', variables=['n.v', 'n.fired'])
                         })
    
    layer3 = NeuronGroup(net=My_Network,
                         tag='third_layer_neurons',
                         size=get_squared_dim(params['layer3_size']**2),
                         behaviour={
                             1: main(v_rest=params['layer3_rest'],
                                         v_reset=params['layer3_reset'],
                                         v_threshold=params['layer3_threshold'],pooling_size=params['pooling_size'],padding=params['padding']),
                             9: Recorder(tag='rec3', variables=['n.v', 'n.fired'])
                         })

    SynapseGroup(net=My_Network, src=layer1, dst=layer2, tag='conv')
    SynapseGroup(net=My_Network, src=layer2, dst=layer3, tag='pool')

    My_Network.initialize()
    My_Network.simulate_iterations(params['Iterations'], measure_block_time=True)

    print('u-t plot layer2 (convolution layer): ')
    plot_ut(layer2,params['layer2_threshold'])
    print('u-t plot layer3 (pooling layer): ')
    plot_ut(layer3,params['layer3_threshold'])

    print('mean u-t plot layer2 (convolution layer): ')
    plot_mean_vt(layer2)
    print('mean u-t plot layer3 (pooling layer): ')
    plot_mean_vt(layer3)

    print('2D raster plot plot for layer1(input, blue) , layer2(convolution, orange) , layer3 (pooling, green): ')
    plot_raster(layer3, layer2, layer1, params['layer1_size'], params['layer2_size'])

    plot_layers(layer1, layer2, layer3, params['layer1_size'], params['layer2_size'], params['layer3_size'])

    return
