import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    #print(filter1)
    print('{} filter : '.format(mode))
    plt.imshow(filter1 if mode == 'on-center' else filter1 * -1, cmap='gray')
    plt.show()
    print('{}-filtered image: '.format(mode))
    plt.imshow(filtered_img, cmap='gray')
    plt.show()
    return

# from p04
def get_spiked_neurons(spikes):
    spiked_neurons = list(map(lambda x: x[0],filter(lambda x: x[1] == 1 ,enumerate(spikes))))
    return torch.tensor(spiked_neurons)

# from p04
def raster_plot(population_spikes, dt):
    b = 0
    for spikes_per_step in population_spikes:
        for step, spikes in enumerate(spikes_per_step):
            spikes_flatten = torch.flatten(spikes)
            spiked_neurons = get_spiked_neurons(spikes_flatten)
            plot_neuron_index = list(map(lambda x: x + b, spiked_neurons))
            plt.scatter([dt*step]*len(spiked_neurons),plot_neuron_index,c='black',s=2)
    
        b += len(spikes_flatten)
    plt.show()

# from p04
def min_max_scaler(min_val, max_val, val):
    return (val - min_val) / (max_val - min_val)

def raster_plot3D(coded_img):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    for z,step in enumerate(coded_img):
        for y,n1 in enumerate(step):
            for x,n2 in enumerate(n1):
                if coded_img[z][y][x] == 1:
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

# from p04
def TimeToFirstSpikeEncoder(img, time, dt, min_val, max_val):
    shape = img.shape
    steps = int(time / dt)
    coded_img = torch.zeros(steps, *shape)
        
    img_scaled_flatten = [int(min_max_scaler(min_val, max_val, val) * steps) for val in img.flatten()]

    for i in range(steps):
        spikes = torch.tensor(list(map(lambda x: 1 if x == steps - i else 0, img_scaled_flatten))).view(shape)
        coded_img[i] = spikes
    return coded_img

# from p04
def poisson_encoder(img, max_spikes, time, dt, max_val):
    steps = int(time/dt)
    shape = img.shape
    coded_img = torch.zeros(steps, *shape)
    img_flatten = img.flatten()

    for i in range(steps):
        spikes = torch.tensor(
            list(map(lambda x: 1 if np.random.rand()<(max_spikes/steps)*(x/max_val) else 0,img_flatten))).view(shape)

        coded_img[i] = spikes
    return coded_img

def present_DoG_filter(img_tensor, par):
    print('original image:')
    plt.imshow(img_tensor, cmap='gray')
    plt.show()

    std1 = par['std1']
    std2 = par['std2']
    filter_size = par['DoGfilter_size']
    DoG_filter = DoG(std1, std2, filter_size)

    mode = 'on-center'
    DoG_result_on_c = convolve(img_tensor, DoG_filter, mode)
    show_with_filter(DoG_result_on_c, DoG_filter, mode)

    mode = 'off-center'
    DoG_result_off_c = convolve(img_tensor, DoG_filter, mode)
    show_with_filter(DoG_result_off_c, DoG_filter, mode)

    min_val = par['min_val']
    max_val = par['max_val']
    time = par['time']
    dt = par['dt']
    
    coded_img = TimeToFirstSpikeEncoder(img_tensor, time, dt, min_val, max_val)
    '''print('time to first spike raster plot (original image):')
    raster_plot([coded_img], dt)'''
    print('time to first spike 3D raster plot (original image):')
    raster_plot3D(coded_img)

    coded_on_c = TimeToFirstSpikeEncoder(DoG_result_on_c, time, dt, min_val, max_val)
    '''print('time to first spike raster plot (on_center DoG-filtered):')
    raster_plot([coded_on_c], dt)'''
    print('time to first spike 3D raster plot (on_center DoG-filtered):')
    raster_plot3D(coded_on_c)

    coded_off_c = TimeToFirstSpikeEncoder(DoG_result_off_c, time, dt, min_val, max_val)
    '''print('time to first spike raster plot (off_center DoG-filtered):')
    raster_plot([coded_off_c], dt)'''
    print('time to first spike 3D raster plot (off_center DoG-filtered):')
    raster_plot3D(coded_off_c)

    max_spikes = par['max_spikes']
    
    coded_img = poisson_encoder(img_tensor, max_spikes, time, dt, max_val)
    '''print('poisson raster plot (original image):')
    raster_plot([coded_img], dt)'''
    print('poisson 3D raster plot (original image):')
    raster_plot3D(coded_img)

    coded_on_c = poisson_encoder(DoG_result_on_c, max_spikes, time, dt, max_val)
    '''print('poisson raster plot (on_center DoG-filtered):')
    raster_plot([coded_on_c], dt)'''
    print('poisson 3D raster plot (on_center DoG-filtered):')
    raster_plot3D(coded_on_c)

    coded_off_c = poisson_encoder(DoG_result_off_c, max_spikes, time, dt, max_val)
    '''print('poisson raster plot (off_center DoG-filtered):')
    raster_plot([coded_off_c], dt)'''
    print('poisson 3D raster plot (off_center DoG-filtered):')
    raster_plot3D(coded_off_c)

def present_gabor_filter(img_tensor, par):
    print('original image:')
    plt.imshow(img_tensor, cmap='gray')
    plt.show()
    
    min_val = par['min_val']
    max_val = par['max_val']
    time = par['time']
    dt = par['dt']
    l = par['l']
    theta = par['theta']
    sigma = par['sigma']
    gamma = par['gamma']
    size = par['gabor_filter_size']
    max_spikes = par['max_spikes']
    
    gabor_filter = gabor(l, theta, sigma, gamma, size)

    mode = 'on-center'
    gabor_result_on_c = convolve(img_tensor, gabor_filter, mode)
    show_with_filter(gabor_result_on_c, gabor_filter, mode)

    mode = 'off-center'
    gabor_result_off_c = convolve(img_tensor, gabor_filter, mode)
    show_with_filter(gabor_result_off_c, gabor_filter, mode)

    coded_img = TimeToFirstSpikeEncoder(img_tensor, time, dt, min_val, max_val)
    '''print('time to first spike raster plot (original image):')
    raster_plot([coded_img], dt)'''
    print('time to first spike 3D raster plot (original image):')
    raster_plot3D(coded_img)

    coded_on_c = TimeToFirstSpikeEncoder(gabor_result_on_c, time, dt, min_val, max_val)
    '''print('time to first spike raster plot (on_center gabor-filtered):')
    raster_plot([coded_on_c], dt)'''
    print('time to first spike 3D raster plot (on_center gabor-filtered):')
    raster_plot3D(coded_on_c)

    coded_off_c = TimeToFirstSpikeEncoder(gabor_result_off_c, time, dt, min_val, max_val)
    '''print('time to first spike raster plot (off_center gabor-filtered):')
    raster_plot([coded_off_c], dt)'''
    print('time to first spike 3D raster plot (off_center gabor-filtered):')
    raster_plot3D(coded_off_c)

    coded_img = poisson_encoder(img_tensor, max_spikes, time, dt, max_val)
    '''print('poisson raster plot (original image):')
    raster_plot([coded_img], dt)'''
    print('poisson 3D raster plot (original image):')
    raster_plot3D(coded_img)

    coded_on_c = poisson_encoder(gabor_result_on_c, max_spikes, time, dt, max_val)
    '''print('poisson raster plot (on_center gabor-filtered):')
    raster_plot([coded_on_c], dt)'''
    print('poisson 3D raster plot (on_center gabor-filtered):')
    raster_plot3D(coded_on_c)

    coded_off_c = poisson_encoder(gabor_result_off_c, max_spikes, time, dt, max_val)
    '''print('poisson raster plot (off_center gabor-filtered):')
    raster_plot([coded_off_c], dt)'''
    print('poisson 3D raster plot (off_center gabor-filtered):')
    raster_plot3D(coded_off_c)

