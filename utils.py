from imports import *
from const import *

def select_file(experiment):
    # Get the current directory
    current_dir = os.getcwd()

    # Path to open the file dialog window in the data branch
    path_folder_data = os.path.join(current_dir, f"{experiment}_Data")

    # Select H5 dataset file
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(initialdir=path_folder_data)

    return filename

def triangleSmooth(data,smoothing_triangle):
    factor = smoothing_triangle-3
    f = np.zeros((1+2*factor))
    for i in range(factor):
        f[i] = i+1
        f[-i-1] = i+1
    f[factor] = factor + 1
    triangle_filter = f / np.sum(f)

    padded = np.pad(data, factor, mode='edge')
    smoothed_data = np.convolve(padded, triangle_filter, mode='valid')
    return smoothed_data

def fullPSTH(point_array,binsize,smoothing_triangle,sr,start_offset):
    sr_factor = (1000/sr)
    num_trials = point_array.shape[0]
    xtimes = np.where(point_array)[1]*sr_factor
    
    nsamples = point_array.shape[1]
    ms_length = nsamples*sr_factor
    last_bin = binsize*np.ceil(ms_length/binsize)

    edges = np.arange(0,last_bin+binsize,binsize)
    bin_centers = edges[0:-1]+binsize/2+start_offset

    counts = np.histogram(xtimes,bins=edges)
    norm_count = [counts[0]/(num_trials*binsize/1000)]

    if smoothing_triangle==1:
        PSTH = norm_count[0]
    else:
        PSTH = triangleSmooth(norm_count[0],smoothing_triangle)
    return PSTH,bin_centers

def create_spike_train(spike_data, len_pre_fixed_point, len_post_fixed_point):
    spike_data = np.array(spike_data.squeeze(), dtype=int)

    spike_train_post_fixed_point = np.zeros(len_post_fixed_point)
    post_fixed_point = spike_data[(spike_data >= 0) & (spike_data < len_post_fixed_point)]
    if post_fixed_point.size > 0:
        spike_train_post_fixed_point[post_fixed_point] = 1

    spike_train_pre_fixed_point = np.zeros(len_pre_fixed_point)
    pre_fixed_point = -1*spike_data[(spike_data < 0) & (spike_data > -len_pre_fixed_point)]
    if pre_fixed_point.size > 0:
        spike_train_pre_fixed_point[pre_fixed_point] = 1
    spike_train = np.concatenate((spike_train_pre_fixed_point[::-1], spike_train_post_fixed_point))

    spike_train = np.reshape(spike_train, (1, spike_train.size))

    return spike_train

def plot_psths(ax, spikes, markers, marker_labels, title, len_pre_fixed_point, len_post_fixed_point, max_trials=1000):
    bin_size = 1
    trial = 1
    psths = np.array([])
    for spike_data in spikes:

        if trial > max_trials:
            break

        spike_train = create_spike_train(spike_data, len_pre_fixed_point, len_post_fixed_point)
        psth, _ = fullPSTH(spike_train, bin_size, smoothing_triangle=50, sr=1000, start_offset=0)

        if psths.size > 0:
            psths = np.vstack((psths, psth))
        else:
            psths = psth
        trial = trial + 1

    psths = zscore(psths, axis=1)

    ax.pcolormesh(np.linspace(-len_pre_fixed_point, len_post_fixed_point-1, len_pre_fixed_point+len_post_fixed_point), np.arange(psths.shape[0]), psths, cmap='magma', vmin=-3, vmax=3)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial #')

def plot_raster(ax, spikes, markers, marker_labels, title, max_trials=1000):
    trial = 1
    first_iteration = True
    for spike_data, marker_data in zip(spikes, markers):

        if trial > max_trials:
            break

        ax.scatter(spike_data, np.ones_like(spike_data) * trial, marker='|', color='k')
        for mk in range(marker_data.size):
            ax.scatter(marker_data[mk], np.ones_like(marker_data[mk]) * trial, s=30, color=clmap[mk], marker='o', edgecolors=clmap[mk], label=marker_labels[mk] if first_iteration else None)
        trial = trial + 1
        first_iteration = False

    ax.set_ylim([0.5, trial ])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial #')