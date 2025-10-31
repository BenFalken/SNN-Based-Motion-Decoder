from imports import *
from const import *

def select_file(experiment):
    '''Prompts user to select the exact file they want'''
    current_dir = os.getcwd()
    # Path to open the file dialog window in the data branch
    path_folder_data = os.path.join(current_dir, f"data/{experiment}")
    # Select H5 dataset file
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(initialdir=path_folder_data)
    return filename

def process_velocities(y):
    '''Return trial-averaged velocity traces'''
    T = np.arange(y.shape[0])
    # Compute mean velocity across trial (centered at cue time)
    mean_x_vel = np.mean(y[:, :, 0], axis=1)
    mean_y_vel = np.mean(y[:, :, 1], axis=1)
    # Compute velocity errors across trial (centered at cue time)
    x_vel_err = 2*np.std(y[:, :, 0], axis=1)/np.sqrt(y.shape[1])
    y_vel_err = 2*np.std(y[:, :, 1], axis=1)/np.sqrt(y.shape[1])
    return T, mean_x_vel, mean_y_vel, x_vel_err, y_vel_err