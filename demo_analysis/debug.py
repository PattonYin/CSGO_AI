from config import data_map_X_1
from plot_module import plot_move_seq


def debug_seq_data(X_seq, Y_seq, traj_index):
    """
        After self.prepare_seq, ensure that X_seq and Y_seq are correct by manually plotting the sequence information on the map using a annimation
        
        Args:
        X_seq (np.array): sequences of predictor information shape(num of seqs, tick_per_seq, X_dim)
        Y_seq (np.array): sequences of winner boolean (num of seqs, 1)
        traj_index (int): index of the sequence to be visualized
    
    """    
    seq_to_visualize = X_seq[traj_index]
    plot_move_seq(seq_to_visualize, self.data_map)
    
    
    
    
    return 