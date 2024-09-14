import os
import sys

from tqdm import trange
from loguru import logger
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)


class DINEOF(BaseEstimator):
    def __init__(self, R, tensor_shape, mask=None,
                 nitemax=300, toliter=1e-3, tol=1e-8, to_center=True, 
                 keep_non_negative_only=True,
                 with_energy=False,
                 early_stopping=True):
        self.K = R  # HACK: Make interface consistent with DINEOF3, but want to keep intrinsics as is
        self.nitemax = nitemax
        self.toliter = toliter
        self.tol = tol
        self.to_center = to_center
        self.keep_non_negative_only = keep_non_negative_only
        self.tensor_shape = tensor_shape
        self.with_energy = with_energy
        self.mask = np.load(mask).astype(bool) if mask is not None else np.ones(tensor_shape).astype(bool)
        self.mask = self._broadcast_mask(self.mask, tensor_shape[-1])
        self.inverse_mask = ~self.mask
        self.early_stopping = early_stopping

    def _broadcast_mask(self, mask, t):
        mask = np.repeat(mask[:, :, None], t, axis=2)
        return rectify_tensor(mask)
        
    def score(self, X, y):
        """
            You can think of this like negative error (bigger is better due to error diminishing.
            It is made like so to be compatible with scikit-learn grid search utilities.
        """
        y_hat = self.predict(X)
        return -nrmse(y_hat, y)
    
    def rmse(self, X, y):
        return -self.score(X, y) * y.std()
    
    def nrmse(self, X, y):
        return -self.score(X, y)
        
    def predict(self, X):
        output = np.array([self.reconstructed_tensor[x[0], x[1], x[2]] for x in X])
        return output
        
    def fit(self, X, y):
        tensor = tensorify(X, y, self.tensor_shape)
        self._fit(rectify_tensor(tensor))
        
    def _fit(self, mat):
        if mat.ndim > 2:
            mat = rectify_tensor(mat)

        if self.to_center:
            mat, *means = center_mat(mat)

        # Initial guess
        nan_mask = np.isnan(mat)
        non_nan_mask = ~nan_mask
        mat[nan_mask] = 0
        # Outside of an investigated area everything is considered to be zero

        pbar = trange(self.nitemax, desc='Reconstruction')
        conv_error = 0
        energy_per_iter = []
        for i in pbar:
            u, s, vt = svds(mat, k=self.K, tol=self.tol)

            # Save energy characteristics for this iteration
            if self.with_energy:
                energy_i = calculate_mat_energy(mat, s)
                energy_per_iter.append(energy_i)
            
            mat_hat = u @ np.diag(s) @ vt
            mat_hat[non_nan_mask] = mat[non_nan_mask]

            new_conv_error = np.sqrt(np.mean(np.power(mat_hat[nan_mask] - mat[nan_mask], 2))) / mat[non_nan_mask].std()
            mat = mat_hat

            pbar.set_postfix(error=new_conv_error, rel_error=abs(new_conv_error - conv_error))
            
            grad_conv_error = abs(new_conv_error - conv_error)
            conv_error = new_conv_error
            
            logger.info(f'Error/Relative Error at iteraion {i}: {conv_error}, {grad_conv_error}')
            
            if self.early_stopping:
                break_condition = (conv_error <= self.toliter) or (grad_conv_error < self.toliter)
            else:
                break_condition = (conv_error <= self.toliter)
                
            if break_condition:              
                break

        energy_per_iter = np.array(energy_per_iter)

        if self.to_center:
            mat = decenter_mat(mat, *means)

        if self.keep_non_negative_only:
            mat[mat < 0] = 0

        # Save energies in model for distinct components (lat, lon, t)
        if self.with_energy:
            for i in range(mat.ndim):
                setattr(self, f'total_energy_{i}', np.array(energy_per_iter[:, i, 0]))
                setattr(self, f'explained_energy_{i}', np.array(energy_per_iter[:, i, 1]))
                setattr(self, f'explained_energy_ratio_{i}', np.array(energy_per_iter[:, i, 2]))

        self.final_iter = i
        self.conv_error = conv_error
        self.grad_conv_error = grad_conv_error
        self.reconstructed_tensor = unrectify_mat(mat, spatial_shape=self.tensor_shape[:-1])
        self.singular_values_ = s
        self.ucomponents_ = u
        self.vtcomponents_ = vt

def unrectify_mat(mat, spatial_shape):
    tensor = []

    for t in range(mat.shape[-1]):
        col = mat[:, t]
        unrectified_col = col.reshape(spatial_shape)
        tensor.append(unrectified_col)

    tensor = np.array(tensor)
    tensor = np.moveaxis(tensor, 0, -1)

    return tensor


def rectify_tensor(tensor):
    rect_mat = []
    for t in range(tensor.shape[-1]):
        rect_mat.append(tensor[:, :, t].flatten())
    rect_mat = np.array(rect_mat)
    rect_mat = np.moveaxis(rect_mat, 0, -1)
    return rect_mat

def tensorify(X, y, shape):
    tensor = np.full(shape, np.nan)
    for i, d in enumerate(X):
        lat, lon, t = d.astype(np.int32)
        tensor[lat, lon, t] = y[i]

    return tensor

def nrmse(y_hat, y):
    """
        Normalized root mean squared error
    """
    root_meaned_sqd_diff = np.sqrt(np.mean(np.power(y_hat - y, 2)))
    return root_meaned_sqd_diff / np.std(y)

def calculate_mat_energy(mat, s):
    sample_count_0 = mat.shape[1]
    sample_coef_0 = 1 / (sample_count_0 - 1)
    total_energy_0 = np.array([sample_coef_0 * np.trace(mat @ mat.T) for _ in range(len(s))])
    expl_energy_0 = -np.sort(-sample_coef_0 * s * s)
    expl_energy_ratio_0 = expl_energy_0 / total_energy_0

    sample_count_1 = mat.shape[0]
    sample_coef_1 = 1 / (sample_count_1 - 1)
    total_energy_1 = np.array([sample_coef_1 * np.trace(mat.T @ mat) for _ in range(len(s))])
    expl_energy_1 = -np.sort(-sample_coef_1 * s * s)
    expl_energy_ratio_1 = expl_energy_1 / total_energy_1

    return np.array([[total_energy_0, expl_energy_0, expl_energy_ratio_0],
                        [total_energy_1, expl_energy_1, expl_energy_ratio_1]])
                    
def center_mat(mat):
    nan_mask = np.isnan(mat)
    temp_mat = mat.copy()
    temp_mat[nan_mask] = 0

    m0 = temp_mat.mean(axis=0)
    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] -= m0

    m1 = temp_mat.mean(axis=1)
    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] -= m1

    temp_mat[nan_mask] = np.nan
    return temp_mat, m0, m1


def decenter_mat(mat, m0, m1):
    temp_mat = mat.copy()

    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] += m0

    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] += m1

    return temp_mat