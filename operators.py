import torch
import os
import numpy as np
from abc import abstractmethod
from matplotlib import pyplot as plt
from utils import fft2c, ifft2c, complex_abs, embed_tensor_complex
from torch.autograd import Variable

class Operator(object):
    def __init__(self, size):
        self.size = size
       
    @staticmethod
    def complex_to_real(ndarray):
        r = np.expand_dims(np.real(ndarray), axis=-1)
        i = np.expand_dims(np.imag(ndarray), axis=-1)
        return np.concatenate([r,i], axis=-1)

    @staticmethod
    def real_to_complex(ndarray):
        return ndarray[...,0]+1j*ndarray[...,1]

    # Expect numpy array, return numpy array
    @abstractmethod
    def forward(self, x, mask=True):
        pass

    @abstractmethod
    def adjoint(self, y, mask=True):
        pass

    @abstractmethod
    def inverse(self, y, mask=True):
        pass

    @abstractmethod
    def add_noise(self, y, noise_level):
        pass

    # Convenience batch wrapping for basic np methods.
    def _forward_batch(self, x, mask=True):
        res = np.zeros(shape=x.shape)
        for k in range(x.shape[0]):
            res[k,0,...] = self.forward(x[k,0,...], mask=mask)
        return res

    def _adjoint_batch(self, y, mask=True):
        res = np.zeros(shape=y.shape)
        for k in range(y.shape[0]):
            res[k, 0, ...] = self.adjoint(y[k, 0, ...], mask=mask)
        return res

    def _inverse_batch(self, y, mask=True):
        res = np.zeros(shape=y.shape)
        for k in range(y.shape[0]):
            res[k, 0, ...] = self.inverse(y[k, 0, ...], mask=mask)
        return res

    # Default methods to wrap python implementations in torch by looping. Can be overwritten with more efficient
    # implementation in subclass.
    def forward_torch(self, x, mask=True):
        x = x.cpu().numpy()
        if len(x.shape) == 5:
            x = self.real_to_complex(x)
        res = self._forward_batch(x, mask=mask)
        if np.iscomplexobj(res):
            res = self.complex_to_real(res)
        return torch.Tensor(res).cuda()
    
    def adjoint_torch(self, y, mask=True):
        y = y.cpu().numpy()
        if len(y.shape) == 5:
            y = self.real_to_complex(y)
        res = self._adjoint_batch(y, mask=mask)
        if np.iscomplexobj(res):
            res = self.complex_to_real(res)
        return torch.Tensor(res).cuda()
       
    def inverse_torch(self, y, mask=True):
        y = y.cpu().numpy()
        if len(y.shape) == 5:
            y = self.real_to_complex(y)
        res = self._inverse_batch(y, mask=mask)
        if np.iscomplexobj(res):
            res = self.complex_to_real(res)
        return torch.Tensor(res).cuda()
    


class MRI(Operator):

    def __init__(self, size=256, subsampling=2.1e3, n_directions=None, direcotry='/store/CCIMI/sl767/MRI_masks/'):
        '''
        :param size: The size of image and measurement space
        :param subsampling: The decay parameter for the Gaussian kernel in the sampling pattern.
        A higher value corresponds to more measurements being taken.
        :param n_directions: Setting this value changes the sampling pattern to a radial pattern with n_directions
        sampling lines emerging from the origin. Ignores subsampling.
        :param direcotry: Directory to read and write generated sampling patterns from and to.
        '''
        Operator.__init__(self, size)
        self.subsampling = subsampling
        self.n_directions=n_directions
        self.directory = direcotry
        self.mask = self._get_mask()
        self.mask_torch = torch.Tensor(self.mask).unsqueeze(-1).cuda()
        self.sampling_level = self.mask.mean()

    def forward(self, x, mask=True):
        y =  np.fft.fftshift(np.fft.fft2(x, norm="ortho"))
        if mask:
            y = y * self.mask
        return y

    def adjoint(self, y, mask=True):
        if mask:
            y = y * self.mask
        x = np.real(np.fft.ifft2(np.fft.ifftshift(y), norm="ortho"))
        return x

    def inverse(self, y, mask=True):
        return self.adjoint(y=y, mask=mask)

    def add_noise(self, y, noise_level=3e-3):
        '''
        :param y: Measurements
        :param noise_level: Noise Level
        :return: Measurements corrupted with real valued Gaussian noise
        '''
        assert len(y.shape) == 5
        noise = (Variable(y.new(y.shape[:-1]).normal_(0, 1))).unsqueeze(-1)
        zeros = (y.new_zeros(y.shape[:-1])).unsqueeze(-1)
        n = torch.cat([noise, zeros], axis=-1)
        max_val = complex_abs(y).max()
        return y + noise_level*max_val*n

    def forward_torch(self, x, mask=True):
        '''
        :param x: The image as rank 4 tensor (batch, ch, x, y)
        :param mask: If mask should be applied in Fourier space
        :return: The measurements as rank 5 tensor (batch, ch, t_x, t_y, real/imag)
        '''
        x = embed_tensor_complex(x)
        y = fft2c(x)
        if mask:
            y = y * self.mask_torch
        return y

    def adjoint_torch(self, y, mask=True):
        '''
        :param y: The measurements as rank 5 tensor (batch, ch, t_x, t_y, real/imag)
        :param mask: If mask should be applied in Fourier space
        :return: The image as rank 4 tensor (batch, ch, x, y)
        '''
        if mask:
            y = y * self.mask_torch
        x = ifft2c(y)
        return x[...,0]

    def inverse_torch(self, y, mask=True):
        '''
        :param y: The measurements as rank 5 tensor (batch, ch, t_x, t_y, real/imag)
        :param mask: If mask should be applied in Fourier space
        :return: The image as rank 4 tensor (batch, ch, x, y)
        '''
        return self.adjoint_torch(y=y, mask=mask)

    def __repr__(self):
        plt.figure(figsize=(4,4))
        plt.imshow(self.mask)
        plt.axis('off')
        plt.show()
        return f'MRI Operator\nResolution {self.size}\nSubsampling level: {self.sampling_level}'

    def __str__(self):
        return f'MRI Operator\nResolution {self.size}\nSubsampling level: {self.sampling_level}'

    def _get_mask(self):
        if self.n_directions is None:
            path = self.directory + f'Size_{self.size}_Subsamping_{int(self.subsampling)}.npy'
            if os.path.exists(path):
                print('Sampling pattern loaded')
                return np.load(path)
            else:
                x, y = np.meshgrid(range(-self.size//2, self.size//2), range(-self.size//2, self.size//2))
                d = x ** 2 + y ** 2
                exp = np.exp(-d / self.subsampling)
                mask = (np.random.uniform(size=(self.size, self.size)) < exp).astype(int)
                with open(path, 'wb') as f:
                    np.save(f, mask)
                print('New sampling pattern created')
                return mask
        else:
            path = self.directory + f'Size_{self.size}_N_Directions_{int(self.n_directions)}.npy'
            if os.path.exists(path):
                print('Sampling pattern loaded')
                return np.load(path)
            else:
                x, y = np.meshgrid(range(-self.size // 2, self.size // 2), range(-self.size // 2, self.size // 2))
                angles = np.linspace(0, np.pi, num=self.n_directions, endpoint=False)
                directions = [(np.sin(x), np.cos(x)) for x in angles]
                mask = np.zeros(shape=(self.size, self.size))
                for d in directions:
                    voxels = (np.abs(d[0] * x + d[1] * y)) < .5
                    mask += voxels
                mask[mask > 1] = 1
                with open(path, 'wb') as f:
                    np.save(f, mask)
                print('New sampling pattern created')
                return mask
