import torch
from torch import nn
from mfa import MFA
from pathlib import Path
import numpy as np

class MFAWrapper(nn.Module):
    """Wrapper for dmfa_inpainting. It's not meant to be trained, only for inference."""

    def __init__(self, mfa_model: MFA):
        super().__init__()
        self.mfa = mfa_model

    
    def forward(self, X: torch.Tensor, J: torch.Tensor):
        """X, J -> P, M, A, D"""

        X_masked = X * (J==1) # 1 = mc.KNOWN

        P = torch.ones(len(X), 1).to(X.device)
        M = []
        A = []
        D = []

        for (x, j, x_masked) in zip(X, J, X_masked):

            used_features = torch.nonzero(j.flatten()).flatten()
            
            x_masked = x_masked.reshape(1, -1)
            x = x.reshape(1, -1)

            _, _, _, _, _, log_likelihood, (m_full, a_full, d_full) = self.mfa.conditional_reconstruct(
                    x_masked,
                    observed_features=used_features, 
                    original_full_samples = x
                )
            
            M.append(m_full)
            A.append(a_full)
            D.append(d_full)

        M, A, D = [torch.stack(t) for t in [M, A, D]]

        return P, M, A, D

    
    @classmethod
    def from_path(cls, dump_path: Path):
        # mnist hyperparams
        image_shape = [28, 28]  # The input image shape
        n_components = 50  # Number of components in the mixture model
        n_factors = 6  # Number of factors - the latent dimension (same for all components)
        batch_size = 1000  # The EM batch size
        num_iterations = 30  # Number of EM iterations (=epochs)
        feature_sampling = False  # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0


        model = MFA(
            n_components=n_components, 
            n_features=np.prod(image_shape), 
            n_factors=n_factors
        )

        model.load_state_dict(
            torch.load(
                dump_path / f"model_c_{n_components}_l_{n_factors}_init_kmeans.pth",
                map_location="cpu"
                )
            )
        
        return cls(model)