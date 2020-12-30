import os
from torchvision.datasets import CelebA, MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from mfa import MFA
from utils import *
from imageio import imwrite
from tqdm import tqdm
from datasets import mnist_train_val_datasets, celeba_train_val_datasets, RandomRectangleMaskConfig, UNKNOWN_LOSS
from sys import argv
from torchvision.datasets import MNIST, CelebA, FashionMNIST

"""
Examples for inference using the trained MFA model - likelihood evaluation and (conditional) reconstruction
"""




if __name__ == "__main__":
    print(argv)
    dataset = dataset = argv[1] if len(argv) >= 2 else 'mnist'
    find_outliers = False
    reconstruction = True
    inpainting = True
    mask_h, mask_w = [int(a)  for a in argv[2:4]] if len(argv) >= 4 else (None, None)

    print('Preparing dataset and parameters for', dataset, '...')
    if dataset == 'celeba':
        image_shape = [32, 32, 3]       # The input image shape
        n_components = 300              # Number of components in the mixture model
        n_factors = 10                  # Number of factors - the latent dimension (same for all components)
        batch_size = 128                # The EM batch size
        num_iterations = 10             # Number of EM iterations (=epochs)
        feature_sampling = 0.2          # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        # trans = transforms.Compose([CropTransform((25, 50, 25+128, 50+128)), transforms.Resize(image_shape[0]),
        #                             transforms.ToTensor(),  ReshapeTransform([-1])])
        # test_dataset = CelebA(root='./data', split='test', transform=trans, download=True)
        # The train set has more interesting outliers...
        # test_dataset = CelebA(root='./data', split='train', transform=trans, download=True)

        img_to_crop = 1.875
        img_size = image_shape[0]
        full_img_size = int(img_size * img_to_crop)
        mask_h = mask_h or img_size // 2
        mask_w = mask_w or img_size // 2
        train_dataset, test_dataset = celeba_train_val_datasets(
            with_mask=True,
            mask_configs=[
                RandomRectangleMaskConfig(UNKNOWN_LOSS, mask_h, mask_w)
            ],
            resize_size=(full_img_size, full_img_size),
            crop_size=(img_size, img_size),

        )
    elif 'mnist' in dataset:
        image_shape = [28, 28]  # The input image shape
        n_components = 50  # Number of components in the mixture model
        n_factors = 6  # Number of factors - the latent dimension (same for all components)
        batch_size = 1000  # The EM batch size
        num_iterations = 30  # Number of EM iterations (=epochs)
        feature_sampling = False  # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0  # Perform additional training with diagonal (per-pixel) covariance, using SGD
        # init_method = 'kmeans'  # Initialize by using k-means clustering
        # trans = transforms.Compose([transforms.ToTensor(), ReshapeTransform([-1])])
        # # train_set = MNIST(root='./data', train=True, transform=trans, download=True)
        # test_dataset = MNIST(root='./data', train=False, transform=trans, download=True)
        img_size = image_shape[0]
        mask_h = mask_h or img_size // 2
        mask_w = mask_w or img_size // 2        
        train_dataset, test_dataset = mnist_train_val_datasets(
            ds_type=FashionMNIST if dataset == "fashion_mnist" else MNIST,
            with_mask=True,
             mask_configs=[
                RandomRectangleMaskConfig(UNKNOWN_LOSS, mask_h, mask_w)
            ],
            resize_size=(img_size, img_size),
        )

    else:
        assert False, 'Unknown dataset: ' + dataset

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device("cpu")
    print(device)
    model_dir = './models/' + f"{dataset}_{image_shape[0]}_{image_shape[1]}"
    figures_dir = './figures/'+ f"{dataset}_{image_shape[0]}_{image_shape[1]}"
    os.makedirs(figures_dir, exist_ok=True)

    print('Loading pre-trained MFA model...')
    model = MFA(n_components=n_components, n_features=np.prod(image_shape), n_factors=n_factors).to(device=device)
    if "mnist" in dataset:
        model.load_state_dict(
            torch.load(
                os.path.join(model_dir, 'model_c_{}_l_{}_init_kmeans.pth'.format(n_components, n_factors)),
                map_location=device
                )
            )
    else:
        model.load_state_dict(
            torch.load(
                os.path.join(model_dir, 'model_c_{}_l_{}_init_rnd_samples.pth'.format(n_components, n_factors)),
                map_location=device

                )
            )

    if find_outliers:
        print('Finding dataset outliers...')
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        all_ll = []
        for batch_x, _ in tqdm(loader):
            all_ll.append(model.log_prob(batch_x.to(device)))
        all_ll = torch.cat(all_ll, dim=0)
        ll_sorted = torch.argsort(all_ll).cpu().numpy()

        all_keys = [key for key in SequentialSampler(test_dataset)]
        outlier_samples, _ = zip(*[test_dataset[all_keys[ll_sorted[i]]] for i in range(100)])
        mosaic = samples_to_mosaic(torch.stack(outlier_samples), image_shape=image_shape)
        imwrite(os.path.join(figures_dir, 'outliers.jpg'), mosaic)

    if reconstruction:
        print('Reconstructing images from the trained model...')
        random_samples_with_masks, _ = zip(*[test_dataset[k] for k in RandomSampler(test_dataset, replacement=True, num_samples=25)])

        random_samples = [
            rs[0] for rs in random_samples_with_masks
        ]
        masks = [
            rs[1] for rs in random_samples_with_masks
        ]

        random_samples = torch.stack(random_samples).to(device)
        masks = torch.stack(masks).to(device)
        # print(random_samples.shape, mask.shape)

        if inpainting:
            # if dataset == "mnist":
            # Hide part of each image

            original_full_samples = random_samples.clone()
            random_samples *= masks

            reconstructed_samples = []
            for samp, msk, orig, in tqdm(zip(random_samples, masks, original_full_samples)):
                used_features = torch.nonzero(msk.flatten()).flatten()


                rec_samp, means_samples, A_samples, D_samples, reconstructed_A, log_likelihood, _ = model.conditional_reconstruct(
                    samp.unsqueeze(0),
                    observed_features=used_features, 
                    original_full_samples = orig.unsqueeze(0)
                )#.cpu()



                reconstructed_samples.append(rec_samp)
            reconstructed_samples = torch.stack(reconstructed_samples).squeeze()


        else:
            reconstructed_samples = model.reconstruct(random_samples.to(device)).cpu()

        if inpainting:
            reconstructed_samples = random_samples * masks + reconstructed_samples * (1 - masks)


        mosaic_original = samples_to_mosaic(random_samples, image_shape=image_shape)
        imwrite(os.path.join(figures_dir, 'original_samples.jpg'), mosaic_original)
        mosaic_recontructed = samples_to_mosaic(reconstructed_samples, image_shape=image_shape)
        imwrite(os.path.join(figures_dir, 'reconstructed_samples.jpg'), mosaic_recontructed)

        for d in range(6):
            mosaic_recontructed = samples_to_mosaic(reconstructed_A[:,:,d], image_shape=image_shape)
            imwrite(os.path.join(figures_dir, 'reconstructed_A' + str(d) + '.jpg'), mosaic_recontructed)


if len(image_shape) == 2:
    image_shape.append(1)

to_dump = []
for ((x, j), y) in tqdm(test_dataset):
    x, j = [t.to(device) for t in [x, j]]
    used_features = torch.nonzero(j.flatten()).flatten()

    x_masked = x * j

    
    _, _, _, _, _, log_likelihood, (m_full, a_full, d_full) = model.conditional_reconstruct(
        x_masked.unsqueeze(0),
        observed_features=used_features, 
        original_full_samples = x.unsqueeze(0)
    )

    x_resh = x.reshape(list(reversed(image_shape)))
    j_resh = j.reshape(list(reversed(image_shape)))

    print("sh", [t.shape for t in [x_masked, x, j_resh, m_full, a_full, d_full]])
    to_dump.append(
        (
            x_resh.detach().cpu().numpy(), 
            j_resh.detach().cpu().numpy(), 
            np.array([1]), 
            m_full.detach().cpu().numpy(), 
            a_full.detach().cpu().numpy(), 
            d_full.detach().cpu().numpy(), 
            (
                np.array([
                    y if "mnist" in dataset else 0
                ]), 
                log_likelihood.sum().item()
            )
        )
    )

from pathlib import Path
import pickle

with (Path(model_dir) / f"val_predictions_{mask_h}x{mask_w}.pkl").open("wb") as f:
    pickle.dump(to_dump, f)
