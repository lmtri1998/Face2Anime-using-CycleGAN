import matplotlib.pyplot as plt
import torch
import argparse

from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from cyclegan_code import Generator_Res


"""Do style transfer on multiple images"""
def style_transfer_make_grid(Gen_A, Gen_B, LoaderA, LoaderB, which_cycle, indices=[], n_img = 3, plot=False, savename='test_mult.png'):  
    
    if n_img == 0:
        n_img = 5
    
    if not indices:
        indices = []
    
    grid = torch.Tensor()
    while (len(indices) < n_img):
        r_ind = np.random.randint(0, len(LoaderA.dataset))
        indices.append(r_ind)
    print(indices)
    
    if which_cycle == "A":
    
        for index in indices:
            loadedA = LoaderA.dataset[index][0]
            A_in = loadedA.unsqueeze(0).cpu()
            fake_B = Gen_A(loadedA.unsqueeze(0)).cpu()
            A_recon = Gen_B(loadedA.unsqueeze(0)).cpu()

            new_grid = torch.cat((A_in,fake_B,A_recon),dim=0)
            grid = torch.cat((grid,new_grid),dim=0)
    
    if which_cycle == "B":

        for index in indices:
            loadedB = LoaderB.dataset[index][0]
            B_in = loadedB.unsqueeze(0).cpu()
            fake_A = Gen_B(loadedB.unsqueeze(0)).cpu()
            B_recon = Gen_A(loadedB.unsqueeze(0)).cpu()

            new_grid = torch.cat((B_in,fake_A,B_recon),dim=0)
            grid = torch.cat((grid,new_grid),dim=0)
            
    show_grid = torchvision.utils.make_grid(grid, nrow=3, normalize=True).detach().numpy()
    plt.figure(figsize = (15,15))
    plt.axis('off')
    # plt.imshow(np.transpose(show_grid, (1,2,0)), interpolation='nearest')
    plt.imsave(savename, np.transpose(show_grid, (1,2,0)))
    

"""Do style transfer on one image"""
def style_transfer_one_img(Gen_A, Gen_B, loadedA, savename='test.png'):  
    
    grid = torch.Tensor()
    
    A_in = loadedA.cpu() # Assume the input is already unsqueezed
    fake_B = Gen_A(loadedA).cpu()
    A_recon = Gen_B(loadedA).cpu()

    new_grid = torch.cat((A_in,fake_B,A_recon),dim=0)
    grid = torch.cat((grid,new_grid),dim=0)
    
    show_grid = torchvision.utils.make_grid(grid, nrow=3, normalize=True).detach().numpy()
    plt.figure(figsize = (15,15))
    plt.axis('off')
    plt.imsave(savename, np.transpose(show_grid, (1,2,0)))

"""Do style transfer on one image without making a grid"""
def style_transfer_only_fake(Gen_A, Gen_B, loadedA, savename='test.png'):
    A_in = loadedA.cpu() # Assume the input is already unsqueezed
    fake_B = Gen_A(loadedA).cpu()
    save_image(fake_B, savename, normalize=True)


"""Plot the training curve and the validation curve"""
def Visualize_plot(checkpoint, savename='losses_plot.png'):  

    plt.figure(figsize=(20,20))

    plt.subplot(3,1,1)
    plt.plot([item['a_dis_loss'] for item in checkpoint['epoch_loss']], label='a_dis_loss')
    plt.plot([item['b_dis_loss'] for item in checkpoint['epoch_loss']], label='b_dis_loss')
    plt.legend(loc='upper right')

    plt.subplot(3,1,2)
    plt.plot([item['a_gen_loss'] for item in checkpoint['epoch_loss']], label='A Gen')
    plt.plot([item['b_gen_loss'] for item in checkpoint['epoch_loss']], label='B Gen')
    plt.plot([item['a_cycle_loss'] for item in checkpoint['epoch_loss']], label='A Cycle')
    plt.plot([item['b_cycle_loss'] for item in checkpoint['epoch_loss']], label='B Cycle')
    plt.plot([item['a_idt_losses'] for item in checkpoint['epoch_loss']], label='A Idt')
    plt.plot([item['b_idt_losses'] for item in checkpoint['epoch_loss']], label='B Idt')
    plt.legend(loc='upper right')

    plt.subplot(3,1,3)
    plt.plot([item['a_dis_real_loss'] for item in checkpoint['epoch_loss']], label='A Real')
    plt.plot([item['a_dis_fake_loss'] for item in checkpoint['epoch_loss']], label='A Fake')
    plt.plot([item['b_dis_real_loss'] for item in checkpoint['epoch_loss']], label='B Real')
    plt.plot([item['b_dis_fake_loss'] for item in checkpoint['epoch_loss']], label='B Fake')
    plt.legend(loc='upper right')

    plt.savefig(savename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test and Visualize')
    parser.add_argument('--cp', default='./cp/face2anime_netG_95.pt',
                        help='path to the checkpoint')
    parser.add_argument('--device', default='cpu',
                        help='use gpu or cpu')
    parser.add_argument('--image', default='test_images/Emma_Watson_2013_edited.jpg',
                        help='Path to the face image')
    parser.add_argument('--grid', default=False, action="store_true",
                        help='Produce a grid of real, fake, and converted fake images')
    parser.add_argument('--outname', default='emma.png',
                        help='The name of the output image')
    args = parser.parse_args()

    """# Load and Test"""

    # Change the below in accordance with your local directories
    cp_dir = "./cp"

    device = 'cpu'
    # load the checkpoint
    checkpoint = torch.load(args.cp, map_location=torch.device(args.device))

    # separate the Generators 
    Gen_A1 = Generator_Res()
    Gen_A1.load_state_dict(checkpoint['Gen_A_state_dict'])
    Gen_B1 = Generator_Res()
    Gen_B1.load_state_dict(checkpoint['Gen_B_state_dict'])

    # A Cycle
    # good_set = [63,43,2,89,11,71,68]
    # bad_set = [75,96,73,46,64]

    # B Cycle
    # good_set = [31,20,56,61]
    # bad_set = [99,44,15,11]

    # Change this to any set above if you want to see fixed results
    # Change this to [] if you want to see random results
    # preset_index = bad_set

    # Change the fifth parameter to "B" if you want Cycle B result
    # "A" for Cycle A result
    # style_transfer_make_grid(Gen_A1.eval(), Gen_B1.eval(), LoadertestA, LoadertestB, "A", preset_index, n_img=len(preset_index))


    # Put the test image in your local directory
    test_img = Image.open(args.image).convert('RGB')
    newsize = (128, 128)
    test_img = test_img.resize(newsize)
    
    test_img = TF.resize(test_img,128)
    
    test_img = TF.to_tensor(test_img)
    normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    test_img = normalizer(test_img).unsqueeze_(0)

    if args.grid:
        style_transfer_one_img(Gen_A1.eval(), Gen_B1.eval(),test_img, args.outname)
    else:
        style_transfer_only_fake(Gen_A1.eval(), Gen_B1.eval(),test_img, args.outname)

    # Visualize_plot(checkpoint)