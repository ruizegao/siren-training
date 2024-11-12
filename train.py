import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import dataio
from torch import nn
from siren_pytorch import SirenNet, Siren
from loss_functions import sdf
from accelerate import Accelerator
import argparse

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor_0 = torch.zeros(1).to('cuda')
p = argparse.ArgumentParser()

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--data_path', type=str, default='obj/thai_stat.xyz')
p.add_argument('--model_path', default=None, help='Checkpoint to trained model.')
args = p.parse_args()

sdf_dataset = dataio.PointCloud(args.data_path, on_surface_points=args.batch_size)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=4)

accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=len(dataloader))
device = accelerator.device

model = SirenNet(
    dim_in = 3,                        # input dimension, ex. 2d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 1,                       # output dimension, ex. rgb value
    num_layers = 5,                    # number of layers
    final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
).to(device)

optimizer = Adam(model.parameters(), lr=args.lr)


model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

loss_fn = sdf
best_loss = float('inf')
print("Training starts.")
for epoch_index in range(args.num_epochs):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    batch_index = 0
    for data in dataloader:
        with accelerator.accumulate(model):
            # Every data instance is an input + label pair
            optimizer.zero_grad()
            inputs, labels = data
            coords_org = inputs['coords'].clone().detach().requires_grad_(True).to(device)
            coords = coords_org

            gt_sdf = labels['sdf'].to(device)
            gt_normals = labels['normals'].to(device)
            # Zero your gradients for every batch!

            # Make predictions for this batch
            outputs = model(coords)
            # print(outputs.device)
            # Compute the loss and its gradients
            loss = loss_fn(coords_org, outputs, gt_sdf, gt_normals)
            accelerator.backward(loss)

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            batch_index += 1
        # if i % 3 == 2:
        #     last_loss = running_loss / 3  # loss per batch
        #     print('Epoch {}  batch {} loss: {}\n'.format(epoch_index, i + 1, last_loss))
        #     running_loss = 0.
    last_loss = running_loss / batch_index
    print('Epoch {}  loss: {}\n'.format(epoch_index, last_loss))
    # if last_loss < best_loss:
    #     best_loss = last_loss
    #     if accelerator.is_local_main_process:
    #         os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    #         print('Saving ckpt at Epoch {}  loss: {}\n'.format(epoch_index, last_loss))
    #         torch.save(model.state_dict(), args.model_path)
    all_losses = accelerator.gather(torch.tensor([last_loss]).to(device))

    global_best_loss = torch.min(all_losses).item()

    # If the current loss is the best, save the checkpoint in the process with the best loss
    if global_best_loss < best_loss:
        best_loss = global_best_loss
        if last_loss == best_loss:
            print(f'Saving checkpoint at Epoch {epoch_index} with loss: {last_loss}')
            print(device)
            # Ensure the model path exists and save the checkpoint
            torch.save(model.state_dict(), args.model_path)

    # Synchronize best loss across processes
    # accelerator.wait_for_everyone()
torch.save(model.state_dict(), args.model_path[:-4]+'_last.pth')