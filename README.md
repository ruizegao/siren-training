Required packages:

torch

numpy

siren-pytorch

accelerate

To test siren-pytorch installation run: python test.py

To train models run: accelerate launch --num_processes=4 train.py --num_epochs 50000 --data_path obj/obj.xyz --model_path model/model.pth --batch_size 500000
