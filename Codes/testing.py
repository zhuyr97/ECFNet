import torch
import sys,os
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from networks.Network_V1_6 import MIMOUNet_complete
from datasets.dataset_pairs_npy import my_dataset_eval

sys.path.append(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device -------- :',device)

transform = transforms.Compose(
        [
         transforms.ToTensor()
        ])

pre_trained_model_path = './ckpt/model.pth'

SAVE_npy_PATH = './results/'
if not os.path.exists(SAVE_npy_PATH):
    os.mkdir(SAVE_npy_PATH)

## test data path
val_in_path = "./test_data/"
val_gt_path = val_in_path


def Inverse_tonemap( y, alpha=0.25):
    mapped_y = alpha*y / (np.abs(1-y ) + 0.00001)
    return mapped_y
trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])

if __name__ == '__main__':
    net =MIMOUNet_complete(base_channel=24, num_res=6).to(device)
    net.eval()
    net.load_state_dict(torch.load(pre_trained_model_path))
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label=val_gt_path, transform=trans_eval)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=0)

    with torch.no_grad():
        for index, (data_in, _, name) in enumerate(eval_loader, 0):
            print('i,-----------------:', index, '----', name)
            new_name = name[0].split('.')[0]
            inputs = Variable(data_in).to(device)

            outputs = net(inputs)

            outputs_np = torch.clamp(outputs[3], 0.0, 1.0).cpu().detach().numpy()[0, :, :, :]
            outputs_numpy = np.transpose(np.float32(outputs_np), (1, 2, 0))

            outputs_numpy_npy = Inverse_tonemap(outputs_numpy)
            np.save(SAVE_npy_PATH + new_name, outputs_numpy_npy)

