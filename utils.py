import os
import shutil

import torch
import yaml

import subprocess
import os
import shutil


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def install_package(package):
    subprocess.check_call(["pip", "install", package])

def download_file(file_id, output_name):
    install_package("gdown")
    import gdown
    gdown.download(id=file_id, output=output_name, quiet=False)

def unzip_file(zip_file):
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(zip_file)

def remove_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)



if __name__ == "__main__":
    download_file('1LShk6tZlsdBO-DciChK7y7nivUOvTAFk', 'FERPlus.zip')
    unzip_file('FERPlus.zip')
    remove_directory('fer_plus/train/contempt')
    remove_directory('fer_plus/val/contempt')
    remove_directory('fer_plus/test/contempt')