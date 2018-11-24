import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os

from sacred.observers import MongoObserver
import urllib
from private import user, pw, ip, db


def load(batch_size=256, seed=7):
    """Loads the MNIST data set.
    """
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
        
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

    torch.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return train_loader, test_loader


def get_mongo_uri():
    mongo_uri = ('mongodb://' + user + ':' + urllib.parse.quote(pw)
                 + '@' + ip + '/' + db)
    return mongo_uri


def get_ExpDB():
    mongo_uri = get_mongo_uri()
    ExpDB = MongoObserver.create(url=mongo_uri, db_name=db)
    return ExpDB