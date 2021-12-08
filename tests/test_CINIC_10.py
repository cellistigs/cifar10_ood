## Test pytorch implementation of CINIC Imagenet only dataset
import pytest
import sys
import os 
from pathlib import Path
import numpy as np
from torchvision import transforms as T
from PIL import Image
import zipfile

here = os.path.abspath(os.path.dirname(__file__))

from cifar10_ood.data import CINIC10,CIFAR10_C
from torchvision.datasets import CIFAR10

class Test_CINIC10:
    def test_init(self,tmpdir):
        """Should unzip to the right folder if it doesn't exist. 

        """
        downloads = tmpdir / "cinic-10"
        downloads.mkdir()
        downloads = os.path.join(here,"../data/cinic-10") 
        CINIC10(Path(downloads),"test")
        assert [li in ["train","test","valid"] for li in os.listdir(downloads)] ## basic check. 
        assert len(os.listdir(os.path.join(downloads,"test","airplane"))) == 7000


    #def test_parity(self,tmpdir):
    #    downloads = tmpdir / "cinic-10"
    #    downloads.mkdir()
    #    downloads = os.path.join(here,"../data/cinic-10") 
    #    cifar10 = CIFAR10(tmpdir,train = False,download = True) 
    #    cinic10 = CINIC10(Path(downloads),"test")
    #    for attr,value in cifar10.__dict__.items():
    #        if attr not in ["root","train"]:
    #            assert type(cinic10.__dict__[attr]) == type(value)
    #            if attr == "target":
    #                assert type(cinic10.__dict__[attr][0]) == type(value[0])

    #def test_parity_transforms(self,tmpdir):    
    #    """Check that the field parity is maintained under transforms. 

    #    """
    #    downloads = tmpdir / "cinic-10"
    #    downloads.mkdir()
    #    downloads = os.path.join(here,"../data/cinic-10") 
    #    transform = T.Compose(
    #        [
    #            T.RandomCrop(32, padding=4),
    #            T.RandomHorizontalFlip(),
    #            T.ToTensor(),
    #            T.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2)),
    #        ]
    #    )
    #    cifar10 = CIFAR10(tmpdir,train = False,download = True,transform = transform) 
    #    cinic10 = CINIC10(Path(downloads),"test",transform = transform)
    #    for attr,value in cifar10.__dict__.items():
    #        if attr not in ["root","train"]:
    #            assert type(cinic10.__dict__[attr]) == type(value)

    def test_classes(self,tmpdir):    
        """Save some images from both sets and make sure that the class labels make sense. 

        """
        downloads = tmpdir / "cinic-10"
        downloads.mkdir()
        downloads = os.path.join(here,"../data/cinic-10") 
        cifar10 = CIFAR10(downloads,train = False,download = True,transform=None ) 
        cinic10 = CINIC10(Path(downloads),"test",transform = None)
        class_indices = range(10) 
        for i in class_indices:
            cifar_ind = 0
            cinic_ind = 0
            cifar_classlabel = cifar10[cifar_ind][1]
            cinic_classlabel = cinic10[cinic_ind][1]

            while cifar_ind != i:
                cifar_ind+=1
            while cinic_ind != i:
                cinic_ind+=1
            cifarimage = cifar10[cifar_ind][0]
            cinicimage = cinic10[cinic_ind][0]

            cifarimage.save(os.path.join(here,"test_mats","cifar10_ex_{}.jpg").format(cifar10[cifar_ind][1]))    
            cinicimage.save(os.path.join(here,"test_mats","cinic10_ex_{}.jpg").format(cinic10[cinic_ind][1]))    

