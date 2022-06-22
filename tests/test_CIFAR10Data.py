## tests not for the main function of the CIFAR10Data module, but the addition of softmax based labels. 
from cifar10_ood.data import CIFAR10Data
import pytest
import os 
from hydra import compose, initialize

here = os.path.abspath(os.path.dirname(__file__))
## location where data is located. not using temp directory bc downloading new data takes time. 
data_dir = os.path.join(here,"../data")

class Test_CIFAR10Data():
    with initialize(config_path="./test_mats", job_name="test_app"):
        cfg = compose(config_name="run_default_cpu")
    traindata = CIFAR10Data(cfg).train_dataloader().dataset
    testdata = CIFAR10Data(cfg).test_dataloader().dataset
    traintargets = traindata.targets
    testtargets = testdata.targets

    def test_CIFAR10Data_train(self):
        """Check that passing a set of softmax outputs of the appropriate size as softmax_targets works, but otherwise will fail (i.e. if with wrong number of classes or wrong number of labels). 

        """
        with initialize(config_path="./test_mats", job_name="test_app"):
            cfg = compose(config_name="run_default_cpu")
            softmaxcfg = compose(config_name="run_default_cpu", overrides=["+softmax_targets={}".format(os.path.join(here,"test_mats","train_softmax_preds.npy"))])
            excfg = compose(config_name="run_default_cpu", overrides=["+softmax_targets={}".format(os.path.join(here,"test_mats","test_softmax_preds_less_classes.npy"))])
            classcfg = compose(config_name="run_default_cpu", overrides=["+softmax_targets={}".format(os.path.join(here,"test_mats","test_softmax_preds_less_examples.npy"))])

        data = CIFAR10Data(cfg).train_dataloader().dataset
        assert data.targets == self.traintargets
        data = CIFAR10Data(softmaxcfg).train_dataloader().dataset
        assert data.targets != self.traintargets
        with pytest.raises(AssertionError):
            data = CIFAR10Data(excfg).train_dataloader().dataset
        with pytest.raises(AssertionError):
            data = CIFAR10Data(classcfg).train_dataloader().dataset

    def test_CIFAR10Data_test(self):
        """Check that passing a set of softmax outputs of the appropriate size as softmax_targets works, but otherwise will fail (i.e. if with wrong number of classes or wrong number of labels) for test data. note that working with testing number of classes is kinda wonky. 

        """
        with initialize(config_path="./test_mats", job_name="test_app"):
            cfg = compose(config_name="run_default_cpu")
            softmaxcfg = compose(config_name="run_default_cpu", overrides=["+softmax_targets={}".format(os.path.join(here,"test_mats","test_softmax_preds.npy"))])
            excfg = compose(config_name="run_default_cpu", overrides=["+softmax_targets={}".format(os.path.join(here,"test_mats","test_softmax_preds_less_classes.npy"))])
            classcfg = compose(config_name="run_default_cpu", overrides=["+softmax_targets={}".format(os.path.join(here,"test_mats","test_softmax_preds_less_examples.npy"))])

        data = CIFAR10Data(cfg).test_dataloader().dataset
        assert data.targets == self.testtargets
        data = CIFAR10Data(softmaxcfg).test_dataloader().dataset
        assert data.targets != self.testtargets
        with pytest.raises(AssertionError):
            data = CIFAR10Data(classcfg).test_dataloader().dataset
        





