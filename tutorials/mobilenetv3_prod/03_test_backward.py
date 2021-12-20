# comparing backward grad with pytorch
# add loss comparing code

import torch
import paddle
import numpy as np
from reprod_log import ReprodDiffHelper

from models.mobilenet_v3_paddle import mobilenet_v3_small as mv3_small_paddle
from models.mobilenet_v3_torch import mobilenet_v3_small as mv3_small_torch


def train_one_epoch_paddle(inputs, labels, model, criterion, optimizer, lr_sche, max_iter)
    # train some iters 
    for idx in range(max_iter):
        image = paddle.to_tensor(inputs, dtype="float32")
        target = paddle.to_tensor(labels, dtype="int64")

        output = model(image)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        reprod_logger.add("lr_{}".format(idx), np.array(lr_sche.get_lr()))
    
    reprod_logger.save("./result/losses_paddle.npy")

def train_one_epoch_torch(inputs, labels, model, criterion, optimizer, lr_sche, max_iter)
    # train some iters 
    for idx in range(max_iter):
        image = torch.tensor(inputs, dtype="float32")
        target = torch.tensor(labels, dtype="int64")

        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sche.step()

        reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        reprod_logger.add("lr_{}".format(idx), np.array(lr_sche.get_last_lr()))
    
    reprod_logger.save("./result/losses_torch.npy")

def test_forward():
    max_iter = 10
    lr = 1e-3
    momentum = 0.9
    lr_gamma = 0.1

    # load paddle model
    paddle_model = mv3_small_paddle()
    paddle_model.eval()
    paddle_state_dict = paddle.load("./data/mv3_small_paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict)

    # load torch model
    torch_model = mv3_small_torch()
    torch_model.eval()
    torch_state_dict = torch.load("./data/mobilenet_v3_small-047dcff4.pth")
    torch_model.load_state_dict(torch_state_dict)

    # init loss
    criterion_paddle = paddle.nn.CrossEntropyLoss()
    criterion_torch = torch.nn.CrossEntropyLoss()

    # init optimizer
    lr_scheduler_paddle = paddle.optimizer.lr.StepDecay(
        lr, step_size=max_iter//3, gamma=lr_gamma)
    opt_paddle = paddle.optimizer.Momentum(
            learning_rate=lr_scheduler_paddle,
            momentum=momentum,
            parameters=model.parameters())
    lr_scheduler_torch = StepLR(opt_torch, step_size=max_iter//3, gamma=lr_gamma)
    opt_torch = torch.optim.SGD(model.parameters(), 
                                lr=lr, momentum=momentum)

    # prepare logger & load data
    reprod_logger = ReprodLogger()
    inputs = np.load("./data/fake_data.npy")
    labels = np.load("./data/fake_label.npy")
    print(inputs.shape, labels.shape)
    
    train_one_epoch_paddle(inputs, labels, paddle_model, criterion_torch, opt_paddle, lr_scheduler_paddle, max_iter)
    
    train_one_epoch_torch(inputs, labels, torch_model, criterion_paddle, opt_paddle, lr_scheduler_torch, max_iter)


if __name__ == "__main__":
    test_forward()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/losses_torch.npy")
    paddle_info = diff_helper.load_info("./result/losses_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/backward_diff.log")



