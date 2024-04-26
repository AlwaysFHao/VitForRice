import datetime
import os

import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossAndAccuracyHistory:
    def __init__(self, log_dir, model, input_shape, device):
        # 时间格式
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        # 日志保存路径
        self.log_dir = os.path.join(log_dir, "loss_and_acc_" + str(time_str))
        # self.log_dir = os.path.join(log_dir, "loss_and_acc_2023_04_19_14_14_12")
        # 训练损失保存列表
        self.losses = []
        # 验证损失保存列表
        self.val_loss = []
        # 训练准确度保存列表
        self.accs = []
        # 验证准确度保存列表
        self.val_accs = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

        # try:
        #     example_input = torch.rand(input_shape).to(device)
        #     # traced_script_module = torch.jit.trace(model, example_input)
        #     # traced_script_module.save("my_network.pt")
        #     # dummy_input = torch.randn(2, 10, 1, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, example_input)
        # except:
        #     pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def append_acc(self, epoch, acc, val_acc):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.accs.append(acc)
        self.val_accs.append(val_acc)

        with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
            f.write(str(acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f:
            f.write(str(val_acc))
            f.write("\n")

        self.writer.add_scalar('acc', acc, epoch)
        self.writer.add_scalar('val_acc', val_acc, epoch)
        self.acc_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.pdf"), format="pdf")

        plt.cla()
        plt.close("all")

    def acc_plot(self):
        iters = range(len(self.accs))

        plt.figure()
        plt.plot(iters, self.accs, 'red', linewidth=2, label='train accuracy')
        plt.plot(iters, self.val_accs, 'coral', linewidth=2, label='val accuracy')
        try:
            if len(self.accs) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.accs, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train accuracy')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_accs, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val accuracy')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right", fontsize='x-small')

        plt.savefig(os.path.join(self.log_dir, "epoch_accuracy.pdf"), format="pdf")

        plt.cla()
        plt.close("all")
