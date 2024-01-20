import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging

from config import get_config
from data import load_dataset_test
from model import ResNet18, Nin, GoogleNet, Resnet34, Resnet50, Resnet101, Resnext50_32x4d, Resnext101_32x8d

class Niubility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        # Operate the method
        if args.model_name == 'GoogleNet':
            self.Mymodel = GoogleNet()
        elif args.model_name == 'Nin':
            self.Mymodel = Nin()
        elif args.model_name == 'ResNet18':
            self.Mymodel = ResNet18()
        elif args.model_name == 'ResNet34':
            self.Mymodel = Resnet34(num_classes=2, include_top=True)
        elif args.model_name == 'Resnet50':
            self.Mymodel = Resnet50(num_classes=2, include_top=True) # 我希望就是输出横纵坐标
        elif args.model_name == 'Resnet50_32x4d':
            self.Mymodel = Resnext50_32x4d(num_classes=2, include_top=True) # 我希望就是输出横纵坐标
        elif args.model_name == 'Resnet101':
            self.Mymodel = Resnet101(num_classes=2, include_top=True) # 我希望就是输出横纵坐标
        elif args.model_name == 'Resnet101_32x8d':
            self.Mymodel = Resnext101_32x8d(num_classes=2, include_top=True) # 我希望就是输出横纵坐标
        else:
            raise ValueError('unknown model')

        # 这个地方就把模型放到GPT上去了
        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        self.args.index += 1
        train_loss, n_train = 0, 0
        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            # print("input.shape:", inputs.shape)
            # print("targets.shape:", targets.shape)
            inputs, targets = inputs.to(torch.float).to(self.args.device), targets.to(torch.float).to(self.args.device)
            pred = self.Mymodel(inputs)
            # print("Shape of pred:", pred.shape)
            # print("Shape of targets:", targets.shape)
            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  #targets.size(0)指第一个维度的大小，也就是样本数量
            n_train += 1
        # print("n_train",n_train)
        return  train_loss / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_test = 0, 0
        self.Mymodel.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                # ascii：控制是否使用 ASCII 字符显示进度条，默认为 False disable 控制是否禁用进度条
                inputs, targets = inputs.to(torch.float).to(self.args.device), targets.to(torch.float).to(self.args.device)
                pred = self.Mymodel(inputs)
                # print("targets:",targets)
                # print("perd:",pred)
                loss = criterion(pred, targets)
                # print("targets.size(0):", targets.size(0))
                # print("loss.item()",loss.item())
                test_loss += loss.item()
                # print("n_test",n_test)
                n_test += 1
        # print("n_test",n_test)
        return test_loss / n_test

    def _predict(self, dataloader):
        self.Mymodel.load_state_dict(torch.load(f'weights/best_resnet50_500.pth'))

        self.Mymodel.eval()

        with torch.no_grad():
            for inputs in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = inputs.to(torch.float).to(self.args.device)
                pred = self.Mymodel(inputs)
            result = pred.cpu()
            return result.numpy()


    def run(self):
        # Print the parameters of model
        for name, layer in self.Mymodel.named_parameters(recurse=True):
            print(name, layer.shape, sep=" ")

        train_dataloader, test_dataloader, predict_dataloader = load_dataset_test(self)
        if (self.args.TP == 'train'):
            _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay) #该方法更利用冻结某些部分的参数
            # optimizer = torch.optim.AdamW(self.Mymodel.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

            index = 0
            l_teloss, l_trloss, l_epo = [], [], []
            # Get the best_loss and the best_acc
            best_loss = float('inf')
            for epoch in range(self.args.num_epoch):
                train_loss = self._train(train_dataloader, criterion, optimizer)
                test_loss = self._test(test_dataloader, criterion)
                if(epoch>50):
                    l_epo.append(epoch), l_trloss.append(train_loss), l_teloss.append(test_loss)
                if test_loss < best_loss:
                    best_loss = test_loss
                    index = epoch
                    torch.save(self.Mymodel.state_dict(), f'weights/best_{args.model_name}_{args.num_epoch}.pth')  # 保存模型训练过程中的最佳参数
                self.logger.info(
                    '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
                self.logger.info('[train] loss: {:.10f}, [test] loss: {:.10f}'.format(train_loss, test_loss))
            self.logger.info(
                'best loss: {:.10f}, best index: {:d}'.format(best_loss, index+1))
            self.logger.info('log saved: {}'.format(self.args.log_name))
            print(l_teloss)
            plt.figure(1)
            plt.plot(l_epo, l_teloss, label='Test Loss')
            plt.plot(l_epo, l_trloss, label='Train Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.show()
        elif (self.args.TP == 'predict'):
            result = self._predict(predict_dataloader)
            # 如果对标签进行了了归一化，需要加上下面的代码
            # for i in range(20):
            #     for j in range(2):
            #         if j == 0:
            #             result[i][j] = result[i][j] * 1956
            #         else:
            #             result[i][j] = result[i][j] * 1934
            # 如果对标签未使用归一化，使用下面的代码
            for i in range(20):
                for j in range(2):
                    if(i!=4):
                        if(j==0):
                            result[i][j] += 518
                        else:
                            result[i][j] += 33
            print(result)
            result = result.reshape(-1,1)
            res = pd.DataFrame(result, index=range(40))
            print(res)
            res.to_csv(f'result/submit_{args.model_name}_{args.num_epoch}__.csv')

if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    net = Niubility(args, logger)
    net.run()

