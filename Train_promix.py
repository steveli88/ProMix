from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
import dataloader_cifarn as dataloader
from model import *
from utils.utils import *
from utils.fmix import *
from sklearn.mixture import GaussianMixture
from datetime import datetime
import time


def label_guessing(idx_chosen, w_x, batch_size, score1, score2, match, args):
    w_x2 = w_x.clone()
    # when clean data is insufficient, try to incorporate more examples
    if (1. * idx_chosen.shape[0] / batch_size) < args.threshold:
        # both networks agree
        high_conf_cond2 = (score1 > args.tau) * (score2 > args.tau) * match
        # remove already selected examples; newly selected
        high_conf_cond2 = (1. * high_conf_cond2 - w_x.squeeze()) > 0     
        hc2_idx = torch.where(high_conf_cond2)[0]
        
        # maximally select (batch_size * args.threshold); where (idx_chosen.shape[0]) selected already
        max_to_sel_num = int(batch_size * args.threshold) - idx_chosen.shape[0]
        
        if high_conf_cond2.sum() > max_to_sel_num:
            # to many examples selected, remove some low conf examples
            score_mean = (score1 + score2) / 2
            idx_remove = (-score_mean[hc2_idx]).sort()[1][max_to_sel_num:]
            high_conf_cond2[hc2_idx[idx_remove]] = False
        w_x2[high_conf_cond2] = 1
    return w_x2

# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, pi1, pi2, pi1_unrel, pi2_unrel, args, directory, training_records):
    net.train()
    net2.train()  # train two peer networks in parallel

    # selection ratio for CSS
    rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup2(epoch, args.warmup_ep)
    # loss weight gamma(w) and lambda_u(beta)
    w = linear_rampup2(epoch, args.warmup_ep)
    alpha_output = args.debias_output
    debias_beta_pl = args.debias_pl
    beta = 0.1 * linear_rampup2(epoch, 2*args.warmup_ep) if debias_beta_pl else 1
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, w_x2, true_labels, index) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)
        w_x2 = w_x2.view(-1, 1).type(torch.FloatTensor)

        # inputs_x: weak augmentation
        # inputs_x2: strong augmentation
        inputs_x, inputs_x2, labels_x, w_x , w_x2= inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda(), w_x2.cuda()
        outputs_x, outputs_x_ph, _ = net(inputs_x,train=True,use_ph=True)
        outputs_x2, outputs_x2_ph, _ = net(inputs_x2,train=True,use_ph=True)
        outputs_a, outputs_a_ph, _ = net2(inputs_x,train=True,use_ph=True)
        outputs_a2, outputs_a2_ph, _ = net2(inputs_x2,train=True,use_ph=True)
        outputs_x_ori = outputs_x.clone().detach()
        outputs_a_ori = outputs_a.clone().detach()

        # debiasing logit for Debiased Margin-based Loss calculation of reliable samples D_l on primary head
        outputs_x = debias_output(outputs_x,pi1,alpha_output)
        outputs_x2 = debias_output(outputs_x2,pi1,alpha_output)
        outputs_a = debias_output(outputs_a,pi2,alpha_output)
        outputs_a2 = debias_output(outputs_a2,pi2,alpha_output)

        # debiasing logit for Debiased Margin-based Loss calculation of unreliable samples D_u on pseudo head
        outputs_x_unrel_ph = debias_output(outputs_x_ph,pi1_unrel,alpha_output)
        outputs_x2_unrel_ph = debias_output(outputs_x2_ph,pi1_unrel,alpha_output)
        outputs_a_unrel_ph = debias_output(outputs_a_ph,pi2_unrel,alpha_output)
        outputs_a2_unrel_ph = debias_output(outputs_a2_ph,pi2_unrel,alpha_output)

        # debiasing logit for Debiased Margin-based Loss calculation of reliable samples D_u on pseudo head
        outputs_x_ph = debias_output(outputs_x_ph,pi1,alpha_output)
        outputs_x2_ph = debias_output(outputs_x2_ph,pi1,alpha_output)
        outputs_a_ph = debias_output(outputs_a_ph,pi2,alpha_output)
        outputs_a2_ph = debias_output(outputs_a2_ph,pi2,alpha_output)

        with torch.no_grad():
            # original p, stored for distribution estimation
            px = torch.softmax(outputs_x_ori, dim=1)
            px2 = torch.softmax(outputs_a_ori, dim=1)

            # debiasing for the generation of pseudo-labels
            debias_px = debias_pl(outputs_x_ori,pi1,debias_beta_pl)
            debias_px2 = debias_pl(outputs_a_ori,pi2,debias_beta_pl)
            debias_px_unrel = debias_pl(outputs_x_ori,pi1_unrel,debias_beta_pl)
            debias_px2_unrel = debias_pl(outputs_a_ori,pi2_unrel,debias_beta_pl)
            # one-hot label for the samples selected by label guessing (LGA)
            pred_net = F.one_hot(debias_px.max(dim=1)[1], args.num_class).float()
            pred_net2 = F.one_hot(debias_px2.max(dim=1)[1], args.num_class).float()

            # matched high-confidence selection (MHCS)
            high_conf_cond = (labels_x * px).sum(dim=1) > args.tau
            high_conf_cond2 = (labels_x * px2).sum(dim=1) > args.tau
            w_x[high_conf_cond] = 1
            w_x2[high_conf_cond2] = 1

            # For CSS&MHCS: adopt original label; For LGA: adopt predicted label
            pseudo_label_l = labels_x * w_x + pred_net * (1 - w_x)
            pseudo_label_l2 = labels_x * w_x2 + pred_net2 * (1 - w_x2)

            idx_chosen = torch.where(w_x == 1)[0]
            idx_unchosen = torch.where(w_x != 1)[0]
            idx_chosen_2 = torch.where(w_x2 == 1)[0]
            idx_unchosen_2 = torch.where(w_x2 != 1)[0]

            # label guessing by agreement (LGA) for last K epochs
            if epoch > args.num_epochs - args.start_expand:
                score1 = px.max(dim=1)[0]
                score2 = px2.max(dim=1)[0]
                match = px.max(dim=1)[1] == px2.max(dim=1)[1]
                hc2_sel_wx1 = label_guessing(idx_chosen, w_x, batch_size, score1, score2, match, args)
                hc2_sel_wx2 = label_guessing(idx_chosen_2, w_x2, batch_size, score1, score2, match, args)
                idx_chosen = torch.where(hc2_sel_wx1 == 1)[0]
                idx_chosen_2 = torch.where(hc2_sel_wx2 == 1)[0]
                idx_unchosen = torch.where(hc2_sel_wx1 != 1)[0]
                idx_unchosen_2 = torch.where(hc2_sel_wx2 != 1)[0]

        # mixup loss for primary head $h$ of Net 1; adopt vanilla mixup and fmix: https://github.com/ecs-vlc/FMix
        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        X_w_c = inputs_x[idx_chosen]
        pseudo_label_c = pseudo_label_l[idx_chosen]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
        pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix = net(X_w_c_mix)
        logits_mix = debias_output(logits_mix,pi1,alpha_output)
        loss_mix = CEsoft(logits_mix, targets=pseudo_label_c_mix).mean()
        x_fmix = fmix(X_w_c)
        logits_fmix = net(x_fmix)
        logits_fmix = debias_output(logits_fmix,pi1,alpha_output)
        loss_fmix = fmix.loss(logits_fmix, (pseudo_label_c.detach()).long())

        # mixup loss for pseudo head $h_{AP}$ of Net 1
        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        X_w_c_ph = inputs_x[idx_chosen]
        pseudo_label_c = pseudo_label_l[idx_chosen]
        idx_ph = torch.randperm(X_w_c_ph.size(0))
        X_w_c_rand_ph = X_w_c_ph[idx_ph]
        pseudo_label_c_rand = pseudo_label_c[idx_ph]
        X_w_c_mix_ph = l * X_w_c_ph + (1 - l) * X_w_c_rand_ph
        pseudo_label_c_mix_ph = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        _,logits_mix_ph = net(X_w_c_mix_ph,use_ph=True)
        logits_mix_ph = debias_output(logits_mix_ph,pi1,alpha_output)
        loss_mix_ph = CEsoft(logits_mix_ph, targets=pseudo_label_c_mix_ph).mean()
        x_fmix_ph = fmix(X_w_c_ph)
        _,logits_fmix_ph = net(x_fmix_ph,use_ph=True)
        logits_fmix_ph = debias_output(logits_fmix_ph,pi1,alpha_output)
        loss_fmix_ph = fmix.loss(logits_fmix_ph, (pseudo_label_c.detach()).long())

        # consistency loss for primary head and pseudo head
        loss_cr = CEsoft(outputs_x2[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
        loss_cr_ph = CEsoft(outputs_x2_ph[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()

        # cross entropy loss for primary head and pseudo head
        loss_ce = CEsoft(outputs_x[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
        loss_ce_ph = CEsoft(outputs_x_ph[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
        # loss for net1-primary head
        loss_net1 = loss_ce + w * (loss_cr + loss_mix + loss_fmix)

        # loss for noisy samples on the pseudo head
        ptx = debias_px_unrel ** (1 / args.T)
        ptx = ptx / ptx.sum(dim=1, keepdim=True)
        beta = 0 if (epoch >= 2*args.warmup_ep and beta < 1) else beta
        targets_urel = ptx
        loss_unrel_ph = CEsoft(outputs_x_unrel_ph[idx_unchosen], targets=targets_urel[idx_unchosen]).mean()\
                  + w * CEsoft(outputs_x2_unrel_ph[idx_unchosen], targets=targets_urel[idx_unchosen]).mean()
        # loss for net1-pseudo head
        loss_net1_ph = beta * loss_unrel_ph + loss_ce_ph + w * (loss_cr_ph + loss_mix_ph + loss_fmix_ph)

        # -----Below: loss for net2, similar to net1-----

        # mixup loss for primary head of Net 2
        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        X_w_c = inputs_x[idx_chosen_2]
        pseudo_label_c = pseudo_label_l2[idx_chosen_2]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix2 = l * X_w_c + (1 - l) * X_w_c_rand        
        pseudo_label_c_mix2 = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix2 = net2(X_w_c_mix2)
        logits_mix2 = debias_output(logits_mix2,pi2,alpha_output)
        loss_mix2 = CEsoft(logits_mix2, targets=pseudo_label_c_mix2).mean()
        x_fmix2 = fmix(X_w_c)
        logits_fmix2 = net2(x_fmix2)
        logits_fmix2 = debias_output(logits_fmix2,pi2,alpha_output)
        loss_fmix2 = fmix.loss(logits_fmix2, (pseudo_label_c.detach()).long())

        # mixup loss for pseudo head of Net 2
        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        X_w_c_ph = inputs_x[idx_chosen_2]
        pseudo_label_c = pseudo_label_l2[idx_chosen_2]
        idx = torch.randperm(X_w_c_ph.size(0))
        X_w_c_rand_ph = X_w_c_ph[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix2_ph = l * X_w_c_ph + (1 - l) * X_w_c_rand_ph        
        pseudo_label_c_mix2 = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        _,logits_mix2_ph = net2(X_w_c_mix2_ph,use_ph=True)
        logits_mix2_ph = debias_output(logits_mix2_ph,pi2,alpha_output)
        loss_mix2_ph = CEsoft(logits_mix2_ph, targets=pseudo_label_c_mix2).mean()
        x_fmix2_ph = fmix(X_w_c_ph)
        _,logits_fmix2_ph = net2(x_fmix2_ph,use_ph=True)
        logits_fmix2_ph = debias_output(logits_fmix2_ph,pi2,alpha_output)
        loss_fmix2_ph = fmix.loss(logits_fmix2_ph, (pseudo_label_c.detach()).long())

        # consistency loss for primary head and pseudo head
        loss_cr2 = CEsoft(outputs_a2[idx_chosen_2], targets=pseudo_label_l2[idx_chosen_2]).mean()
        loss_cr2_ph = CEsoft(outputs_a2_ph[idx_chosen_2], targets=pseudo_label_l2[idx_chosen_2]).mean()
        # cross entropy loss for primary head and pseudo head
        loss_ce2 = CEsoft(outputs_a[idx_chosen_2], targets=pseudo_label_l2[idx_chosen_2]).mean()
        loss_ce2_ph = CEsoft(outputs_a_ph[idx_chosen_2], targets=pseudo_label_l2[idx_chosen_2]).mean()
        loss_net2 = loss_ce2 + w * (loss_cr2 + loss_mix2 + loss_fmix2)
        # Above: loss for net2-primary head

        # unrel loss for reliable samples on the pseudo head
        ptx2 = debias_px2_unrel  ** (1 / args.T)
        ptx2 = ptx2 / ptx2.sum(dim=1, keepdim=True)
        targets_urel2 = ptx2
        loss_unrel2_ph = CEsoft(outputs_a_unrel_ph[idx_unchosen_2], targets=targets_urel2[idx_unchosen_2]).mean()\
                  + w * CEsoft(outputs_a2_unrel_ph[idx_unchosen_2], targets=targets_urel2[idx_unchosen_2]).mean()
        # loss for net2-pseudo head
        loss_net2_ph = beta * loss_unrel2_ph + loss_ce2_ph + w * (loss_cr2_ph + loss_mix2_ph + loss_fmix2_ph)
        #

        # total loss
        loss = loss_net1 + loss_net2 + loss_net1_ph + loss_net2_ph
        # moving average estimation of bias for D_l and D_u seperately
        pi1 = bias_update(px[idx_chosen], pi1, args.bias_m)
        pi2 = bias_update(px2[idx_chosen_2], pi2, args.bias_m)
        pi1_unrel = bias_update(px[idx_unchosen], pi1_unrel, args.bias_m)
        pi2_unrel = bias_update(px2[idx_unchosen_2], pi2_unrel, args.bias_m)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # todo gradient clip for low noise
        if torch.isnan(loss_net1) and torch.isnan(loss_net2):
            exit()

        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "%s:%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Net1 loss: %.2f  Net2 loss: %.2f"
                % (
                    args.dataset,
                    args.noise_type,
                    epoch,
                    args.num_epochs,
                    batch_idx + 1,
                    num_iter,
                    loss_net1.item(),
                    loss_net2.item(),
                )
            )

    return pi1,pi2,pi1_unrel,pi2_unrel


def warmup(epoch, net, net2, optimizer, dataloader):
    net.train()
    net2.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs_w, inputs_s, labels, _) in enumerate(dataloader):
        inputs_w, inputs_s, labels = inputs_w.cuda(), inputs_s.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs_w)
        outputs2 = net2(inputs_w)
        l_ce = CEloss(outputs, labels)
        l_ce2 = CEloss(outputs2, labels)
        loss = l_ce + l_ce2
        penalty = conf_penalty(outputs) + conf_penalty(outputs2)
        if args.noise_mode=='asym': 
            L = loss + penalty
        else:
            L = loss
        L.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('%s:%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f  Penalty-loss: %.4f  All-loss: %.4f'
                         % (
                         args.dataset, args.noise_type, epoch, args.num_epochs, batch_idx + 1, num_iter,loss.item(),penalty.item(), L.item()))

def evaluate(loader, model, save = False, best_acc = 0.0):
    model.eval()    # Change model to 'eval' mode.
    
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images = torch.autograd.Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)
    if save:
        if acc > best_acc:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            save_path= os.path.join('./', args.noise_type +'best.pth.tar')
            torch.save(state,save_path)
            best_acc = acc
            print(f'model saved to {save_path}!')
    return acc

def test(epoch, net1, net2, test_loader, test_log):
    net1.eval()
    net2.eval()
    correct = 0
    correct2 = 0
    correctmean = 0
    correctmean_ori = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1_ori,outputs1 = net1(inputs,use_ph=True)
            outputs2_ori,outputs2 = net2(inputs,use_ph=True)
            score1, predicted = torch.max(outputs1, 1)
            score2, predicted_2 = torch.max(outputs2, 1)
            #model ensemble for inference
            outputs_mean_ori = (outputs1_ori + outputs2_ori) / 2
            _, predicted_mean_ori = torch.max(outputs_mean_ori, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            correct2 += predicted_2.eq(targets).cpu().sum().item()
            correctmean_ori += predicted_mean_ori.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    acc2 = 100. * correct2 / total
    accmean_ori = 100. * correctmean_ori / total
    message = "| Test Epoch #%d\t Acc Net1: %.2f%%, Acc Net2: %.2f%% Acc Mean: %.2f%%\n" % (epoch, acc, acc2, accmean_ori)
    print(message)
    test_log.write(message)
    test_log.flush()
    return accmean_ori

def eval_train(model, rho, num_class, eval_loader, args, all_loss):
    w = linear_rampup2(epoch, args.warmup_ep)
    model.eval()
    losses = torch.zeros(50000)
    targets_list = torch.zeros(50000, dtype=torch.long)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            losses[index] = loss.cpu()
            targets_list[index] = targets.cpu()

    #class-wise small-loss selection (CSS for base selection set)
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = losses.detach()
    # # Todo remove the gradient information
    # all_loss.append(losses)
    # # todo consider average loss when using low loss criterion
    # average_losses = torch.mean(torch.stack(all_loss[-5:]), dim=0)

    prob = np.zeros(targets_list.shape[0])

    for c in range(num_class):
        indices_class = (targets_list == c).nonzero().squeeze()
        input_loss_class = torch.index_select(losses, 0, indices_class)
        # todo adjust number of labels controlled here
        num_sample_per_class = targets_list.shape[0] * (1. / num_class)
        k = max(min(int(math.ceil(num_sample_per_class * rho)), len(indices_class)), 1)
        _, low_loss_indices_class = input_loss_class.topk(k, dim=0, largest=False)
        prob[indices_class[low_loss_indices_class.squeeze()]] = 1

    return prob, all_loss

class NegEntropy(object):
    def __call__(self, outputs):
        outputs = outputs.clamp(min=1e-12)
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

def create_model():
    model = DualNet(args.num_class)
    model = model.cuda()
    return model


def sort_dict(myDict):
    myKeys = list(myDict.keys())
    myKeys.sort()
    sorted_dict = {i: myDict[i] for i in myKeys}
    return sorted_dict


def label_stats(noisy_label, true_label, epoch, rho, selection_acc, log):
    label_stats = {}
    correct_label_stats = {}
    correct_label = 0
    for i in range(len(noisy_label)):
        if noisy_label[i] in label_stats:
            label_stats[noisy_label[i]] += 1
        else:
            label_stats[noisy_label[i]] = 1
        if noisy_label[i] == true_label[i]:
            correct_label += 1
            if noisy_label[i] in correct_label_stats:
                correct_label_stats[noisy_label[i]] += 1
            else:
                correct_label_stats[noisy_label[i]] = 1

    label_stats = sort_dict(label_stats)
    correct_label_stats = sort_dict(correct_label_stats)

    log.write(f"Current rho is {rho} \n")
    log.write("Epoch %d \n" % epoch)
    log.write("Number of labels for classes: %s \n" % label_stats)
    log.write("Correct labels for classes: %s \n" % correct_label_stats)
    log.write("Overall accuracy: %.2f \n" % (correct_label / len(noisy_label)))
    log.write("Total sample selected: %.2f \n" % (sum(label_stats.values())))
    log.write("Total clean sample selected: %.2f \n" % (sum(correct_label_stats.values())))
    # for key in correct_label_stats:
    #     log.write('The Precision of Class %d is %.2f \n' % (key, correct_label_stats[key] / label_stats[key]))
    log.flush()

    if selection_acc < (correct_label / len(noisy_label)):
        selection_acc = correct_label / len(noisy_label)

    return rho, selection_acc


def low_loss_sample_stats(
    prob1, prob2, total_trainloader, noisy_labels, epoch, rho, selection_acc, stats_log
):
    """ """
    low_loss_labels = torch.tensor(noisy_labels)[np.nonzero(prob1)]
    low_loss_labels_truth = torch.tensor(total_trainloader.dataset.train_labels)[
        np.nonzero(prob1)
    ]
    stats_log.write("Low loss labels from Net 1\n")
    _, _ = label_stats(
        low_loss_labels.tolist(),
        low_loss_labels_truth.tolist(),
        epoch,
        rho,
        selection_acc,
        stats_log,
    )
    low_loss_labels = torch.tensor(noisy_labels)[np.nonzero(prob2)]
    low_loss_labels_truth = torch.tensor(total_trainloader.dataset.train_labels)[
        np.nonzero(prob2)
    ]
    stats_log.write("Low loss labels from Net 2\n")
    _, _ = label_stats(
        low_loss_labels.tolist(),
        low_loss_labels_truth.tolist(),
        epoch,
        rho,
        selection_acc,
        stats_log,
    )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--cosine', action='store_true', default=False,
                        help='use cosine lr schedule')
    parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100',
                        default='clean')
    parser.add_argument('--noise_path', type=str, help='path of CIFAR-10_human.pt', default=None)
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=600, type=int)
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=100, type=int)
    parser.add_argument('--data_path', default=None, type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--is_human', action='store_true', default=False)
    parser.add_argument('--rho_range', default='0.2,0.6', type=str,
                        help='ratio of selecting clean labels (rho)')
    parser.add_argument('--tau', default=0.99, type=float,
                        help='high-confidence selection threshold')
    parser.add_argument('--pretrain_ep', default=10, type=int, help = 'warm-up training epoch')
    parser.add_argument('--warmup_ep', default=50, type=int, help = 'parameter ramp-up epoch')
    parser.add_argument('--low_conf_del', action='store_true', default=False)
    parser.add_argument('--threshold', default=0.9, type=float, help = 'threshold of label guessing')
    parser.add_argument('--fmix', action='store_true', default=False)
    parser.add_argument('--start_expand', default=250, type=int)               
    parser.add_argument('--debias_output', default=0.8, type=float,
                        help='debias strength for loss calculation')
    parser.add_argument('--debias_pl', default=0.8, type=float,
                        help='debias strength for pseudo-label generation')
    parser.add_argument('--noise_mode', default='cifarn', type=str,help='cifarn, sym, asym')
    parser.add_argument('--noise_rate', default=0.2, type=float,
                        help='noise rate for synthetic noise')
    parser.add_argument('--bias_m', default=0.9999, type=float,
                        help='moving average parameter of bias estimation')
    parser.add_argument('--eps', default=1, type=float, help='Epsilon')
    parser.add_argument('--cluster_prior_epoch', default=100, type=int)
    parser.add_argument('--cluster_file', default='feature_clusters_cifar10_r50_b384_e1000_c1000.pt', 
                        type=str, help='path to cluster file')
    parser.add_argument('--num_cluster', default=1000, type=int)
    args = parser.parse_args()
    [args.rho_start, args.rho_end] = [float(item) for item in args.rho_range.split(',')]
    print(args)

    cluster_file = args.cluster_file
    n_clusters = args.num_cluster

    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Hyper Parameters
    noise_type_map = {
        "clean": "clean_label",
        "worst": "worse_label",
        "aggre": "aggre_label",
        "rand1": "random_label1",
        "rand2": "random_label2",
        "rand3": "random_label3",
        "clean100": "clean_label",
        "noisy100": "noisy_label",
    }
    args.noise_type = noise_type_map[args.noise_type]
    # load dataset
    # please change it to your own datapath
    if args.data_path is None:
        if args.dataset == 'cifar10':
            args.data_path = 'data/cifar-10-batches-py'
            args.num_class = 10
            torchvision.datasets.CIFAR10('data',train=True, download=True)
            torchvision.datasets.CIFAR10('data',train=False, download=True)
        elif args.dataset == 'cifar100':
            args.data_path = 'data/cifar-100-python'
            args.num_class = 100
            torchvision.datasets.CIFAR100('data',train=True, download=True)
            torchvision.datasets.CIFAR100('data',train=False, download=True)
        else:
            pass
    # please change it to your own datapath for CIFAR-N
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './data/CIFAR-100_human.pt'
        else:
            pass

    curr_time = time.strftime("%m%d%H%M", time.localtime())
    directory = os.path.join('checkpoint', f'{curr_time}_{args.dataset}_{args.noise_mode}_{args.noise_rate}')
    if not os.path.exists(directory):
        os.makedirs(directory)

    stats_name = f'{args.dataset}_{args.noise_type}_{args.num_epochs}_stats_{curr_time}.txt'
    test_name = f'{args.dataset}_{args.noise_type}_{args.num_epochs}_acc_{curr_time}.txt'
    stats_log = open(os.path.join(directory, stats_name), 'w')
    test_log = open(os.path.join(directory, test_name), 'w')
    test_log.write(str(args) + '\n')
    # test_log.write(cluster_file + '\n')

    warm_up = args.pretrain_ep
    # unique file name to record the synthetic noise for CIFAR-10/100
    time_digits = str(datetime.now())[-6:]
    noise_file = os.path.join(directory, f'{args.noise_type}_{time_digits}.json')
    loader = dataloader.cifarn_dataloader(
        args.dataset,
        noise_type=args.noise_type,
        noise_path=args.noise_path,
        is_human=args.is_human,
        batch_size=args.batch_size,
        num_workers=8,
        root_dir=args.data_path,
        log=stats_log,
        noise_file=noise_file,
        r=args.noise_rate,
        noise_mode=args.noise_mode,
    )
    test_log.write(noise_file + '\n')

    print('| Building net')
    dualnet = create_model()
    cudnn.benchmark = True

    conf_penalty = NegEntropy()
    optimizer1 = optim.SGD(
        [{"params": dualnet.net1.parameters()}, {"params": dualnet.net2.parameters()}],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    fmix = FMix()
    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    CEsoft = CE_Soft_Label()
    eval_loader, noise_or_not = loader.run('eval_train')
    test_loader = loader.run('test')

    all_loss = []
    rho = args.rho_start
    selection_acc = 0
    best_acc = 0
    # uniform initialization of distribution estimation
    pi1 = bias_initial(args.num_class)
    pi2 = bias_initial(args.num_class)
    pi1_unrel = bias_initial(args.num_class)
    pi2_unrel = bias_initial(args.num_class)

    training_records = None

    clean_sample_1 = np.zeros((50000, args.num_epochs), dtype=int)
    clean_sample_2 = np.zeros((50000, args.num_epochs), dtype=int)
    clean_sample_cluster_1 = np.zeros((50000, args.num_epochs), dtype=int)
    clean_sample_cluster_2 = np.zeros((50000, args.num_epochs), dtype=int)

    for epoch in range(args.num_epochs + 1):
        adjust_learning_rate(args, optimizer1, epoch)

        if epoch < warm_up:
            warmup_trainloader, noisy_labels = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, dualnet.net1, dualnet.net2, optimizer1, warmup_trainloader)
        else:
            rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup2(epoch, args.warmup_ep)
            prob1, all_loss = eval_train(dualnet.net1, rho, args.num_class, eval_loader, args, all_loss)
            prob2, all_loss = eval_train(dualnet.net2, rho, args.num_class, eval_loader, args, all_loss)
            pred1 = (prob1 > args.p_threshold)

            clean_sample_1[:, epoch] = prob1
            np.save(os.path.join(directory, 'clean_sample_1.npy'), clean_sample_1)
            clean_sample_2[:, epoch] = prob2
            np.save(os.path.join(directory, 'clean_sample_2.npy'), clean_sample_2)       

            prob1_before_expansion, prob2_before_expansion = prob1, prob2            

            if epoch <= args.cluster_prior_epoch:
                low_loss_idx_1 = torch.tensor(prob1 > args.p_threshold).cuda()
                low_loss_idx_2 = torch.tensor(prob2 > args.p_threshold).cuda()
                expanded_low_loss_idx_1 = torch.clone(low_loss_idx_1)
                expanded_low_loss_idx_2 = torch.clone(low_loss_idx_2)
                # expand correct label via clustering
                cls = torch.load(cluster_file)
                noisy_labels_tensor = torch.tensor(noisy_labels).cuda()
                for i in range(n_clusters):
                    correct_labels_1 = torch.masked_select(noisy_labels_tensor[cls[i]['idx']], low_loss_idx_1[cls[i]['idx']])
                    expanded_low_loss_1 = torch.isin(noisy_labels_tensor[cls[i]['idx']], correct_labels_1) + low_loss_idx_1[cls[i]['idx']]
                    expanded_low_loss_idx_1[cls[i]['idx']] = expanded_low_loss_1

                    correct_labels_2 = torch.masked_select(noisy_labels_tensor[cls[i]['idx']], low_loss_idx_2[cls[i]['idx']])
                    expanded_low_loss_2 = torch.isin(noisy_labels_tensor[cls[i]['idx']], correct_labels_2) + low_loss_idx_2[cls[i]['idx']]
                    expanded_low_loss_idx_2[cls[i]['idx']] = expanded_low_loss_2

                prob1 = (expanded_low_loss_idx_1 * 1.).cpu().numpy()
                prob2 = (expanded_low_loss_idx_2 * 1.).cpu().numpy()

                clean_sample_cluster_1[:, epoch] = prob1
                np.save(os.path.join(directory, 'clean_sample_cluster_1.npy'), clean_sample_cluster_1)
                clean_sample_cluster_2[:, epoch] = prob2
                np.save(os.path.join(directory, 'clean_sample_cluster_2.npy'), clean_sample_cluster_2)

            total_trainloader, noisy_labels = loader.run('train', pred1, prob1, prob2)  # co-divide

            # Before expansion
            stats_log.write('Before expansion\n')
            low_loss_sample_stats(prob1_before_expansion, prob2_before_expansion, total_trainloader, noisy_labels, epoch, rho, selection_acc, stats_log)
            # after expansion
            stats_log.write('After expansion\n')
            low_loss_sample_stats(prob1, prob2, total_trainloader, noisy_labels, epoch, rho, selection_acc, stats_log)
            stats_log.write('\n')

            pi1, pi2, pi1_unrel, pi2_unrel = train(
                epoch,
                dualnet.net1,
                dualnet.net2,
                optimizer1,
                total_trainloader,
                pi1,
                pi2,
                pi1_unrel,
                pi2_unrel,
                args,
                directory,
                training_records,
            )

        acc = test(epoch, dualnet.net1, dualnet.net2, test_loader, test_log)

        if acc >= best_acc:
            best_acc = acc
            torch.save(dualnet, os.path.join(directory, f"{args.dataset}_{args.noise_type}best.pth.tar"))
            print(f'Model saved!')

        # save all information for resume training
        # pi1, pi2, pi1_unrel, pi2_unrel
        # best_acc, epoch, args,
        # dualnet, optimizer1, stats_log, test_log, noise_file
        save_filename = f'latest_model_{curr_time}.pth.tar'
        torch.save(
            {
                'model': dualnet,
                'optimizer': optimizer1,
                'args': args,
                'epoch': epoch,
                'best_acc': best_acc,
                'pi1': pi1,
                'pi2': pi2,
                'pi1_unrel': pi1_unrel,
                'pi2_unrel': pi2_unrel,
                'curr_time': curr_time,
                'noise_file': noise_file,
            },
            os.path.join(directory, save_filename))

        torch.cuda.empty_cache()
