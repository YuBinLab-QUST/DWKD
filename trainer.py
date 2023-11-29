# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather, AverageMeter
import torch.utils.data.distributed
from monai.data import decollate_batch
import pdb

def compute_loss(model, teacher_model, inputs, target):
    # pdb.set_trace()
    with torch.no_grad():
        teacher_logits = teacher_model(inputs)
    # pdb.set_trace()
    loss,kd_loss, dice_loss,entropy_loss = model(inputs, target, teacher_logits)

    return loss,kd_loss, dice_loss,entropy_loss

def train_epoch(model,
                teacher_model,
                loader,
                optimizer,
                scaler,
                epoch,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_kd_loss = AverageMeter()
    run_dice_loss = AverageMeter()
    run_entropy_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            loss, kd_loss, dice_loss,entropy_loss = compute_loss(model, teacher_model, data, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # pdb.set_trace()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name)
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],out_numpy=True,is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),n=args.batch_size * args.world_size)
            
            run_kd_list = distributed_all_gather([kd_loss],out_numpy=True,is_valid=idx < loader.sampler.valid_length)
            run_kd_loss.update(np.mean(np.mean(np.stack(run_kd_list, axis=0), axis=0), axis=0),n=args.batch_size * args.world_size)
            
            run_dice_list = distributed_all_gather([dice_loss],out_numpy=True,is_valid=idx < loader.sampler.valid_length)
            run_dice_loss.update(np.mean(np.mean(np.stack(run_dice_list, axis=0), axis=0), axis=0),n=args.batch_size * args.world_size)
            
            run_entropy_list = distributed_all_gather([entropy_loss],out_numpy=True,is_valid=idx < loader.sampler.valid_length)
            run_entropy_loss.update(np.mean(np.mean(np.stack(run_entropy_list, axis=0), axis=0), axis=0),n=args.batch_size * args.world_size)
        else:
            run_dice_loss.update(kd_loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'kd_loss: {:.8f}'.format(run_kd_loss.avg),
                    'entropy_loss: {:.8f}'.format(run_entropy_loss.avg),
                    'dice_loss: {:.4f}'.format(run_dice_loss.avg),
                  'lr {:.8f}'.format(optimizer.param_groups[-1]['lr']),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters(): param.grad = None
    return run_dice_loss.avg


def val_epoch(model,
              loader,
              epoch,
              acc_func,
              args,
              model_inferer=None,
              post_sigmoid=None,
              post_pred=None,
              ):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data['image'], batch_data['label']
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans],
                                                                 out_numpy=True,
                                                                 is_valid=idx < loader.sampler.valid_length)
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                Dice_TC = run_acc.avg[0]
                Dice_WT = run_acc.avg[1]
                Dice_ET = run_acc.avg[2]
                print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                      ', Dice_TC:', Dice_TC,
                      ', Dice_WT:', Dice_WT,
                      ', Dice_ET:', Dice_ET,
                      ', time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()

    return run_acc.avg


def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
        'epoch': epoch,
        'best_acc': best_acc,
        'state_dict': state_dict
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)


def run_training(model,
                 teacher_model,
                 train_loader,
                 val_loader,
                 optimizer,
                 acc_func,
                 args,
                 val_acc_max=0.,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 post_sigmoid=None,
                 post_pred=None,
                 semantic_classes=None
                 ):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    loss = 1.
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()
        # model.module.student.output_hidden_states=True
        train_loss = train_epoch(model,
                                teacher_model,
                                 train_loader,
                                 optimizer,
                                 scaler=scaler,
                                 epoch=epoch,
                                 args=args)
        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))
        if args.rank == 0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            
        if loss > train_loss and args.rank == 0:
            model.eval()
            save_checkpoint(model,
                epoch,
                args,
                best_acc=val_acc_max,
                filename='model_best_loss.pt')
            loss = train_loss
            
        if (epoch + 1) % 100 == 0 and args.rank == 0:
            model.eval()
            save_checkpoint(model,
                epoch,
                args,
                best_acc=val_acc_max,
                filename='model_epoch'+str(epoch)+'.pt')
        b_new_best = False  
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            # model.module.student.output_hidden_states=False
            val_acc = val_epoch(model,
                                val_loader,
                                epoch=epoch,
                                acc_func=acc_func,
                                model_inferer=model_inferer,
                                args=args,
                                post_sigmoid=post_sigmoid,
                                post_pred=post_pred,
                                )

            if args.rank == 0:
                Dice_TC = val_acc[0]
                Dice_WT = val_acc[1]
                Dice_ET = val_acc[2]
                print('Final validation stats {}/{}'.format(epoch, args.max_epochs - 1),
                      ', Dice_TC:', Dice_TC,
                      ', Dice_WT:', Dice_WT,
                      ', Dice_ET:', Dice_ET,
                      ', time {:.2f}s'.format(time.time() - epoch_time))

                if writer is not None:
                    writer.add_scalar('Mean_Val_Dice', np.mean(val_acc), epoch)
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
                val_avg_acc = np.mean(val_acc)
                if val_avg_acc > val_acc_max:
                    print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args,
                                        best_acc=val_acc_max,
                                        optimizer=optimizer,
                                        scheduler=scheduler)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_acc_max,
                                filename='model_final.pt')
                if b_new_best:
                    print('Copying to model.pt new best model!!!!')
                    shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

        if scheduler is not None:
            scheduler.step()

    print('Training Finished !, Best Accuracy: ', val_acc_max)
    if writer is not None:
        writer.close()
    return val_acc_max
