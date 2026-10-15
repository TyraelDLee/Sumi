import math
import argparse
import open3d as o3d
from datetime import datetime
from torch import optim
from torch.utils.data import DataLoader
from utils.dataset import *

from model.Sumi import SumiCompletion
from utils.mailman import mailman

import shutil

from visualization import plot_pcd_one_view

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

parser = argparse.ArgumentParser()
# Experiment setting
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--val_batch_size', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda')  # mps for mac
parser.add_argument('--log', type=bool, default=False)
parser.add_argument('--save_dir', type=str, default='./exp')
parser.add_argument('--exp_name', type=str, default='Sumi')
parser.add_argument('--early_stop',type=bool, default=True)
parser.add_argument('--warmup', type=int, default=200) # 0 for disable warmup
parser.add_argument('--loss', type=str, default='cdl1')

# Grouping setting
parser.add_argument('--num_output', type=int, default=16384)
parser.add_argument('--up_factor', type=int, nargs="+", default=[2, 8])
parser.add_argument('--include_input', type=bool, default=False)


parser.add_argument('--lst_dim', type=int, default=512)
parser.add_argument('--encoder_dim', type=int, default=128)


# diffusion
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')

# sche / optim
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--decay_step', type=int, default=21)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--lowest_decay', type=float, default=0.02)

parser.add_argument('--bn_decay', type=float, default=0.5)
parser.add_argument('--bn_momentum', type=float, default=0.9)
parser.add_argument('--bn_lowest_decay', type=float, default=0.01)


parser.add_argument('--remark', type=str, default='Abl test with more fusion heads')

args = parser.parse_args()
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M")
save_dir = os.path.join(args.save_dir, '{exp}_{date}'.format(exp=args.exp_name, date=time_now))
log_file = ""
epochs_dir = os.path.join(save_dir, 'epochs')
evaluate_dir = os.path.join(save_dir, 'eval')
if args.log:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(epochs_dir):
        os.makedirs(epochs_dir)
    if not os.path.exists(evaluate_dir):
        os.makedirs(evaluate_dir)
    shutil.copy("./model/{exp}.py".format(exp=args.exp_name), os.path.join(save_dir, 'model.py'))
    shutil.copy("train_{exp}.py".format(exp=args.exp_name), os.path.join(save_dir, 'train.py'))
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s::%(levelname)s]  %(message)s')
    log_file = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    log_file.setLevel(logging.INFO)
    log_file.setFormatter(formatter)
    log.addHandler(log_file)



if args.log:
    log.info('loading dataset')
print('loading dataset')

train_dset = PCN_pcd('../dataset/ShapeNetCompletion', 'train')
trn_loader = DataLoader(train_dset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=8)

val_dset = PCN_pcd('../dataset/ShapeNetCompletion', 'test')
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, pin_memory=True, shuffle=False, num_workers=8)

if args.log:
    log.info('Training Stable Diffusion')
    log.info('config:')
    log.info(args)
    log.info('dataset loaded')
print('dataset loaded')

# pretrined = AdaPoinTr(args).to(args.device)
# pretrined = torch.nn.DataParallel(pretrined, device_ids=[0,1,2])
# check_point_dir = os.path.join('./results/OriAdaPointr_baseline/best.pt')
# check_point = torch.load(check_point_dir)['model']
# pretrined.load_state_dict(check_point, strict=True)

model = SumiCompletion(args).to(args.device)
model = torch.nn.DataParallel(model, device_ids=[0])

# check_point_dir = os.path.join('./results/original_adapointtr_w_noise_2025_12_19_16_11/best.pt')
# check_point = torch.load(check_point_dir)['model']
# model.load_state_dict(check_point, strict=False)

optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0005)

# warming_up_t = 0
# lr_lbmd = lambda e: max(args.lr_decay ** ((e - warming_up_t) / args.decay_step), args.lowest_decay) if e >= warming_up_t else max(e / warming_up_t, 0.001)
#
#
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=-1)
lr_cos = lambda e: (e + 1) / 20 if e < 20 else \
    1.0 / 0.0002 * (0.00001 + 0.5 * (0.0002 - 0.00001) * (
                1.0 + math.cos((e - 20) / (350 - 20) * math.pi)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_cos)
# bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay ** (e / args.decay_step), args.lowest_decay)
# scheduler2 = BNMomentumScheduler(model, bnm_lmbd, last_epoch=-1)

# scheduler = [scheduler, scheduler2]

BEST_RESULTS = 100000
BEST_EPOCH = -1

def train(i, batch, epoch):
    _, partial, gt = batch
    partial = partial.float().cuda()
    gt = gt.float().cuda()
    if i == 0:
        if args.early_stop and epoch > args.warmup or ((epoch - 1) % 10 == 0 and epoch > 250 or epoch == 21):
            early_stop_validate(epoch)
        print('Saving checkpoint at epoch {epoch}'.format(epoch=epoch))
        save_model = {'args': args, 'model': model.state_dict()}
        if args.log:
            torch.save(save_model, os.path.join(save_dir, 'autoencoder_diff.pt'))
    optimizer.zero_grad()
    # pretrined.eval()
    # x, global_features, coor, coarse, _ = pretrined.module.get_coarse(partial, training=True)
    model.train()
    t = torch.randint(1, args.num_steps, (gt.size(0),))
    # pred = (x, coor, coarse, 64)
    ret = model(partial, gt, t)
    loss, sparse_loss, dense_loss, sparse_penalty, dense_penalty = model.module.get_loss(ret, gt)
    # _loss = loss_recon + loss_denoised
    loss.backward()


    optimizer.step()
    text = ('epoch: {epoch:>3}, iteration: {i:>4}'
            ', loss: {loss:.10f}'
            ', fine loss: {lossF:.4f}'
            ', cen loss: {lossC:.4f}'
            ', den loss: {lossD:.4f}').format(i=i, epoch=epoch,
                                              loss=loss.mean(), lossF=dense_loss.mean(), lossC=sparse_loss.mean(), lossD=0)
    if args.log and i == 0:
        log.info(text)
    print(text)


def validate_cat(current_epoch):
    label_map = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']
    cds = []
    for label in label_map:
        val_dset_cats = PCN_pcd('../dataset/ShapeNetCompletion', 'test', cat=label)
        val_loader_cats = DataLoader(val_dset_cats, batch_size=args.val_batch_size, pin_memory=True, shuffle=False, num_workers=8)

        all_cd = []

        all_fine_cd = []

        total = 0
        for i, batch in enumerate(val_loader_cats):
            with torch.no_grad():
                _, partial, gt, _ = batch
                partial = partial.float().cuda()
                gt = gt.float().cuda()
                # pretrined.eval()
                # x, global_features, coor, coarse = pretrined.module.get_coarse(partial, training=False)
                # pred = (x, coor, coarse)

                model.eval()
                # t = torch.randint(1, args.num_steps, (ref.size(0),)).cuda()
                ret = model(partial)
                loss, sparse_loss, dense_loss, sparse_penalty, dense_penalty = model.module.get_loss(ret, gt)

                all_cd.append(sparse_loss.mean().item() * gt.size(0))

                all_fine_cd.append(dense_loss.mean().item() * gt.size(0))

                total += gt.size(0)

                rand_iter = random.randint(0, len(val_loader_cats) - 1)
                if rand_iter == i:
                    index = random.randint(0, gt.shape[0] - 1)
                    plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(current_epoch - 1)),
                                      [partial[index].detach().cpu().numpy(),
                                       ret[0][index].detach().cpu().numpy(),
                                       ret[1][index].detach().cpu().numpy(),
                                       gt[index].detach().cpu().numpy()
                                       ],
                                      ['Input', 'Coarse', 'Fine', 'Ground Truth'], xlim=(-0.35, 0.35),
                                      ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
                print("evaluating model: {i}".format(i=i))
                # log.info("evaluating model: {i}, CD={cd}".format(i=i, cd=cd))
        mean_cd = 1000 * sum(all_cd) / total
        mean_fcd = 1000 * sum(all_fine_cd) / total
        info = "{label}: Mean Center CD L1: {mCD:.6f}\nMean Fine CD L1: {mfCD:.6f}\n".format(mCD=mean_cd,mfCD=mean_fcd,label=label)
        print(info)
        log.info(info)
        cds.append(mean_fcd)
    mean_cd_cats = sum(cds) / len(cds)
    info = "Overall CD: {cd}.\n".format(cd=mean_cd_cats)
    print(info)
    log.info(info)

def early_stop_validate(current_epoch):
    global BEST_RESULTS, BEST_EPOCH
    all_cd = []

    all_fine_cd = []

    total = 0
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            _, partial, gt, _ = batch
            partial = partial.float().cuda()
            gt = gt.float().cuda()
            # pretrined.eval()
            # x, global_features, coor, coarse = pretrined.module.get_coarse(partial, training=False)
            # pred = (x, coor, coarse)

            model.eval()
            # t = torch.randint(1, args.num_steps, (ref.size(0),)).cuda()
            ret = model(partial)
            loss, sparse_loss, dense_loss, sparse_penalty, dense_penalty = model.module.get_loss(ret, gt)

            all_cd.append(sparse_loss.mean().item() * gt.size(0))


            all_fine_cd.append(dense_loss.mean().item() * gt.size(0))

            total += gt.size(0)

            rand_iter = random.randint(0, len(val_loader) - 1)
            if rand_iter == i:
                index = random.randint(0, gt.shape[0] - 1)
                plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(current_epoch - 1)),
                                  [partial[index].detach().cpu().numpy(),
                                   ret[0][index].detach().cpu().numpy(),
                                   ret[1][index].detach().cpu().numpy(),
                                   ret[2][index].detach().cpu().numpy(),
                                   gt[index].detach().cpu().numpy()
                                   ],
                                  ['Input', 'Coarse', 'Fine L1', 'Fine L2', 'Ground Truth'], xlim=(-0.35, 0.35),
                                  ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            print("evaluating model: {i}".format(i=i))
            # log.info("evaluating model: {i}, CD={cd}".format(i=i, cd=cd))
    mean_cd = 1000 * sum(all_cd) / total
    mean_fcd = 1000 * sum(all_fine_cd) / total
    info = "Epoch: {epoch}\nMean Center CD L1: {mCD:.6f}\nMean Fine CD L1: {mfCD:.6f}\n".format(mCD=mean_cd,mfCD=mean_fcd,epoch=current_epoch - 1)
    print(info)
    log.info(info)
    if mean_cd < BEST_RESULTS:
        BEST_RESULTS = mean_cd
        BEST_EPOCH = current_epoch
        save_model = {'args': args, 'model': model.state_dict()}
        if args.log:
            torch.save(save_model, os.path.join(save_dir, 'best.pt'))

def vis_patch(current_epoch):
    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch")):
        with torch.no_grad():
            _, partial, gt, _ = batch
            partial = partial.float().cuda()
            gt = gt.float().cuda()
            model.eval()
            coarse, c_hr, lr, recons = model.module.completion(partial, gt)

            rand_iter = random.randint(0, len(val_loader) - 1)
            if rand_iter == i:
                index = random.randint(0, gt.shape[0] - 1)
                plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(current_epoch - 1)),
                                  [partial[index].detach().cpu().numpy(),
                                   c_hr[index].detach().cpu().numpy(),
                                   coarse[index].detach().cpu().numpy(),
                                   lr[index].detach().cpu().numpy(),
                                   recons[index].detach().cpu().numpy(),
                                   gt[index].detach().cpu().numpy()
                                   ],
                                  ['Partial', 'Coarse', '512 points', '2K', 'Fine', 'GT'], xlim=(-0.35, 0.35),
                                  ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))

def validate():
    # all_refs = []
    # all_recons = []
    for i, batch in enumerate(val_loader):
        print('sampling model {i}'.format(i=i))
        _, inputs, gt, _ = batch
        ref = gt.float().cuda()
        if i > 200:
            break
        with torch.no_grad():
            model.eval()
            lr = model.module.get_lr(ref)
            x_vis, z_masked, mask, center, vis_pc, msk_pc = model.module.encoder.encode(ref, masked=False)
            recons = model.module.decoder.sampling(x_vis, mask, center)
            # full = torch.cat([vis_pc, recons], dim=1)
            plot_pcd_one_view(os.path.join(evaluate_dir, 'sample_{:03d}.png'.format(i)),
                              [lr[0].detach().cpu().numpy(),
                               recons[0].detach().cpu().numpy(),
                               # full[0].detach().cpu().numpy(),
                               ref[0].detach().cpu().numpy()],
                              ['LR', 'Output', 'Ground Truth'], xlim=(-0.35, 0.35),
                              ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            pts = full.detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud(os.path.join(evaluate_dir, 'sample_{:03d}.pcd'.format(i)), pcd)
        # all_recons.append(recons)
    if args.early_stop:
        print("Best: {best}, at epoch: {epoch}".format(best=BEST_RESULTS, epoch=BEST_EPOCH - 1))
        log.info("Best: {best}, at epoch: {epoch}".format(best=BEST_RESULTS, epoch=BEST_EPOCH - 1))

def forward_diff():
    all_recons = []
    for i, batch in enumerate(val_loader):
        ref = batch['lr'].to(args.device)
        if i > 200:
            break
        with torch.no_grad():
            model.eval()
            if i == 100:
                print('sampling model {i}'.format(i=i))
                x_vis, z_masked, mask, center, vis_pc, msk_pc = model.module.encoder.encode(ref, masked=False)

                for idx in range(1, args.num_steps, 1):
                    print('sampling step {idx}'.format(idx=idx))
                    t = torch.Tensor([idx]).type(torch.int64)
                    x_noisze, _ = model.module.forward_diffusion(msk_pc, t)
                    recons = _.reshape(1, -1, 3)

                    all_recons.append(torch.cat([recons, vis_pc.reshape(1, -1, 3)], dim=1))

    all_recons = torch.cat(all_recons, dim=0)
    np.save(os.path.join(save_dir, 'out.npy'), all_recons.cpu().numpy())


try:
    n_it = 350
    epoch = 1
    while epoch <= n_it:
        # early_stop_validate(2)
        model.train()
        for i, pc in enumerate(trn_loader):
            train(i, pc, epoch)
        # for s in scheduler:
        #     s.step()
        scheduler.step()
        if epoch == n_it:
            if args.log:
                saved_file = {'args': args, 'model': model.state_dict()}
                torch.save(saved_file, os.path.join(save_dir, 'autoencoder_diff.pt'))
            # validate()
        epoch += 1
    mailman(args.exp_name, log_file)
except Exception as e:
    log.error(e)
    print(e)

