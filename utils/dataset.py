import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm
import open3d as o3d
import pickle
import random
import transforms3d
import math

class ShapeNet(Dataset):
    def __init__(self, data_path, pc_path, n_points, subset, downsample=True):
        self.data_root = data_path
        self.pc_path = pc_path
        self.subset = subset
        self.npoints = 8192
        self.downsample = downsample

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = n_points
        self.whole = False
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })

        self.permutation = np.arange(self.sample_points_num)

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def uniform_down_sample(self, pts, down_sampling_nums):
        down_sampling_nums = self.npoints // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_nums)
        return np.asarray(ds_pcd.points)

    def geometric_down_sample(self, pts, down_sampling_nums):
        def vector_angle(x, y):
            Lx = np.sqrt(x.dot(x))
            Ly = (np.sum(y ** 2, axis=1)) ** 0.5
            cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
            angle = np.arccos(cos_angle)
            angle2 = angle * 360 / 2 / np.pi
            return angle2

        knn_num = 10
        angle_thre = 30
        down_sampling_ratio = self.npoints // down_sampling_nums

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        point_size = pts.shape[0]
        tree = o3d.geometry.KDTreeFlann(pcd)
        o3d.geometry.PointCloud.estimate_normals(
            pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))
        normal = np.asarray(pcd.normals)
        normal_angle = np.zeros((point_size))
        for i in range(point_size):
            [_, idx, dis] = tree.search_knn_vector_3d(pts[i], knn_num + 1)
            current_normal = normal[i]
            knn_normal = normal[idx[1:]]
            normal_angle[i] = np.mean(vector_angle(current_normal, knn_normal))

        high_condition = normal_angle >= angle_thre
        point_high = pts[np.where(high_condition)]
        point_low = pts[np.where(~high_condition)]

        pcd_high = o3d.geometry.PointCloud()
        pcd_high.points = o3d.utility.Vector3dVector(point_high)
        pcd_low = o3d.geometry.PointCloud()
        pcd_low.points = o3d.utility.Vector3dVector(point_low)
        pcd_high_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_high, down_sampling_ratio)
        pcd_low_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_low, down_sampling_ratio)
        pcd_finl = o3d.geometry.PointCloud()
        pcd_finl.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.points),
                                                                     np.asarray(pcd_low_down.points))))
        return self.random_sample(np.asarray(pcd_finl.points), down_sampling_nums)


    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        # data = self.random_sample(data, self.sample_points_num)
        if not self.downsample:
            data = self.random_sample(data, self.sample_points_num)
            data = pc_norm(data)
            data = torch.from_numpy(data).float()
            return {'lr': data}
        elif not self.npoints == self.sample_points_num and self.downsample:
            lr = self.uniform_down_sample(data, self.sample_points_num)
            lr = pc_norm(lr)
            lr = torch.from_numpy(lr).float()
            data = pc_norm(data)
            data = torch.from_numpy(data).float()
            return {'hr': data, 'lr': lr}
        else:
            data = pc_norm(data)
            data = torch.from_numpy(data).float()
            return {'hr': data}

    def __len__(self):
        return len(self.file_list)


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def uniform_down_sample(pts, down_sampling_nums):
    down_sampling_nums = 16384 // down_sampling_nums
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_nums)
    return np.asarray(ds_pcd.points)


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNet(Dataset):
    def __init__(self, root, number_pts, use_normal, cats, subset, downsampling=8192):
        self.root = root
        self.npoints = number_pts
        self.downsample = downsampling
        self.use_normals = use_normal
        self.num_category = cats
        self.process_data = True
        self.uniform = True
        split = subset
        self.subset = subset

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def uniform_down_sample(self, pts, down_sampling_nums):
        down_sampling_nums = 8192 // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_nums)
        return np.asarray(ds_pcd.points)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_norm(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        if self.downsample != self.npoints:
            point_set = self.uniform_down_sample(point_set, self.downsample)
        lr = self.uniform_down_sample(point_set, 2048)
        return {'model': point_set, 'label': label[0], 'lr': lr}


    def __getitem__(self, index):
        data = self._get_item(index)
        points = data['model']
        label = data['label']
        lr = data['lr']
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        lr = torch.from_numpy(lr).float()
        return {'model': current_points, 'label': label, 'lr': lr}


class Semantic_KITTI(Dataset):
    def __init__(self, datapath, subset, npoints=-1, npointslr=-1, ignore_index=-1, norm=False):
        self.npoints = npoints
        self.npointslr = npointslr
        self.subset = subset
        self.data_root = datapath
        self.data_list = self.get_data_list()
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.norm = norm

    def _uniform_down_sample(self, pts, down_sampling_nums):

        down_sampling_factor = len(pts) // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_factor)

        clamp = np.asarray(ds_pcd.points)
        if len(clamp) <= down_sampling_nums:
            return clamp
        clamp_delete_num = len(clamp) - down_sampling_nums
        clamp_delete_elem = np.random.choice(len(clamp), clamp_delete_num, replace=False)
        clamp = np.delete(clamp, clamp_delete_elem, axis=0)

        return clamp

    """
    Ref: https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/semantic_kitti.py
    """
    def get_data_list(self):
        split2seq = dict(
            test=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            train=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.subset, str):
            seq_list = split2seq[self.subset]
        elif isinstance(self.subset, list):
            seq_list = []
            for split in self.subset:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        if self.norm:
            coord = pc_norm(coord)
        if self.npoints > -1 and self.npointslr > -1:
            coord = self._uniform_down_sample(coord, self.npoints)
            coord_lr = self._uniform_down_sample(coord, self.npointslr)
            coord_tensor = torch.from_numpy(coord).float()
            coord_tensor_lr = torch.from_numpy(coord_lr).float()
            return dict(hr=coord_tensor, lr=coord_tensor_lr)

        if self.npoints > -1:
            coord = self._uniform_down_sample(coord, self.npoints)
        coord_tensor = torch.from_numpy(coord).float()
        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    def __getitem__(self, item):
        data_dict = self.get_data(item)
        return data_dict

    def __len__(self):
        return len(self.data_list)


class KITTI_Object(Dataset):
    def __init__(self, datapath, subset, npoints=-1, npointslr=-1, ignore_index=-1, norm=False):
        self.npoints = npoints
        self.npointslr = npointslr
        self.subset = subset
        self.data_root = datapath
        self.data_list = self.get_data_list()
        self.ignore_index = ignore_index
        # self.learning_map = self.get_learning_map(ignore_index)
        self.norm = norm

    def _uniform_down_sample(self, pts, down_sampling_nums):

        down_sampling_factor = len(pts) // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_factor)

        clamp = np.asarray(ds_pcd.points)
        if len(clamp) <= down_sampling_nums:
            return clamp
        clamp_delete_num = len(clamp) - down_sampling_nums
        clamp_delete_elem = np.random.choice(len(clamp), clamp_delete_num, replace=False)
        clamp = np.delete(clamp, clamp_delete_elem, axis=0)

        return clamp

    """
    Ref: https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/semantic_kitti.py
    """
    def get_data_list(self):
        if self.subset == 'train':
            seq_folder = os.path.join(self.data_root, 'training')
        else:
            seq_folder = os.path.join(self.data_root, 'testing')
        data_list = []
        seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
        data_list += [
            os.path.join(seq_folder, "velodyne", file) for file in seq_files
        ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]

        if self.norm:
            coord = pc_norm(coord)
        if self.npoints > -1 and self.npointslr > -1:
            coord = self._uniform_down_sample(coord, self.npoints)
            coord_lr = self._uniform_down_sample(coord, self.npointslr)
            coord_tensor = torch.from_numpy(coord).float()
            coord_tensor_lr = torch.from_numpy(coord_lr).float()
            return dict(hr=coord_tensor, lr=coord_tensor_lr)

        if self.npoints > -1:
            coord = self._uniform_down_sample(coord, self.npoints)
        coord_tensor = torch.from_numpy(coord).float()
        data_dict = dict(hr=coord_tensor)
        return data_dict

    def __getitem__(self, item):
        data_dict = self.get_data(item)
        return data_dict

    def __len__(self):
        return len(self.data_list)


class ScanObjectNN(Dataset):
    """
    Ref: https://github.com/qizekun/ShapeLLM/blob/main/ReConV2/datasets/ScanObjectNNDataset.py
    """
    def __init__(self, data_path, subset):
        super().__init__()
        self.subset = subset
        self.root = data_path

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        current_points = pc_norm(current_points)
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]

        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]


class PCN_pcd(Dataset):
    def __init__(self, path, prefix="train", cat='all'):
        if prefix == "train":
            self.file_path = os.path.join(path, 'train')
        elif prefix == "val":
            self.file_path = os.path.join(path, 'val')
        elif prefix == "test":
            self.file_path = os.path.join(path, 'test')
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.label_map = {'02691156': '0', '02933112': '1', '02958343': '2',
                          '03001627': '3', '03636649': '4', '04256520': '5',
                          '04379243': '6', '04530566': '7', 'all': '8'}

        self.label_map_inverse = {'0': '02691156', '1': '02933112', '2': '02958343',
                                  '3': '03001627', '4': '03636649', '5': '04256520',
                                  '6': '04379243', '7': '04530566', '8': 'all'}

        self.cat = cat

        self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        random.shuffle(self.input_data)

        self.len = len(self.input_data)

        self.scale = 0
        self.mirror = True
        self.rot = 0
        self.sample = 1


    def __len__(self):
        return self.len

    def read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        return points

    def get_data(self, path):
        cls = os.listdir(path)
        data = []
        labels = []
        for c in cls:
            if c == self.cat or self.cat == 'all':
                objs = os.listdir(os.path.join(path, c))
                for obj in objs:
                    f_names = os.listdir(os.path.join(path, c, obj))
                    obj_list = []
                    for f_name in f_names:
                        data_path = os.path.join(path, c, obj, f_name)
                        obj_list.append(data_path)
                        # points = self.read_pcd(os.path.join(path, c, obj, f_name))
                    data.append(obj_list)
                    labels.append(self.label_map[c])

        return data, labels

    def randomsample(self, ptcloud, n_points):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:n_points]]

        if ptcloud.shape[0] < n_points:
            zeros = np.zeros((n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])
        return ptcloud

    def upsample(self, ptcloud, n_points):
        curr = ptcloud.shape[0]
        need = n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            # ptcloud = np.concatenate([ptcloud,np.zeros_like(ptcloud)],dim=0)
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud

    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        if self.mirror and self.prefix == 'train':

            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:  # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * self.scale
            result.append(ptcloud)

        return result[0], result[1]

    def __getitem__(self, index):

        partial_path = self.input_data[index]
        n_sample = len(partial_path)
        idx = random.randint(0, n_sample - 1)
        partial_path = partial_path[idx]

        partial = self.read_pcd(partial_path)

        # if self.prefix == 'train' and self.sample:
        partial = self.upsample(partial, 2048)

        gt_path = partial_path.replace('/' + partial_path.split('/')[-1], '.pcd')
        gt_path = gt_path.replace('partial', 'complete')

        if self.prefix == 'train':
            complete = self.read_pcd(gt_path)
            partial, complete = self.get_transform([partial, complete])
        else:
            complete = self.read_pcd(gt_path)

        complete = torch.from_numpy(complete)
        partial = torch.from_numpy(partial)
        label = partial_path.split('/')[-3]
        # label = self.label_map_inverse[label]
        obj = partial_path.split('/')[-2]

        if self.prefix == 'test':
            return label, partial, complete, obj
        else:
            return label, partial, complete

class PCN_vis(Dataset):
    def __init__(self, path, prefix="train"):
        if prefix == "train":
            self.file_path = os.path.join(path, 'train')
        elif prefix == "val":
            self.file_path = os.path.join(path, 'val')
        elif prefix == "test":
            self.file_path = os.path.join(path, 'test')
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.label_map = {'02691156': '0', '02933112': '1', '02958343': '2',
                          '03001627': '3', '03636649': '4', '04256520': '5',
                          '04379243': '6', '04530566': '7', 'all': '8'}

        self.label_map_inverse = {'0': '02691156', '1': '02933112', '2': '02958343',
                                  '3': '03001627', '4': '03636649', '5': '04256520',
                                  '6': '04379243', '7': '04530566', '8': 'all'}

        self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        random.shuffle(self.input_data)

        self.len = len(self.input_data)

        self.scale = 0
        self.mirror = 1
        self.rot = 0
        self.sample = 1

    def __len__(self):
        return self.len

    def read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        return points

    def get_data(self, path):
        cls = os.listdir(path)
        data = []
        labels = []
        for c in cls:
            if c != '.DS_Store':
                objs = os.listdir(os.path.join(path, c))
                for obj in objs:
                    f_names = os.listdir(os.path.join(path, c, obj))
                    obj_list = []
                    for f_name in f_names:
                        data_path = os.path.join(path, c, obj, f_name)
                        obj_list.append(data_path)
                        # points = self.read_pcd(os.path.join(path, c, obj, f_name))
                    data.append(obj_list)
                    labels.append(self.label_map[c])

        return data, labels

    def randomsample(self, ptcloud, n_points):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:n_points]]

        if ptcloud.shape[0] < n_points:
            zeros = np.zeros((n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])
        return ptcloud

    def upsample(self, ptcloud, n_points):
        curr = ptcloud.shape[0]
        need = n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            # ptcloud = np.concatenate([ptcloud,np.zeros_like(ptcloud)],dim=0)
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud

    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)

        if self.mirror and self.prefix == 'train':
            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:  # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * self.scale
            result.append(ptcloud)

        return result[0], result[1]

    def __getitem__(self, index):

        partial_path = self.input_data[index]
        n_sample = len(partial_path)
        partials = []
        for i in range(0, n_sample):
            partial = self.read_pcd(partial_path[i])

            # if self.prefix == 'train' and self.sample:
            partial = self.upsample(partial, 2048)
            partials.append(partial)
        label = partial_path[0].split('/')[-3]
        obj = partial_path[0].split('/')[-2]
        gt_path = partial_path[0].replace('/' + partial_path[0].split('/')[-1], '.pcd')
        gt_path = gt_path.replace('partial', 'complete')
        complete = self.read_pcd(gt_path)
        return complete, partials, label, obj

def o3d_visualize_pc(pc):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([point_cloud])


def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30  # 水平倾斜
    azim = -45  # 旋转
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


class ShapeNetCompletion(Dataset):
    def __init__(self, data_path, pc_path, n_points, subset):
        self.data_root = data_path
        self.pc_path = pc_path
        self.subset = subset
        self.npoints = n_points
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def uniform_down_sample(self, pts, down_sampling_nums):
        down_sampling_nums = 8192 // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_nums)
        return np.asarray(ds_pcd.points)

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        if self.npoints < 8192:
            data = self.uniform_down_sample(data, self.npoints)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()

        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)


cat_name = {0: 'airplane',
            1: 'cabinet',
            2: 'car',
            3: 'chair',
            4: 'lamp',
            5: 'sofa',
            6: 'table',
            7: 'watercraft',
            8: 'bed',
            9: 'bench',
            10: 'bookshelf',
            11: 'bus',
            12: 'guitar',
            13: 'motorbike',
            14: 'pistol',
            15: 'skateboard'}

cat_taxonomy_id = {0: '02691156',
                   1: '02933112',
                   2: '02958343',
                   3: '03001627',
                   4: '03636649',
                   5: '04256520',
                   6: '04379243',
                   7: '04530566',
                   8: '02818832',
                   9: '02828884',
                   10: '02871439',
                   11: '02924116',
                   12: '03467517',
                   13: '03790512',
                   14: '03948459',
                   15: '04225987'}

class MVP(Dataset):
    def __init__(self, config):
        self.aug = config.AUG
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.novel_input_only = False
        self.novel_input = True
        self.input_path = config.PARTIAL_POINTS_PATH % self.subset
        self.gt_path = config.COMPLETE_POINTS_PATH % (self.subset, self.npoints)

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if self.novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif self.novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        print('######## MVP dataset ########')
        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)
        print('#############################')
        self.len = self.input_data.shape[0]

        self.pc_augm_scale = 1.5
        self.pc_augm_rot = 1  # 1
        self.pc_augm_mirror_prob = 0.5  # 0.5
        self.pc_augm_jitter = 0

    def augment_cloud(self, Ps):
        """" Augmentation on XYZ and jittering of everything """
        M = transforms3d.zooms.zfdir2mat(1)
        if self.pc_augm_scale > 1:
            #            print('scale')
            s = random.uniform(1 / self.pc_augm_scale, self.pc_augm_scale)
            M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
        if self.pc_augm_rot:
            #            print('rot')
            if random.random() < self.pc_augm_rot / 2:
                angle = random.uniform(0, 2 * math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)  # y=upright assumption
            if random.random() < self.pc_augm_rot / 2:
                angle = random.uniform(0, 2 * math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([1, 0, 0], angle), M)  # y=upright assumption
            if random.random() < self.pc_augm_rot / 2:
                angle = random.uniform(0, 2 * math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # y=upright assumption
        if self.pc_augm_mirror_prob > 0:  # mirroring x&z, not y
            #            print('mirror')
            if random.random() < self.pc_augm_mirror_prob / 2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
            if random.random() < self.pc_augm_mirror_prob / 2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), M)
        result = []
        for P in Ps:
            P[:, :3] = np.dot(P[:, :3], M.T)

            if self.pc_augm_jitter:
                sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
                P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
            result.append(P)
        return result[0], result[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = self.input_data[index]
        complete = self.gt_data[index // 26]
        if self.subset == 'train' and self.aug:
            partial, complete = self.augment_cloud([partial, complete])
        partial = torch.from_numpy(partial)
        complete = torch.from_numpy(complete)
        taxonomy_id = cat_taxonomy_id[int(self.labels[index])]
        model_id = cat_taxonomy_id[int(self.labels[index])]
        return taxonomy_id, model_id, (partial, complete)

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    #
    # import open3d as o3d
    # import matplotlib.pyplot as plt
    # draw = PCN_vis(
    #     '../datasample', 'test'
    # )
    # val_loader = DataLoader(draw, batch_size=1, pin_memory=True, shuffle=False)
    # for i, b in enumerate(val_loader):
    #     gt, partial, label, obj = b
    #     label = label[0]
    #     obj = obj[0]
    #     print('{l}/{i}'.format(l=label, i=obj))
    #
    #     plot_pcd_one_view(os.path.join('../datasample/vis', '{l}/{i}.png'.format(l=label, i=obj)),
    #                       [
    #                           gt[0].detach().cpu().numpy(),
    #                        partial[0][0].detach().cpu().numpy(),
    #                        partial[1][0].detach().cpu().numpy(),
    #                        partial[2][0].detach().cpu().numpy(),
    #                        partial[3][0].detach().cpu().numpy(),
    #                        partial[4][0].detach().cpu().numpy(),
    #                        partial[5][0].detach().cpu().numpy(),
    #                        partial[6][0].detach().cpu().numpy(),
    #                        partial[7][0].detach().cpu().numpy()
    #                        ],
    #                       ['GT', 'Partial 0', 'Partial 1', 'Partial 2', 'Partial 3', 'Partial 4', 'Partial 5', 'Partial 6', 'Partial 7'], xlim=(-0.35, 0.35),
    #                       ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))

    # train_dset = ShapeNet(
    #     data_path='../dataset/ShapeNet55/ShapeNet-55',
    #     pc_path='../dataset/ShapeNet55/shapenet_pc',
    #     subset='train',
    #     n_points=8192,
    #     downsample=True
    # )
    # val_dset = ShapeNet(
    #     data_path='../dataset/ShapeNet55/ShapeNet-55',
    #     pc_path='../dataset/ShapeNet55/shapenet_pc',
    #     subset='test',
    #     n_points=8192,
    #     downsample=True
    # )
    # val_loader = DataLoader(val_dset, batch_size=1, pin_memory=True)
    # trn_loader = DataLoader(train_dset, batch_size=1, pin_memory=True)
    # train_2k_set = []
    # train_8k_set = []
    # val_2k_set = []

    # val_8k_set = []
    # for i, batch in enumerate(tqdm(trn_loader, desc='Training set')):
    #     train_8k_set.append(batch['hr'].cpu().numpy())
    # for i, batch in enumerate(tqdm(val_loader, desc='Validate set')):
    #     val_8k_set.append(batch['hr'].cpu().numpy())

    # print(train_dset.__getitem__(0)['hr'].shape)
    # test_KITTI = Semantic_KITTI(
    #     datapath='../dataset/SEMANTIC_KITTI_DIR',
    #     subset='train',
    #     norm=True,
    #     npoints=8192,
    #     npointslr=2048
    # )
    # # print(test_KITTI.__getitem__(0)['coord'])
    # print(len(test_KITTI.__getitem__(0)['lr']))
    # testK_loader = DataLoader(test_KITTI, batch_size=1, pin_memory=True)

