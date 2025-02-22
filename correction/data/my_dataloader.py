import numpy as np
from torch.utils.data import DataLoader, Dataset
# import netCDF4
# import wrf


class WRFNPDataset(Dataset):
    def __init__(self, wrf_files, era_files, seq_len=4):
        super().__init__()
        print(len(wrf_files), 'wrf_files')
        print(len(era_files), 'era_files')
        wrf_files.sort()
        era_files.sort()
        self.wrf_files = wrf_files
        self.era_files = era_files
        self.processed = False
        self.seq_len = seq_len
        self.file_len = 24

    def __len__(self):
        l = len(self.wrf_files) * self.file_len - self.seq_len
        return l if l > 0 else 0
    
    def process_land_mask(self,gradient_mult):
        if(self.processed):
            return
        self.land_mask = np.load('landmask.npy')
        #for i in range (1,len(self.land_mask)-1):
        #    for j in range (1,len(self.land_mask)-1):
        #        if self.land_mask[i][j] == 0:
        #            self.land_mask[i][j] = max (self.land_mask[i-1][j],self.land_mask[i+1][j],self.land_mask[i][j+1],self.land_mask[i][j-1])/gradient_mult
        #        if (self.land_mask[i][j]==1):
        #            self.land_mask[i][j] = max (self.land_mask[i-1][j],self.land_mask[i+1][j],self.land_mask[i][j+1],self.land_mask[i][j-1])*gradient_mult
        self.processed = True
    
    def get_data_by_id(self, i, file_attr, var_attr):
        needed_len = self.seq_len
        path_i, item_i = self.get_path_id(i)
        npys = []
        while needed_len > 0:
            file_vars = np.load(getattr(self, file_attr)[path_i])

            res=[]
            self.process_land_mask(1)
            land_mask = self.land_mask
            if(len(file_vars[0][0])==210):
                for i in range (0,len(file_vars)):
                    res.append(np.vstack((file_vars[i],[land_mask])))
                file_vars = np.array(res)
                
            else:
                land_mask = [[0 for i in range(215)] for i in range(67)]
                for i in range (0,len(file_vars)):
                    res.append(np.vstack((file_vars[i],[land_mask])))
                file_vars = np.array(res)

            part = file_vars[item_i:item_i + needed_len]
            item_i = 0
            path_i += 1
            npys.append(part)
            needed_len -= len(part)
            
                

        return np.concatenate(npys)

    def get_path_id(self, i):
        path_id, item_id = i // self.file_len, i % self.file_len
        return path_id, item_id

    def __getitem__(self, i):
        data = self.get_data_by_id(i, 'wrf_files', 'wrf_variables')
        data = np.flip(data, 2).copy()
        target = self.get_data_by_id(i, 'era_files', 'era_variables')
        return data, target


def find_files(directory, pattern):
    import os, fnmatch
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\', '/')
                flist.append(filename)
    return flist


if __name__ == "__main__":
    import sys

    sys.path.insert(0, '/home/')
    from correction.data.train_test_split import split_train_val_test

    wrf_folder = ''
    era_folder = ''
    train_files, val_files, test_files = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)

    train_dataset = WRFNPDataset(train_files[0], train_files[1])
    dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=4)
    from time import time

    st = time()
    for data, target in dataloader:
        print(data.shape, target.shape)
    print(time() - st)
