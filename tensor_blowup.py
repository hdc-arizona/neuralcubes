import torch
import random
import itertools
import json
import numpy as np

class TensorBlowup_YellowCab:
    def __init__(self, device='cpu', schema=None):
        self.d1 = 12
        self.d2 = 7
        self.d3 = 24
        self.d4_x = 20
        self.d4_y = 20

        self.dims = [self.d1, self.d2, self.d3, self.d4_x, self.d4_y]
        self.row_counts = self.d1+self.d2+self.d3+self.d4_x*self.d4_y
        self.col_counts = sum(self.dims)

        self.device = torch.device(device)

        self.template = self.createTemplateTensor()
        self.mask = self.createMaskTensor()

    def createSourceTensor(self, vec):
        # t = torch.zeros( (d1+d2+d3+d4_x+d4_y, ), dtype=torch.uint8, device=device)
        t = torch.zeros( (self.col_counts, ), dtype=torch.uint8, device=self.device)

        ranges = []
        for i in range(0, len(vec), 2):
            ranges.append(vec[i:i+2])

        offset = 0
        for r, dim in zip(ranges, self.dims):
            t[offset+r[0]:offset+r[1]] = 1
            offset += dim

        # t = t.repeat((d1+d2+d3+d4_x*d4_y, 1))
        t = t.repeat((self.row_counts, 1))
        return t

    # create template
    def createTemplateTensor(self):
        # template = torch.zeros( (d1+d2+d3+d4_x*d4_y, d1+d2+d3+d4_x+d4_y), dtype=torch.uint8, device=device)
        template = torch.zeros( (self.row_counts, self.col_counts), dtype=torch.uint8, device=self.device)
        for i in range(self.d1+self.d2+self.d3):
            template[i][i] = 1

        offset = self.d1+self.d2+self.d3
        row = self.d1+self.d2+self.d3
        for r in range(self.d4_x):
            for c in range(self.d4_y):
                template[row][offset+r] = 1
                template[row][offset+self.d4_y+c] = 1
                row += 1
        return template

    # create mask
    def createMaskTensor(self):
        mask = torch.ones( (self.row_counts, self.col_counts), dtype=torch.uint8, device=self.device)

        offset_r, offset_c = 0, 0
        for r, c in zip([self.d1, self.d2, self.d3, self.d4_x*self.d4_y], [self.d1, self.d2, self.d3, self.d4_x+self.d4_y]):
            mask[offset_r:r+offset_r, offset_c:c+offset_c] = 0
            offset_r += r
            offset_c += c
        return mask

    def createWeights(self, n=0):
        weights = torch.zeros((self.row_counts, 1))
        pre = 0
        for offset in [self.d1, self.d2, self.d3, self.d4_x*self.d4_y]:
            weights[pre:offset+pre,:] = 1/offset
            pre += offset
        return weights.repeat((n,1))

    def blowup(self, ranges):
        tensors = [ self.createSourceTensor(ranges[i])*self.mask+self.template for i in range(ranges.size()[0]) ]
        return torch.cat(tensors)

    def interval_uniformly_sample(self, d, n, min_interval_size=2,max_interval_size=None,single_interval_mask=None):
        max_interval_size = d+1 if max_interval_size is None else max_interval_size
        rand_interval_sizes = torch.randint(min_interval_size,max_interval_size,(n,),device=self.device)
        if single_interval_mask is not None:
            rand_interval_sizes[single_interval_mask]=1
        rand_start = torch.floor((d-rand_interval_sizes+(1-1e-12))*torch.rand(size=(n,),device=self.device))
        rand_end = rand_start+rand_interval_sizes-1
        return rand_start.int(),rand_end.int(),rand_interval_sizes.int()

    def sample_counter(self, n_counts):
        count_sequence = range(n_counts)
        dims = np.array(['d1','d2','d3','d4x','d4y'])
        all_pairs = np.array(list(itertools.permutations([i for i in range(0,5)],2)),dtype=np.int32)
        random_pair_inds = np.random.randint(0,high=all_pairs.shape[0],size=n_counts)
        rand_split_d = dims[all_pairs[random_pair_inds,0]]
        oh_d = dims[all_pairs[random_pair_inds,1]]

        rand_d1_start,rand_d1_end,_ = self.interval_uniformly_sample(self.d1,n_counts,single_interval_mask=torch.tensor(np.array(oh_d=='d1',dtype=np.int),dtype=torch.uint8))
        rand_d2_start,rand_d2_end,_ = self.interval_uniformly_sample(self.d2,n_counts,single_interval_mask=torch.tensor(np.array(oh_d=='d2',dtype=np.int),dtype=torch.uint8))
        rand_d3_start,rand_d3_end,_ = self.interval_uniformly_sample(self.d3,n_counts,single_interval_mask=torch.tensor(np.array(oh_d=='d3',dtype=np.int),dtype=torch.uint8))
        rand_d4x_start,rand_d4x_end,_ = self.interval_uniformly_sample(self.d4_x,n_counts,single_interval_mask=torch.tensor(np.array(oh_d=='d4x',dtype=np.int),dtype=torch.uint8))
        rand_d4y_start,rand_d4y_end,_ = self.interval_uniformly_sample(self.d4_y,n_counts,single_interval_mask=torch.tensor(np.array(oh_d=='d4y',dtype=np.int),dtype=torch.uint8))

        offsets = {'d1':0,'d2':self.d1,'d3':(self.d1+self.d2),'d4x':(self.d1+self.d2+self.d3),'d4y':(self.d1+self.d2+self.d3+self.d4_x)}
        starts = {'d1':rand_d1_start,'d2':rand_d2_start,'d3':rand_d3_start,'d4x':rand_d4x_start,'d4y':rand_d4y_start}
        ends = {'d1':rand_d1_end,'d2':rand_d2_end,'d3':rand_d3_end,'d4x':rand_d4x_end,'d4y':rand_d4y_end}

        d1_inds = torch.tensor([[sdx,d] for sdx,start,end in zip(count_sequence,rand_d1_start,rand_d1_end+1) for d in range(start,end)],dtype=torch.long)
        d2_inds = torch.tensor([[sdx,d] for sdx,start,end in zip(count_sequence,rand_d2_start,rand_d2_end+1) for d in range(start,end)],dtype=torch.long)
        d3_inds = torch.tensor([[sdx,d] for sdx,start,end in zip(count_sequence,rand_d3_start,rand_d3_end+1) for d in range(start,end)],dtype=torch.long)
        d4x_inds = torch.tensor([[sdx,d] for sdx,start,end in zip(count_sequence,rand_d4x_start,rand_d4x_end+1) for d in range(start,end)],dtype=torch.long)
        d4y_inds = torch.tensor([[sdx,d] for sdx,start,end in zip(count_sequence,rand_d4y_start,rand_d4y_end+1) for d in range(start,end)],dtype=torch.long)

        mh_union = torch.zeros(n_counts,self.col_counts,device=self.device)
        mh_union[d1_inds[:,0],d1_inds[:,1]]=1
        mh_union[d2_inds[:,0],offsets['d2']+d2_inds[:,1]]=1
        mh_union[d3_inds[:,0],offsets['d3']+d3_inds[:,1]]=1
        mh_union[d4x_inds[:,0],offsets['d4x']+d4x_inds[:,1]]=1
        mh_union[d4y_inds[:,0],offsets['d4y']+d4y_inds[:,1]]=1

        d_split_inds = torch.tensor([[offsets[split]+starts[split][sdx], offsets[split]+np.random.randint(starts[split][sdx]+1,ends[split][sdx]+1), offsets[split]+ends[split][sdx]] for sdx,split in zip(count_sequence,rand_split_d)],dtype=torch.long)
        split_1_inds = torch.tensor([[sdx,d] for sdx,split in zip(count_sequence,d_split_inds) for d in range(split[1],split[2]+1)],dtype=torch.long)
        split_2_inds = torch.tensor([[sdx,d] for sdx,split in zip(count_sequence,d_split_inds) for d in range(split[0],split[1])],dtype=torch.long)

        mh_split_1 = mh_union.clone()
        mh_split_2 = mh_union.clone()
        mh_split_1[split_1_inds[:,0],split_1_inds[:,1]]=0
        mh_split_2[split_2_inds[:,0],split_2_inds[:,1]]=0

        return mh_union,mh_split_1,mh_split_2

    def random_count_split(self, counts, n_samples):
        np_counts = counts.numpy()
        d1_off,d2_off,d3_off,d4x_off,d4y_off = 0,self.d1,self.d1+self.d2,self.d1+self.d2+self.d3,self.d1+self.d2+self.d3+self.d4_x
        ios = [[d1_off,d2_off],[d2_off,d3_off],[d3_off,d4x_off],[d4x_off,d4y_off],[d4y_off,d4y_off+self.d4_y]]

        all_inds = np.arange(np_counts.shape[0])
        np.random.shuffle(all_inds)

        all_split_1 = []
        all_split_2 = []
        sampled_inds = []
        for cdx in all_inds:
            full_count = np_counts[cdx,:]
            valid_intervals = [ios[idx] for idx in range(len(ios)) if np.sum(full_count[ios[idx][0]:ios[idx][1]])>1]
            if len(valid_intervals)==0:
                continue
            rand_interval_ind = random.randint(0,len(valid_intervals)-1)
            rand_interval = valid_intervals[rand_interval_ind]
            count_interval = full_count[rand_interval[0]:rand_interval[1]]
            arg_on = np.argwhere(count_interval==1).squeeze()
            rand_split_ind = arg_on[np.random.randint(0,arg_on.shape[0]-1)]

            split_1 = np.array(np_counts[cdx,:],dtype=np_counts.dtype)
            split_2 = np.array(np_counts[cdx,:],dtype=np_counts.dtype)
            split_1[rand_interval[0]+rand_split_ind+1:rand_interval[1]]=0
            split_2[rand_interval[0]:rand_interval[0]+rand_split_ind+1]=0

            all_split_1.append(split_1.tolist())
            all_split_2.append(split_2.tolist())
            sampled_inds.append(cdx)

            if len(sampled_inds) == n_samples:
                break

        return torch.tensor(all_split_1,dtype=torch.uint8,device=self.device),torch.tensor(all_split_2,dtype=torch.uint8,device=self.device),torch.tensor(sampled_inds,dtype=torch.long,device=self.device)

class TensorBlowup:
    def __init__(self, device='cpu', schema=None):
        self.data_schema = schema['data_schema']
        self.net_branches = schema['net_schema']['branches']

        self.dims = []
        self.row_counts = 0

        self.row_block = []
        self.col_block = []
        for branch in self.net_branches:
            if self.data_schema[branch['key']]['type'] != 'spatial':
                d = self.data_schema[branch['key']]['dimension']
                self.dims.append(d)
                self.row_counts += d

                self.row_block.append(d)
                self.col_block.append(d)
            else:
                spatial_res =self.data_schema[branch['key']]['resolution']
                self.dims += [spatial_res['x'], spatial_res['y']]
                self.row_counts += spatial_res['x'] * spatial_res['y']
                self.row_block.append(spatial_res['x'] * spatial_res['y'])
                self.col_block.append(spatial_res['x'] + spatial_res['y'])
        self.col_counts = sum(self.dims)

        self.device = torch.device(device)

        self.template = self.createTemplateTensor()
        self.mask = self.createMaskTensor()

    def createSourceTensor(self, vec):
        # t = torch.zeros( (d1+d2+d3+d4_x+d4_y, ), dtype=torch.uint8, device=device)
        t = torch.zeros( (self.col_counts, ), dtype=torch.uint8, device=self.device)

        ranges = []
        for i in range(0, len(vec), 2):
            ranges.append(vec[i:i+2])

        offset = 0
        for r, dim in zip(ranges, self.dims):
            t[offset+int(r[0]):offset+int(r[1])] = 1
            offset += dim

        # t = t.repeat((d1+d2+d3+d4_x*d4_y, 1))
        t = t.repeat((self.row_counts, 1))
        return t

    # create template
    def createTemplateTensor(self):
        # template = torch.zeros( (self.row_counts, self.col_counts), dtype=torch.uint8, device=self.device)
        # for i in range(self.d1+self.d2+self.d3):
            # template[i][i] = 1

        # offset = self.d1+self.d2+self.d3
        # row = self.d1+self.d2+self.d3
        # for r in range(self.d4_x):
            # for c in range(self.d4_y):
                # template[row][offset+r] = 1
                # template[row][offset+self.d4_y+c] = 1
                # row += 1
        # return template

        template = torch.zeros( (self.row_counts, self.col_counts), dtype=torch.uint8, device=self.device)
        col_offset = 0
        row_offset = 0
        for branch in self.net_branches:
            if self.data_schema[branch['key']]['type'] != 'spatial':
                d = self.data_schema[branch['key']]['dimension']
                for i in range(d):
                    template[row_offset+i][col_offset+i] = 1
                col_offset += d
                row_offset += d
            else:
                spatial_res =self.data_schema[branch['key']]['resolution']
                dx, dy = spatial_res['x'], spatial_res['y']
                for r in range(dx):
                    for c in range(dy):
                        template[row_offset][col_offset+r] = 1
                        template[row_offset][col_offset+dx+c] = 1
                        row_offset += 1
                col_offset += dx + dy
        return template


    # create mask
    def createMaskTensor(self):
        mask = torch.ones( (self.row_counts, self.col_counts), dtype=torch.uint8, device=self.device)

        offset_r, offset_c = 0, 0
        for r, c in zip(self.row_block, self.col_block):
            mask[offset_r:r+offset_r, offset_c:c+offset_c] = 0
            offset_r += r
            offset_c += c
        return mask

    def createWeights(self, n=0):
        weights = torch.zeros((self.row_counts, 1))
        pre = 0
        # for offset in [self.d1, self.d2, self.d3, self.d4_x*self.d4_y]:
        for offset in self.row_block:
            weights[pre:offset+pre,:] = 1/offset
            pre += offset
        return weights.repeat((n,1))

    def blowup(self, ranges):
        tensors = [ self.createSourceTensor(ranges[i])*self.mask+self.template for i in range(ranges.size()[0]) ]
        return torch.cat(tensors)

if __name__ == "__main__":

    '''
    schema = json.load(open('./yellow_cab_config/cfg_100kpm_10k.json'))

    tb = TensorBlowup_YellowCab()
    tb_new = TensorBlowup(schema=schema)

    ranges = torch.tensor([[1,4, 0,3, 1, 4, 0, 7, 2, 9]])

    truth = tb.blowup(ranges)
    new = tb_new.blowup(ranges)

    torch.set_printoptions(profile="full")

    print(truth)
    print(new)
    '''

    schema = json.load(open('./flights_config/flights_10k.json'))

    tb_new = TensorBlowup(schema=schema)


    ranges = torch.tensor([[1,10, 3,5, 7,20, 2,17, 6,7, 0,7, 0,14]])
    results = tb_new.blowup(ranges)

    def printTensor(tensor):
        for r in tensor:
            for i in range(r.shape[0]):
                if i != r.shape[0]-1:
                    print(str(r[i].numpy())+' ', end='')
                else:
                    print(str(r[i].numpy()))

    m = tb_new.createMaskTensor()
    t = tb_new.createTemplateTensor()
    # printTensor(m)
    # printTensor(t)
    printTensor(results)

    '''
    union,split_1,split_2 = tb.sample_counter(20)
    print('union:',union[:,:tb.d1])
    print('split 1:',split_1[:,:tb.d1])
    print('split 2:',split_2[:,:tb.d1])
    '''
    # s1,s2,inds = tb.random_count_split(final,final.shape[0])
    # print('inds:',inds)
    # t = tb.createTemplateTensor()
    # m = tb.createMaskTensor()
    # union,split_1,split_2 = tb.sample_counter(20)
    # print('union:',union[:,:tb.d1])
    # print('split 1:',split_1[:,:tb.d1])
    # print('split 2:',split_2[:,:tb.d1])

    '''
    # ranges = torch.tensor([[[1,2], [0,2], [1, 3], [0, 2], [2, 3]]]*1000)
    ranges = torch.tensor([[1,2, 0,2, 1, 3, 0, 2, 2, 3]]*10)
    for i in range(10):
        final = tb.blowup(ranges)
        print(final)
    # np.savetxt('blowup_tensor.txt', final.cpu().numpy(), fmt='%i', delimiter=' ')
    print(final.size())
    '''

    # w = tb.createWeights(2)
    # print(w)
    # print(w.size())
