import pickle, os, glob, sys, torch, random, gc, fastai
from fastai.vision.all import *
from fastai.vision.augment import _slice
import sklearn.metrics as skm
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
# from optuna.integration import FastAIPruningCallback


## Parse logger file to csv
def parse_log_to_csv(path_to_log, output_path):
    f = open(path_to_log, "r")
    csv = ""
    for line in f:
        line = line.replace("\t", " ").replace("\n", "")
        if len(line.split(" ")) != 1 and "=" not in line:
            row = [x for x in line.split(" ") if x != "" ]
            csv += ",".join(row) + "\n"
    f.close()
    
    csv_file = open(output_path, "w")
    csv_file.write(csv)
    csv_file.close()
    
## Truncate logger file to just the relevant info
def truncate_log(path_to_log, output_path):
    f = open(path_to_log, "r")
    new_log = ""
    for line in f:
        if "Epoch" not in line:
            new_log += line
    f.close()
    
    f = open(output_path, "w")
    f.write(new_log)
    f.close()
    
    
    
    for line in f:
        line = line.replace("\t", " ").replace("\n", "")
        if len(line.split(" ")) != 1 and "=" not in line:
            row = [x for x in line.split(" ") if x != "" ]
            csv += ",".join(row) + "\n"
    f.close()
    
    csv_file = open(output_path, "w")
    csv_file.write(csv)
    csv_file.close()
    

###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################


class SwarmCallback(Callback):   
    def __init__(self, sync_factor, state_dicts_dir, communication_dir, is_sync_node, model_name, subname, clinic, clinic_list, slides_per_state_dict, save_intermdiate_state_dicts=False):
        '''Callback which saves state_dict using a given sync_factor. It then waits for all other nodes to save their state_dicts.
           The node with is_sync_node==True Averages all dicts and saves bool (True) at continue_training_path to signal the other nodes.
        
        
        Paramaters
        ----------
        sync_factor : float
                      Indicates after how many batches the global_state_dict is updated based on the total number of batches.

        state_dicts_dir : str
                          directory where all state_dicts are saved
                          
        communication_dir : str
                            directory where teh progress of all nodes are saved
                          
        ready_for_sync_path : str
                              path to pickled boolean. If True a node is ready to get updated global_state_dict.
                          
        is_sync_node : bool
                       indicates whether the current node is responsible for merging the state_dicts.
                                    
        model_name : str
                     name of the model for current train run. This is used to load all the correct state_dicts.
                  
        subname : str
                  name that is unique for one train run. This is used to load all the correct state_dicts.
                  
        clinic : str
                  name of the cöinic for the current train run. This is used to load all the correct state_dicts.
                  
        clinic_list : list(str)
                      contains the names of all clinics that are involved in the current train run.
                      
        slides_per_state_dict : dict
                                a dictionary that maps each clinic to their total number of slides.
                                
        save_intermdiate_state_dicts : boolean
                                       saves state_dict before every synchronisation for every clinic.
        '''
        ### constructor variables
        store_attr()
        
        ### paths
        self.state_dict_path = f"{self.state_dicts_dir}{self.model_name}_{self.clinic}_{self.subname}_state_dict.pkl"
        self.global_state_dict_path = f"{self.state_dicts_dir}{self.model_name}_{self.subname}_global_state_dict.pkl"
        self.communication_path = f"{self.communication_dir}{self.model_name}_{self.clinic}_{self.subname}_progress.p"
        
        ### misc varibales
        self.sync_interval_counter = 0
        
        
        
        @patch
        def _set_progress_to_finished(self:Learner, communication_path=self.communication_path):
            old_path = glob.glob(communication_path.replace(".p", "_*"))[0]
            pr = old_path.split("_")[-1].replace(".p", "")
            pr = int(pr)

            new_path = glob.glob(communication_path.replace(".p", "_*"))[0]
            new_path = new_path.replace(f"_{pr}.p", f"_-100.p")
            
            os.system(f'mv {old_path} {new_path}')
            
        
        
    def fedavg(self, clinic_state_dict_paths:list):
        """ This function has aggregation method 'mean' """

        global_state_dict = torch.load(self.global_state_dict_path, map_location="cpu")

        clinic_state_dicts = []
        slides_per_clinic = []
        for i in clinic_state_dict_paths:
            clinic_state_dicts.append(torch.load(i, map_location="cpu"))
            slides_per_clinic.append(float(self.slides_per_state_dict[i])/sum(self.slides_per_state_dict.values()))
        ### This will take simple mean of the weights of models ###
        for k in global_state_dict.keys():
            global_state_dict[k] = torch.stack([clinic_state_dicts[i][k].float()*slides_per_clinic[i] for i in range(len(clinic_state_dicts))], 0).sum(0)

        torch.save(global_state_dict, self.global_state_dict_path)
        

        del global_state_dict
        del clinic_state_dicts    
        del slides_per_clinic
        gc.collect()
        
    
    def get_sync_list(self, a, n):
        n =  int(n)
        l = []
        k, m = divmod(len(a), n)
        for i in range(n):
            l.append(a[i*k+min(i, m):(i+1)*k+min(i+1, m)])
        return l
    
    def _update_sync_batch(self):
        # print("sync_list_counter : ", self.sync_list_counter)
        self.sync_batch += len(self.sync_list[self.sync_list_counter])
        # print("self.sync_batch : ", self.sync_batch)
    
        
        
    def _update_progress(self):
        pr = glob.glob(self.communication_path.replace(".p", "_*"))[0].split("_")[-1].replace(".p", "")
        pr = int(pr)
        new_pr = pr + 1
        
        old_path = glob.glob(self.communication_path.replace(".p", "_*"))[0]
        
        new_path = glob.glob(self.communication_path.replace(".p", "_*"))[0].replace(f"_{pr}.p", f"_{new_pr}.p")
        
        os.system(f'mv {old_path} {new_path}')
        
        
        
    def _get_progress(self, alternative_path=None):    
        if alternative_path == None:
            pr = glob.glob(self.communication_path.replace(".p", "_*"))[0].split("_")[-1].replace(".p", "")
            pr = int(pr)
            return pr
        else:
            pr = glob.glob(alternative_path.replace(".p", "_*"))[0].split("_")[-1].replace(".p", "")
            pr = int(pr)
            return pr
        
        
    def _get_progress_from_other_clinics(self):
        progress_all_clinics = []
        for c in self.clinic_list:
            if c != self.clinic:
                progress_all_clinics.append(self._get_progress(f"{self.communication_dir}{self.model_name}_{c}_{self.subname}_progress.p"))
        return progress_all_clinics 
    
                
                
    def _create_misc_files(self):
        if self.is_sync_node == True:
            
            ### create communication file for every clinic
            for c in self.clinic_list:
                pickle.dump(0, open(f"{self.communication_dir}{self.model_name}_{c}_{self.subname}_progress_0.p", "wb"))
            
            ### create global state dict
            os.system(f"cp {self.state_dicts_dir}{self.model_name}_main_state_dict.pkl {self.global_state_dict_path}")
           
        
        ### all other nodes wait for sync_node to get everything set up
        while True:          
            fail_check = True
            if os.path.isfile(self.global_state_dict_path):
                if self._get_progress() == 0:
                    while fail_check:
                        try:
                            time.sleep(2)
                            tmp_torch_model = torch.load(self.global_state_dict_path)
                            self.learn.model.load_state_dict(tmp_torch_model)
                            tmp_torch_model = None
                            time.sleep(2)
                            fail_check = False
                        except():
                            time.sleep(2)
                    self.learn.model.cuda()
                    break
            else:
                time.sleep(2)
        
    
    def after_create(self):
        self._create_misc_files()
        self.sync_list = self.get_sync_list(list(range(len(self.learn.dls.train))), 1/self.sync_factor)
        
        ### counter variables
        self.current_batch = 0
        self.sync_batch = 0
        self.sync_list_counter = 0

        self._update_sync_batch()
        
    def _update_global_state_dict(self):
        ### check if all nodes are ready
        while True:
            progress = self._get_progress_from_other_clinics()
            progress = [x for x in progress if x != -100]
            my_progress = self._get_progress()
            # print(f"check if all nodes are ready:\n\tclinic = {self.clinic}\n\tmy_progress = {my_progress}\n\tprogress = {progress}")
            # print(self.clinic, my_progress, progress)
            if all([True if p == my_progress+1 else False for p in progress]) or len(progress) == 0:
                ### merge state_dicts
                clinic_state_dict_paths = glob.glob(f"{self.state_dicts_dir}{self.model_name}_*_{self.subname}_state_dict.pkl")
                self.fedavg(clinic_state_dict_paths)
                os.system(f"cp {self.global_state_dict_path} {self.global_state_dict_path.replace('.pkl', f'_{self.sync_interval_counter}.pkl')}")
                break
            else:
                time.sleep(2)

        
        
    def _distribute_new_global_state_dict(self):
        ### check for continue_train_path every 10 seconds for updated weights
        while True:
            progress = self._get_progress_from_other_clinics()
            progress = [x for x in progress if x != -100]
            my_progress = self._get_progress()
            if all([True if p >= my_progress else False for p in progress]):
                print("-------------------- UPDATING WEIGHTS --------------------")
                if self.save_intermdiate_state_dicts:
                    os.system(f"cp {self.state_dict_path} {self.state_dict_path.replace('.pkl', f'_{self.sync_interval_counter}.pkl')}")
                # print("old weight: ", self.learn.model.state_dict()["1.2.running_var"][0])
                self.learn.model.load_state_dict(torch.load(self.global_state_dict_path))
                self.learn.model.cuda()
                # print("new weight: ", self.learn.model.state_dict()["1.2.running_var"][0])
                self.sync_interval_counter += 1
                break
            else:
                time.sleep(2)
        
    def _start_distribution_pahse(self):
        ### only exchange weights during training and if sync interval is reached
        if self.learn.training == True and self.current_batch == self.sync_batch:
            # print("current_batch", self.current_batch)
            # print("sync_batch", self.sync_batch)
            self.sync_list_counter += 1
            if self.sync_list_counter < len(self.sync_list):
                self._update_sync_batch()

            ### save state_dict
            torch.save(self.learn.model.state_dict(), self.state_dict_path)
            
            ### update global_state_dict
            if self.is_sync_node == True:
                self._update_global_state_dict()
                
            self._update_progress()
            
            self._distribute_new_global_state_dict()
        
        
    def before_batch(self):
        self._start_distribution_pahse()
            
    def after_batch(self):
        self.current_batch += 1
        
    def before_epoch(self):
        ### counter variables
        self.current_batch = 0
        self.sync_batch = 0
        self.sync_list_counter = 0
        
        self._update_sync_batch()
        
    def after_train(self):
        # print("Hier bin ich in after_epoch reingekommen!")
        self._start_distribution_pahse()
        
    
                
class MyRandomErasing(RandTransform):
    "Randomly selects a rectangle region in an image and randomizes its pixels."
    order = 100 # After Normalize
    def __init__(self, p=0.5, sl=0., sh=0.3, min_aspect=0.3, max_count=1, min_count=1):
        store_attr()
        super().__init__(p=p)
        self.log_ratio = (math.log(min_aspect), math.log(1/min_aspect))

    def _bounds(self, area, img_h, img_w):
        r_area = random.uniform(self.sl,self.sh) * area
        aspect = math.exp(random.uniform(*self.log_ratio))
        return _slice(r_area*aspect, img_h) + _slice(r_area/aspect, img_w)

    def encodes(self,x:TensorImage):
        count = random.randint(self.min_count, self.max_count)
        _,img_h,img_w = x.shape[-3:]
        area = img_h*img_w/count
        areas = [self._bounds(area, img_h, img_w) for _ in range(count)]
        return cutout_gaussian(x, areas)
    
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################




### Custom classes

class CustomError(Exception):
    pass



# @patch
# def dataloaders(self:FilteredBase, 
#         bs:int=64, # Batch size
#         shuffle_train=None, # (Deprecated, use `shuffle`) Shuffle training `DataLoader`
#         shuffle=True, # Shuffle training `DataLoader`
#         val_shuffle=False, # Shuffle validation `DataLoader`
#         n=None, # Size of `Datasets` used to create `DataLoader`
#         path='.', # Path to put in `DataLoaders`
#         dl_type=None, # Type of `DataLoader`
#         dl_kwargs=None, # List of kwargs to pass to individual `DataLoader`s
#         device=None, # Device to put `DataLoaders`
#         drop_last=None, # Drop last incomplete batch, defaults to `shuffle`
#         val_bs=None, # Validation batch size, defaults to `bs`
#         **kwargs
#     ) -> DataLoaders:
#         if shuffle_train is not None: 
#             shuffle=shuffle_train
#             warnings.warn('`shuffle_train` is deprecated. Use `shuffle` instead.',DeprecationWarning)
#         if device is None: device=default_device()
#         if dl_kwargs is None: dl_kwargs = [{}] * self.n_subsets
#         if dl_type is None: dl_type = self._dl_type
#         if drop_last is None: drop_last = shuffle
#         val_kwargs={k[4:]:v for k,v in kwargs.items() if k.startswith('val_')}
#         def_kwargs = {'bs':bs,'shuffle':shuffle,'drop_last':drop_last,'n':n,'device':device}
#         dl = dl_type(self.subset(0), **merge(kwargs,def_kwargs, dl_kwargs[0],{"dl_idx":0}))
#         def_kwargs = {'bs':bs if val_bs is None else val_bs,'shuffle':val_shuffle,'n':None,'drop_last':False}
#         dls = [dl] + [dl.new(self.subset(i), **merge(kwargs,def_kwargs,val_kwargs,dl_kwargs[i],{"dl_idx":i}))
#                       for i in range(1, self.n_subsets)]
#         return self._dbunch_type(*dls, path=path, device=device)
    
        

# class MyDataLoader(TfmdDL):
#     """DEPRECATED!"""
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.custom_sampler = kwargs.get("custom_sampler")
#         self.dl_idx = kwargs.get("dl_idx")
        
#     def get_idxs(self):
#         idxs = Inf.count if self.indexed else Inf.nones
#         if self.n is not None: idxs = list(itertools.islice(idxs, self.n))
#         if self.custom_sampler != None and self.dl_idx == 0: idxs = self.custom_sampler()
#         if self.shuffle:
#             idxs = self.shuffle_fn(idxs)
            
#         return idxs
    



class SamplerCallback(Callback):
    def __init__(self, df, image_size, path_data, batch_size, balancer, sample_rate_majority_class, tfms=None,
                      location_col_name="tile_path", label_col_name="label", splitter_col_name="is_valid",
                      num_workers=8, slide_id_col_name="slide_name", use_custom_random_state=None, shuffle=False,
                      custom_sampler=None):
        
        store_attr('df,image_size,path_data,batch_size,balancer,sample_rate_majority_class,\
        tfms,location_col_name,label_col_name,splitter_col_name,num_workers,\
        slide_id_col_name,use_custom_random_state,shuffle,custom_sampler', self)
        
       
    def before_epoch(self):
        self.learn.dls, current_idxs = create_dataloader(df=self.df, image_size=self.image_size, path_data=self.path_data, batch_size=self.batch_size, balancer=self.balancer,
                                                         sample_rate_majority_class=self.sample_rate_majority_class, tfms=self.tfms, location_col_name=self.location_col_name,
                                                         label_col_name=self.label_col_name, splitter_col_name=self.splitter_col_name, num_workers=self.num_workers,
                                                         slide_id_col_name=self.slide_id_col_name, use_custom_random_state=self.use_custom_random_state, shuffle=self.shuffle,
                                                         custom_sampler=self.custom_sampler)
        
        setattr(self.learn, "current_idxs", current_idxs)
        # print(current_idxs)
    


class Custom_AccumMetric(AccumMetric):
    '''Stores predictions and targets on CPU in accumulate to perform final calculations with `func`.'''
    def __init__(self, func, dim_argmax=None, activation=fastai.metrics.ActivationType.No, thresh=None, to_np=False,
                 invert_arg=False, flatten=True, tile_count_list=None, **kwargs):
        AccumMetric.__init__(self, func, dim_argmax=dim_argmax, activation=activation, thresh=thresh, to_np=to_np,
                 invert_arg=invert_arg, flatten=flatten, **kwargs)
        
        self.tile_count_list = tile_count_list
    
    def accumulate(self, learn):
        "Store targs and preds from `learn`, using activation function and argmax as appropriate"
        pred = learn.pred
        pred = F.softmax(pred, dim=self.dim_argmax)
        self.accum_values(pred,learn.y)
        self.learn = learn
    
    
    @property
    def value(self):
        "Value of the metric using accumulated preds and targs"
        if len(self.preds) == 0: return
        preds,targs = torch.cat(self.preds),torch.cat(self.targs)
        if self.to_np: preds,targs = preds.numpy(),targs.numpy()
        return self.func(preds, targs, self.tile_count_list, hacky_learner=self.learn, name=self.name, **self.kwargs) if self.invert_args else self.func(targs, preds, self.tile_count_list, hacky_learner=self.learn, name=self.name, **self.kwargs)
    
### Custom Functions

def get_slide_lvl_preds(preds, targs, tile_count_list, thresh, hacky_learner=None, tile_slide_list_new=None, name=None, get_pred_per_slide=False, **kwargs):
    '''Summarize tile based predictions to slide based predictions and returns the balanced accuracy base on slide level
    
    Paramaters
    ----------
    preds : list with two
            Typical prediction tensor/array. The array holds the probability for each class in a separat 2-dimensional array
            
    targs : list()-like
            A tensor that holds the ground truth in an 1-dimensional array 
    
    tile_count_list : list()-like
                      List containing number of tiles per slide -> example=df.groupby("slide_name").count().name.values
    
    thresh : float
             probability threshold for the second class to be accepted as True
    
    Returns
    -------
    final_predictions | tile_slide_list : 2d array-like (tensor) | optional list (str)
                                         Typical prediction tensor on slide basis |  contains the names of the slides per prediction
    '''
        
    predictions=(preds, targs)
    
    # calculate the number of tiles per slide and save it in idx_list
    curr_idx = 0
    idx_list = []
    for tile_count in tile_count_list:
        idx = curr_idx + tile_count
        idx_list.append(idx)
        curr_idx = idx
        
    # merge tile based predictions to slide based predictions
    curr_idx = 0
    final_predictions = [list(), list()]
    for idx in idx_list:
        slide_preds, slide_truths = predictions[0][curr_idx:idx], predictions[1][curr_idx:idx]
        slide_pred, slide_truth = np.mean(slide_preds, axis=0), np.mean(slide_truths.tolist())
        curr_idx = idx
        final_predictions[0].append([slide_pred[0], slide_pred[1]])
        final_predictions[1].append(int(slide_truth))
        
        if slide_truth not in [0., 1.]:
            print("Slide has incorrect truth label of {}".format(slide_truth))
            
    if get_pred_per_slide == True:
        if name not in hacky_learner.pred_per_slide:
            hacky_learner.pred_per_slide[name] = []
            
        # print(f"tile_slide_list_new = {tile_slide_list_new}")
        # print(f"final_predictions[0] = {final_predictions[0]}")
        # print(f"final_predictions[1] = {final_predictions[1]}")
        hacky_learner.pred_per_slide[name].append([dict(zip(tile_slide_list_new, final_predictions[0])), dict(zip(tile_slide_list_new, final_predictions[1]))])
    
    return final_predictions


def get_from_to_idx_for_tile_count_list(tcl, pr, from_to_idx):
    if from_to_idx[0] != 0:
        c = get_from_to_idx_for_tile_count_list(tcl, pr, from_to_idx=[0,from_to_idx[0]])[1]
    else:
        c = 0
    
    my_sum = 0
    my_counter = 0
    for i in tcl:
        if my_counter >= c:
            my_sum += i
            my_counter += 1
            if my_sum == len(pr[from_to_idx[0]:from_to_idx[1]]):
                return [c, my_counter]
        else:
            my_counter += 1
    

def slide_lvl_auroc(preds, targs, tile_count_list, thresh=0.5, from_to_idx=None, name=None, hacky_learner=None, get_pred_per_slide=False, **kwargs):   
    if from_to_idx != None:
        preds_new = preds[from_to_idx[0]:from_to_idx[1]]
        targs_new = targs[from_to_idx[0]:from_to_idx[1]]
        
        from_to_idx_tcl = get_from_to_idx_for_tile_count_list(tile_count_list, targs, from_to_idx=from_to_idx)
        tile_count_list_new = tile_count_list[from_to_idx_tcl[0]:from_to_idx_tcl[1]]
        if "tile_slide_list" in kwargs.keys():
            tile_slide_list_new = kwargs["tile_slide_list"][from_to_idx_tcl[0]:from_to_idx_tcl[1]]
    else:
        preds_new = preds[:]
        targs_new = targs[:]
        tile_count_list_new = tile_count_list[:]
        tile_slide_list_new = kwargs.get("tile_slide_list", None)
    
    final_predictions = get_slide_lvl_preds(preds=preds_new, targs=targs_new, tile_count_list=tile_count_list_new, thresh=thresh,
                                            hacky_learner=hacky_learner, tile_slide_list_new=tile_slide_list_new, name=name, get_pred_per_slide=get_pred_per_slide)
    
    return skm.roc_auc_score(final_predictions[1], [x[1] for x in final_predictions[0]])



def SlideLevelAuroc(axis=-1, sample_weight=None, adjusted=False, is_class=True, thresh=None, activation=None, tile_count_list=None,
                    from_to_idx=None, name=None, **kwargs):
    '''This can added to the metrics list in fastai'''
    dim_argmax = axis if is_class and thresh is None else None
    if activation is None:
        activation = fastai.metrics.ActivationType.Sigmoid if (is_class and thresh is not None) else fastai.metrics.ActivationType.No
    return Custom_AccumMetric(slide_lvl_auroc, dim_argmax=dim_argmax, activation=activation, thresh=thresh,
                       to_np=True, invert_arg=True, tile_count_list=tile_count_list, flatten=False, from_to_idx=from_to_idx, name=name, **kwargs)


def slide_lvl_bal_acc(preds, targs, tile_count_list, thresh=0.5, from_to_idx=None,
                      name=None, hacky_learner=None, get_pred_per_slide=False, **kwargs):
       
    if from_to_idx != None:
        preds_new = preds[from_to_idx[0]:from_to_idx[1]]
        targs_new = targs[from_to_idx[0]:from_to_idx[1]]
        
        from_to_idx_tcl = get_from_to_idx_for_tile_count_list(tile_count_list, targs, from_to_idx=from_to_idx)
        tile_count_list_new = tile_count_list[from_to_idx_tcl[0]:from_to_idx_tcl[1]]
        if "slide_tile_list" in kwargs.keys():
            slide_tile_list_new = kwargs["slide_tile_list"][from_to_idx_tcl[0]:from_to_idx_tcl[1]]
            
    else:
        preds_new = preds[:]
        targs_new = targs[:]
        tile_count_list_new = tile_count_list[:]
        tile_slide_list_new = kwargs.get("tile_slide_list", None)
        
        
    final_predictions = get_slide_lvl_preds(preds=preds_new, targs=targs_new, tile_count_list=tile_count_list_new, thresh=thresh,
                                            hacky_learner=hacky_learner, tile_slide_list_new=tile_slide_list_new, name=name, get_pred_per_slide=get_pred_per_slide, **kwargs)
    
    y_pred_new = np.array([1 if val[1] > thresh else 0 for val in final_predictions[0]])
    y_true_new = final_predictions[1]
    
    tn, fp, fn, tp = skm.confusion_matrix(y_true_new, y_pred_new).ravel()
    
    return skm.balanced_accuracy_score(y_true_new, y_pred_new)


def SlideLevelBalAcc(axis=-1, sample_weight=None, adjusted=False, is_class=True, thresh=None, activation=None, tile_count_list=None, 
                     from_to_idx=None, name=None, **kwargs):
    '''This can added to the metrics list in fastai'''
    dim_argmax = axis if is_class and thresh is None else None
    if activation is None:
        activation = fastai.metrics.ActivationType.Sigmoid if (is_class and thresh is not None) else fastai.metrics.ActivationType.No
        
    return Custom_AccumMetric(slide_lvl_bal_acc, dim_argmax=dim_argmax, activation=activation, thresh=thresh,
                       to_np=True, invert_arg=True, tile_count_list=tile_count_list, flatten=False, from_to_idx=from_to_idx, name=name, **kwargs)


def slide_lvl_sens(y_true, y_pred, tile_count_list=None, labels=None, pos_label=1, average='binary',
                   sample_weight=None, zero_division="warn", thresh=0.5, from_to_idx=None, name=None,
                   hacky_learner=None, get_pred_per_slide=False, **kwargs):
    
    if from_to_idx != None:
        y_pred_new = y_pred[from_to_idx[0]:from_to_idx[1]]
        y_true_new = y_true[from_to_idx[0]:from_to_idx[1]]
        from_to_idx_tcl = get_from_to_idx_for_tile_count_list(tile_count_list, y_true, from_to_idx=from_to_idx)
        tile_count_list_new = tile_count_list[from_to_idx_tcl[0]:from_to_idx_tcl[1]]
        if "slide_tile_list" in kwargs.keys():
            slide_tile_list_new = kwargs["slide_tile_list"][from_to_idx_tcl[0]:from_to_idx_tcl[1]]
            
    else:
        y_pred_new = y_pred[:]
        y_true_new = y_true[:]
        tile_count_list_new = tile_count_list[:]
        tile_slide_list_new = kwargs.get("tile_slide_list", None)

    final_predictions = get_slide_lvl_preds(preds=y_pred_new, targs=y_true_new, tile_count_list=tile_count_list_new, thresh=thresh,
                                            hacky_learner=hacky_learner, tile_slide_list_new=tile_slide_list_new, name=name, get_pred_per_slide=get_pred_per_slide, **kwargs)
    
        
    y_pred_new = np.array([1 if val[1] > thresh else 0 for val in final_predictions[0]])
    y_true_new = final_predictions[1]
    
    return skm.recall_score(y_true_new, y_pred_new, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight, zero_division=zero_division)
    

def SlideLevelSensitivity(axis=-1, sample_weight=None, adjusted=False, is_class=True, thresh=None, activation=None,
                          tile_count_list=None, from_to_idx=None, name=None, **kwargs):
    '''This can added to the metrics list in fastai'''
    dim_argmax = axis if is_class and thresh is None else None
    if activation is None:
        activation = fastai.metrics.ActivationType.Sigmoid if (is_class and thresh is not None) else fastai.metrics.ActivationType.No
        
    return Custom_AccumMetric(slide_lvl_sens, dim_argmax=dim_argmax, activation=activation, thresh=thresh,
                       to_np=True, invert_arg=False, tile_count_list=tile_count_list, flatten=False, from_to_idx=from_to_idx, name=name, **kwargs)


def slide_lvl_spec(y_true, y_pred, tile_count_list=None, labels=None, pos_label=0, average='binary',
                 sample_weight=None, zero_division="warn", thresh=0.5, from_to_idx=None, hacky_learner=None,
                   name=None, get_pred_per_slide=False, **kwargs):
    
    if from_to_idx != None:
        y_pred_new = y_pred[from_to_idx[0]:from_to_idx[1]]
        y_true_new = y_true[from_to_idx[0]:from_to_idx[1]]

        from_to_idx_tcl = get_from_to_idx_for_tile_count_list(tile_count_list, y_true, from_to_idx=from_to_idx)
        tile_count_list_new = tile_count_list[from_to_idx_tcl[0]:from_to_idx_tcl[1]]
        if "slide_tile_list" in kwargs.keys():
            slide_tile_list_new = kwargs["slide_tile_list"][from_to_idx_tcl[0]:from_to_idx_tcl[1]]
            
    else:
        y_pred_new = y_pred[:]
        y_true_new = y_true[:]
        tile_count_list_new = tile_count_list[:]
        tile_slide_list_new = kwargs.get("tile_slide_list", None)

    final_predictions = get_slide_lvl_preds(preds=y_pred_new, targs=y_true_new, tile_count_list=tile_count_list_new, thresh=thresh,
                                            hacky_learner=hacky_learner, tile_slide_list_new=tile_slide_list_new, name=name, get_pred_per_slide=get_pred_per_slide, **kwargs)

    y_pred_new = np.array([1 if val[1] > thresh else 0 for val in final_predictions[0]])
    y_true_new = final_predictions[1]
    
    return skm.recall_score(y_true_new, y_pred_new, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight, zero_division=zero_division)
    

def SlideLevelSpecificity(axis=-1, sample_weight=None, adjusted=False, is_class=True, thresh=None, activation=None, tile_count_list=None, 
                          from_to_idx=None, name=None, **kwargs):
    '''This can added to the metrics list in fastai'''
    dim_argmax = axis if is_class and thresh is None else None
    if activation is None:
        activation = fastai.metrics.ActivationType.Sigmoid if (is_class and thresh is not None) else fastai.metrics.ActivationType.No
        
    return Custom_AccumMetric(slide_lvl_spec, dim_argmax=dim_argmax, activation=activation, thresh=thresh,
                       to_np=True, invert_arg=False, tile_count_list=tile_count_list, flatten=False, from_to_idx=from_to_idx, name=name, **kwargs)


def tile_lvl_sensitivity(y_true, y_pred, labels=None, pos_label=1, average='binary',
                 sample_weight=None, zero_division="warn", **kwargs):
    return skm.recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division="warn")
    

def Sensitivity(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None):
    "Recall for single-label classification problems"
    return skm_to_fastai(tile_lvl_sensitivity, axis=axis,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)

def tile_lvl_specificity(y_true, y_pred, labels=None, pos_label=0, average='binary',
                 sample_weight=None, zero_division="warn", **kwargs):
    return skm.recall_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight, zero_division=zero_division)


def Specificity(axis=-1, labels=None, pos_label=0, average='binary', sample_weight=None):
    "Recall for single-label classification problems"
    return skm_to_fastai(tile_lvl_specificity, axis=axis,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)

# def get_sampled_idxs_for_slide(df, slide_id_col_name, unique_slide_name, sample_rate, use_custom_random_state):
#     """DEPRECATED"""
#     idxs_sample = []
#     if use_custom_random_state != None:
#         random.seed(use_custom_random_state)

#     possible_idxs = list(df[df[slide_id_col_name] == unique_slide_name].index.values)
#     if len(possible_idxs) > sample_rate:
#         ### undersample
#         idxs_sample += random.sample(possible_idxs, sample_rate)
#     elif len(possible_idxs) < sample_rate:
#         ### oversample
#         idxs_sample += possible_idxs
#         while len(idxs_sample)+len(possible_idxs) < sample_rate:
#             idxs_sample += possible_idxs
#         idxs_sample += random.sample(possible_idxs, sample_rate-len(idxs_sample))
#     else:
#         ### take all possible idxs exactly once
#         idxs_sample = possible_idxs

#     return idxs_sample


# def get_newly_sampled_df_train(df, batch_size, sample_rate_majority_class, location_col_name,
#                                label_col_name, splitter_col_name, slide_id_col_name, use_custom_random_state):
#     """DEPRECATED"""

#     ### calculate some important parameters
#     df_train = df[df[splitter_col_name]==False]
#     tmp_dict = df_train.drop_duplicates(slide_id_col_name).groupby(label_col_name).count()[slide_id_col_name].to_dict()
#     majority_class = max(tmp_dict, key=tmp_dict.get)
#     num_slide_majority_class = df_train[df_train[label_col_name]==majority_class][slide_id_col_name].nunique()
#     num_slide_minority_class = df_train[df_train[label_col_name]!=majority_class][slide_id_col_name].nunique()
#     sample_rate_minority_class = int(np.round((num_slide_majority_class/num_slide_minority_class)*sample_rate_majority_class))

#     majority_class_idxs = []
#     minority_class_idxs = []

#     ### iterate through every slide_name
#     unique_slide_names = list(df_train[df_train[splitter_col_name]==False].slide_name.unique())
#     for unique_slide_name in unique_slide_names:
#         ### set the sample_rate based on the class of the slide_name
#         if df_train[df_train[slide_id_col_name]==unique_slide_name].label.values[0] == majority_class:
#             majority_class_idxs += get_sampled_idxs_for_slide(df_train, slide_id_col_name, unique_slide_name, sample_rate_majority_class, use_custom_random_state)
#         else:
#             minority_class_idxs += get_sampled_idxs_for_slide(df_train, slide_id_col_name, unique_slide_name, sample_rate_minority_class, use_custom_random_state)


#     ### create a list from df_train based on batch_size that contains an equal amount of patches for each class per <batch_size> elements
#     num_batches = int((len(majority_class_idxs)+len(minority_class_idxs))/batch_size)
#     new_idxs = []

#     if batch_size%2 != 0:
#         raise CustomError("An error occurred in DataLoaders custom sampler: batch_size has to be even!")

#     for i in range(num_batches):
#         tmp_idxs = []
#         if len(majority_class_idxs) >= batch_size//2 and len(minority_class_idxs) >= batch_size//2:
#             tmp_idxs += [majority_class_idxs.pop(random.randrange(len(majority_class_idxs))) for _ in range(batch_size//2)]
#             tmp_idxs += [minority_class_idxs.pop(random.randrange(len(minority_class_idxs))) for _ in range(batch_size//2)]
#             random.shuffle(tmp_idxs)
#             new_idxs += tmp_idxs
#         else:
#             break

#     df_new_train = pd.DataFrame()
#     df_new_train["idx"] = new_idxs

#     df_new_train = pd.merge(df_new_train, df_train, left_on="idx", right_index=True, how="left")

#     df_new_train = pd.concat([df_new_train, df[df[splitter_col_name]==True]])
#     df_new_train = df_new_train.drop(labels="idx", axis=1)
#     return df_new_train


def my_new_sampler(all_majority_class_idxs, all_minority_class_idxs, sample_rate_majority_class, sample_rate_minority_class, use_custom_random_state=None):
    """
    Ziel ist es zwei variablen neu zu berechnen. majority_class_idxs und minority_class_idxs.
    Diese beinhalten jeweils die indices to allen Patches der majority und minority class, welche
    nur für den dataframe gültig sind, der beim erstellen des DataLoaders übergeben wurde. 
    
    Diese Methode soll übergeben bekommen:
        all_majority_class_idxs : dict
                                  beinhalten das mapping vom slide_names auf alle
                                  möglichen indices für majority class.
        
        all_minority_class_idxs : dict
                                  beinhalten das mapping vom slide_names auf alle
                                  möglichen indices für minority class.
                                  
        sample_rate_majority_class : int
                                     zeigt an wie viele indices aus jedem Eintrag von
                                     all_majority_class_idxs random gezogen werden sollen.
        
        sample_rate_minority_class : int
                                     zeigt an wie viele indices aus jedem Eintrag von
                                     all_minority_class_idxs random gezogen werden sollen.
    
    
    Es wird über beide dictionaries iteriert und dabei die respektiv richtige anzahl an inidces
    pro Eintrag gespampled. Alle inidices werden in einer liste gespeichert.
    
    """
    
    # print(f"all_majority_class_idxs = {all_majority_class_idxs}")
    # print(f"all_minority_class_idxs = {all_minority_class_idxs}")
    
    idxs = []
    major_idxs, minor_idxs = [], []
    for i in all_majority_class_idxs:
        if use_custom_random_state != None:
            random.seed(use_custom_random_state)
        tmp = []
        while len(tmp) + len(all_majority_class_idxs[i]) < sample_rate_majority_class:
            tmp += all_majority_class_idxs[i]
        tmp += random.sample(all_majority_class_idxs[i], sample_rate_majority_class-len(tmp))
        major_idxs += tmp
        idxs += tmp
        # print()
        # print(f"slide = {i} : {tmp}")
        
    for i in all_minority_class_idxs:
        if use_custom_random_state != None:
            random.seed(use_custom_random_state)
        tmp = []
        while len(tmp) + len(all_minority_class_idxs[i]) < sample_rate_minority_class:
            tmp += all_minority_class_idxs[i]
        tmp += random.sample(all_minority_class_idxs[i], sample_rate_minority_class-len(tmp))
        minor_idxs += tmp
        idxs += tmp
        # print()
        # print(f"slide = {i} : {tmp}")
    
    return idxs


def create_dataloader(df, image_size, path_data, batch_size, balancer, sample_rate_majority_class=10, tfms=None,
                      location_col_name="tile_path", label_col_name="label", splitter_col_name="is_valid",
                      num_workers=16, slide_id_col_name="slide_name", use_custom_random_state=None, shuffle=True,
                      custom_sampler=None):
    
    idxs = []
    
    ### in case a custom_sampler is needed, use this:
    if custom_sampler != None:
        ### NOT IMPLEMENTED!
        if balancer == True:
            print("Your custom balancer will not be used, because balancer=True.")
        pass
    
    
    ## balance slides on tile level
    if balancer == True:
        tmp_df = df[df["is_valid"]==False]
        tmp_dict = df.drop_duplicates(slide_id_col_name).groupby(label_col_name).count()[slide_id_col_name].to_dict()

        majority_class = max(tmp_dict, key=tmp_dict.get)

        num_slide_majority_class = tmp_df[tmp_df[label_col_name]==majority_class][slide_id_col_name].nunique()

        num_slide_minority_class = tmp_df[tmp_df[label_col_name]!=majority_class][slide_id_col_name].nunique()

        sample_rate_minority_class = int(np.round((num_slide_majority_class/num_slide_minority_class)*sample_rate_majority_class))
        
        ### create all_majority_class_idxs (mapping from slide_id_col_name to a list of idxs majority slides)
        all_majority_class_idxs = dict()
        for i in tmp_df[tmp_df[label_col_name]==majority_class][slide_id_col_name].unique():
            if i not in all_majority_class_idxs:
                all_majority_class_idxs[i] = []
            all_majority_class_idxs[i] += list(tmp_df[tmp_df[slide_id_col_name] == i].index.values)

        ### create all_minority_class_idxs (mapping from slide_id_col_name to a list of idxs for minority slides)
        all_minority_class_idxs = dict()
        for i in tmp_df[tmp_df[label_col_name]!=majority_class][slide_id_col_name].unique():
            if i not in all_minority_class_idxs:
                all_minority_class_idxs[i] = []
            all_minority_class_idxs[i] += list(tmp_df[tmp_df[slide_id_col_name] == i].index.values)
            
        custom_sampler = partial(my_new_sampler, all_majority_class_idxs, all_minority_class_idxs, sample_rate_majority_class, sample_rate_minority_class, use_custom_random_state)
        idxs = custom_sampler()
        df = pd.concat([df.filter(items = idxs, axis=0), df[df["is_valid"] == True]])
    

    
    ## determine transforms
    tfms_orig = [Normalize.from_stats(*imagenet_stats), #]#,  # Normalize the images with the specified mean and standard deviation used in imagenet_stats
                *aug_transforms(size=image_size)]  # Add default transformations
    
    if tfms != None:
        for i in tfms:
            tfms_orig.append(i)

    tfms = tfms_orig
    
    ## create dataloader
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader(location_col_name, pref=path_data),
        get_y=ColReader(label_col_name),
        splitter=ColSplitter(col=splitter_col_name),
        batch_tfms=tfms)
    
    return dblock.dataloaders(df, bs=batch_size, num_workers=num_workers, shuffle=shuffle), idxs
    # return dblock.dataloaders(df, bs=batch_size, num_workers=num_workers, shuffle=shuffle, dl_type=MyDataLoader, custom_sampler=custom_sampler) 


def limit_number_of_tiles(df, target_column, target_name, number_of_tiles, custom_random_state=115):
    """ This function takes a dataframe and reduces the number of tiles for every slide of a given column and name
    
    Parameters
    ----------
    
    df : DataFrame
         dataframe for training
         
    target_column : str
                    name of column header of the df in which target_name is found
         
    target_name : str
                  name of set
              
    number_of_tiles : int
                      determines the final number of tiles in df for each slide
                
    
    Returns
    -------
    
    df : df
         DataFrame with reduced number of tiles per slide for target set
    """
    
    df_tmp = df[df[target_column] != target_name]
    df =  df[df[target_column] == target_name]
    df = df.groupby("slide_name").apply(lambda x: x.sample(n=min(number_of_tiles, x.shape[0]), random_state=custom_random_state)).reset_index(drop=True)
    df = pd.concat([df, df_tmp])
    df = df.reset_index(drop=True)
    return df
                
    
    
class MyCometCallback(Callback):
    """
    Paramater
    ----------
    
    experiment_obj : comet_ml.Experiment
                     Experiment object, which has to be initialized like this -> experiment = comet_ml.Experiment(project_name="my_project").
                     
    learner_hps : list(<str>)
                  List of names, which can be found in 'learn.__stored_args__.keys()'. If this is None all parameters from 'learn.__stored_args__' are used.
                  
    recorder_metrics : list(<str>)
                       List of names, which can be found in 'learn.recorder.hps.keys()'. If this is None all parameters from 'learn.recorder.hps' are used.
                       
    dls_hps : list(<str>)
              List of names, which can be found in 'learn.dls.train.__dict__.keys()'. If this is None all parameters from 'learn.dls.train.__dict__' are used.
              
    save_obj : list(<str>)
               List of paths, which will be uploaded to Comet. This can be used to save DataFrames for each experiment.
               
    rename_params_dict : dictionary
                         Dictionary, which is used to rename hyperparameters or metrics for the Comet-plots in 'learner_hps', 'recorder_metrics' or 'dls_hps'.
                         
    save_pth : bool
               If True, the model weights are saved, using learn.save() after fit, without needing the SaveModelCallback.
               
    model_name : str
                 Name of the model. Is used only if save_pth is True.
               
    log_model_weights : bool
                        If True and SaveModelCallback is used, model_weights are logged.
    
    train_run_counter : int
                        Indicates the number of the current fit.
    """
    
    order = Recorder.order + 1
    def __init__(self, experiment_obj, learner_hps=None, recorder_metrics=None, dls_hps=None, save_obj=[],
                 rename_params_dict=dict(), save_pth=True, model_name="model", log_model_weights=True, 
                 train_run_counter=0, save_pred_per_slide=True):
        
        self.experiment = experiment_obj
        self.train_run_counter = train_run_counter
        self.learner_hps = learner_hps
        self.recorder_metrics = recorder_metrics
        self.dls_hps = dls_hps
        self.save_obj = save_obj
        self.rename_params_dict = rename_params_dict
        self.model_name = model_name
        self.log_model_weights = log_model_weights
        self.save_pth = save_pth
        self.save_pred_per_slide = save_pred_per_slide

    def before_fit(self):
        try:
            self.experiment.log_parameter(f"n_epoch__fit_{self.train_run_counter}", str(self.learn.n_epoch))
            self.experiment.log_parameter("model_class", str(type(self.learn.model)))
        except:
            print(f"Did not log all properties.")
            
        try:
            if self.learner_hps == None:
                self.learner_hps = list(self.learn.__stored_args__.keys())
            
            ### learner_hps
            for i in self.learner_hps:
                if i in self.rename_params_dict.keys():
                    alternate_i = self.rename_params_dict[i]
                else:
                    alternate_i = i
                self.experiment.log_parameter(f"{alternate_i}_{self.train_run_counter}", str(self.learn.__stored_args__[i]))  
                
        except:
            print(f"Did not log all learner hyperparameters.")
            
        try:
            if self.dls_hps == None:
                self.dls_hps = list(self.learn.dls.train.__dict__.keys())
            
            ### dls_hps
            for i in self.dls_hps:
                if i not in ["__stored_args__", "dataset", "_DataLoader__idxs"]:
                    if i in self.rename_params_dict.keys():
                        alternate_i = self.rename_params_dict[i]
                    else:
                        alternate_i = i
                    self.experiment.log_parameter(f"{alternate_i}__fit_{self.train_run_counter}", str(self.learn.dls.train.__dict__[i]))
                    
        except:
            print(f"Did not log all dls hyperparameters.")
                        
        try:
            ### save already stored object 
            if self.save_obj != [] and self.train_run_counter == 0:
                for i in self.save_obj:
                    experiment.log_asset(Path(i))
        except:
            print(f"Did not log all objects")

        try:
            with tempfile.NamedTemporaryFile(mode="w") as f:
                with open(f.name, "w") as g:
                    g.write(repr(self.learn.model))
                self.experiment.log_asset(f.name, f"model_summary__fit_{self.train_run_counter}.txt")
        except:
            print("Did not log model summary. Check if your model is PyTorch model.")
        

    def after_batch(self):
        # log loss and opt.hypers
        if self.learn.training:
            self.experiment.log_metric("smooth_loss", self.learn.smooth_loss)
            self.experiment.log_metric("loss", self.learn.loss)
            self.experiment.log_metric("train_iter", self.learn.train_iter)
            for i, h in enumerate(self.learn.opt.hypers):
                for k, v in h.items():
                    self.experiment.log_metric(f"opt.hypers.{k}_layer_{i}", v)
                    
        try:
            if self.recorder_metrics == None:
                self.recorder_metrics = ["lrs"]
            
            ### recorder_metrics
            for i in self.recorder_metrics:
                if i in self.rename_params_dict.keys():
                    alternate_i = self.rename_params_dict[i]
                else:
                    alternate_i = i
                self.experiment.log_metric(f"{alternate_i}", self.recorder.__dict__[i][-1])
                
        except:
            print(f"Did not log all learner hyperparameters.")

    def after_epoch(self):
        # log metrics
        for n, v in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if n not in ["epoch", "time"]:
                self.experiment.log_metric(f"epoch__{n}", v)
            if n == "time":
                self.experiment.log_text(f"epoch__{n}", str(v))
                
        # log model weights
        if self.log_model_weights and hasattr(self.learn, "save_model"):
            if self.learn.save_model.every_epoch:
                _file = join_path_file(
                    f"{self.learn.save_model.fname}_{self.learn.save_model.epoch}",
                    self.learn.path / self.learn.model_dir,
                    ext=".pth",
                )
            else:
                _file = join_path_file(
                    self.learn.save_model.fname,
                    self.learn.path / self.learn.model_dir,
                    ext=".pth",
                )
            self.experiment.log_asset(_file)

                
    def after_fit(self):       
        if self.save_pth:
            if isinstance(self.learn.model_dir, str):
                tmp_path = Path(self.learn.model_dir)
            else:
                tmp_path = self.learn.model_dir
            _file = join_path_file(
                self.model_name,
                tmp_path,
                ext=".pth",
            )       
            self.learn.save(self.model_name)
            self.experiment.log_asset(_file)

            
        if self.save_pred_per_slide:
            self.experiment.log_asset_data(str(self.learn.pred_per_slide), name="pred_per_slide_dict", overwrite=True)
        
        self.train_run_counter += 1
        
        
def get_random_validation_set(df, percentage_valid=30, lbls=["0", "1"], clinics=["Berlin", "Erlangen", "Wuerzburg", "Muenchen"]):
    """return df with addition column <is_valid>, which denotes whether or not a slide is used for validation"""
    valid_slide_names = []
    
    for l in lbls:
        for c in clinics:
            pct = int(df[(df["clinic"] == c) & (df["label"] == l)]["slide_name"].nunique()/100*percentage_valid)
            random.seed(115)
            tmp = (random.sample(list(df[(df["clinic"] == c) & (df["label"] == l)]["slide_name"].unique()), pct))
            valid_slide_names += tmp
    df["is_valid"] = [True if x in valid_slide_names else False for x in df.slide_name]
    return df