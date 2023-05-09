import comet_ml, pickle, os, glob, sys, torch, random, gc, fastai, optuna, logging, configparser
from fastai.vision.all import *
from fastai.vision.augment import _slice
import sklearn.metrics as skm
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from optuna.integration import FastAIPruningCallback
from misc_funcs import *
from optuna.samplers import TPESampler
              
    
# args
parser = ArgumentParser()
parser.add_argument('-model', default='resnet18', type=str)
parser.add_argument('-model_name', default='resnet18', type=str)
parser.add_argument('-subname', default='test', type=str)
parser.add_argument('-num_workers', default=6, type=int)
parser.add_argument('-name_main_dataframe', default="df_main.p", type=str)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs_frozen', default=1, type=int)
parser.add_argument('-epochs_unfrozen', default=1, type=int)
parser.add_argument('-learning_rate', default=2e-03, type=float)
parser.add_argument('-minimal_training', default='False', type=str)
parser.add_argument('-custom_random_state', default=32, type=int)
parser.add_argument('-sample_rate_majority_class', default=50, type=int)
parser.add_argument('-clinic', default='Berlin', type=str)
parser.add_argument('-random_erasing_probability', default=2e-03, type=float)
parser.add_argument('-random_erasing_max_count', default=2e-03, type=float)


args = parser.parse_args()


### Local paths
path_data_misc = Path("../data_misc/")
path_data = Path("../data")
path_models = Path("../models/")
path_results = Path("../results/")
path_swarm = Path("../swarmCallback/")


### define some parameters
model_dict = dict({"resnet18":resnet18, "resnet34":resnet34, "resnet50":resnet50})
model = model_dict[args.model]
model_name = args.model_name
subname = args.subname
clinic = args.clinic
prefix = model_name + "_" + subname + "_" + clinic + "_"
custom_random_state = (None if args.custom_random_state==0 else args.custom_random_state)


df_main_path = f"{path_data_misc}/{args.name_main_dataframe}"
minimal_training = (True if args.minimal_training=="True" else False)
batch_size = args.batch_size
num_workers = args.num_workers
image_size = 224
sample_rate_majority_class = args.sample_rate_majority_class
epochs_frozen = args.epochs_frozen
epochs_unfrozen = args.epochs_unfrozen
learning_rate = args.learning_rate

### log file handling
old_stdout = sys.stdout
log_file = open(path_results/(prefix+"log_file.txt"), "w")
sys.stdout = log_file


def training_procedure(sample_rate_majority_class, epochs_frozen, epochs_unfrozen, learning_rate, random_erasing_max_count, random_erasing_probability, experiment):

    ### create dataframes
    pd.options.mode.chained_assignment = None  # default='warn'
    df_main = pickle.load(open(df_main_path, "rb"))
    ### retrieve all participating clinics from df
    df_train = df_main[df_main["clinic"] == clinic]
    ### create validation set from all participating clinics
    df_train = get_random_validation_set(df_train, percentage_valid=30, lbls=["0", "1"], clinics=[clinic])
    
    df_train = df_train.sort_values(by=["is_valid", "clinic", "slide_name", "tile_path"], ascending=True).reset_index(drop=True)
    tile_count_list_train = df_train[df_train["is_valid"] == True].groupby(["slide_name"], sort=False).count().tile_path.values
    tile_slide_list_train = df_train[df_train["is_valid"] == True].slide_name.unique()


    if minimal_training:
        df_train = limit_number_of_tiles(df_train, "is_valid", False, 10)  
        df_train = limit_number_of_tiles(df_train, "is_valid", True, 10)
        tile_count_list_train = df_train[df_train["is_valid"] == True].groupby(["slide_name"], sort=False).count().tile_path.values
        sample_rate_majority_class=5

    tfms = [MyRandomErasing(p=random_erasing_probability, max_count=random_erasing_max_count)]


    def get_kwargs(**kwargs): return kwargs

    ### create DataLoaders
    dataloader_args = get_kwargs(df=df_train, image_size=image_size, path_data=path_data, batch_size=batch_size, balancer=True,
                                 sample_rate_majority_class=sample_rate_majority_class, tfms=tfms, location_col_name="tile_path",
                                 label_col_name="label", splitter_col_name="is_valid", num_workers=num_workers,
                                 slide_id_col_name="slide_name", use_custom_random_state=custom_random_state, shuffle=True)
    

    dls, _ = create_dataloader(**dataloader_args)

    
    ### define metrics
    my_metrics = []
    my_metrics.append(SlideLevelSensitivity(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train))
    my_metrics.append(SlideLevelSpecificity(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train))
    my_metrics.append(SlideLevelBalAcc(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train))
    my_metrics.append(SlideLevelAuroc(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train, get_pred_per_slide=True))
    
    ### manage indexing for different sets
    tmp = df_train[df_train["is_valid"]==True].reset_index(drop=True)
    
    
    for current_clinic in df_train.clinic.unique():
        tmp_first_idx = min(tmp[(tmp["clinic"] == current_clinic) & (tmp["is_valid"] == True)].index.values)
        tmp_last_idx = max(tmp[(tmp["clinic"] == current_clinic) & (tmp["is_valid"] == True)].index.values)+1
        my_metrics.append(SlideLevelAuroc(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train, from_to_idx=[tmp_first_idx,tmp_last_idx], name=f"slide_lvl_auroc_{current_clinic}"))



    ### define callbacks
    my_cbs = [SamplerCallback(**dataloader_args)]
       

    ### create learner
    learn = vision_learner(dls, model, metrics=my_metrics, cbs=my_cbs, model_dir=path_models).to_fp16()
    
    setattr(learn, "pred_per_slide", dict())
    
    ### train the model
    with learn.no_bar():
        learn.fine_tune(freeze_epochs=epochs_frozen, epochs=epochs_unfrozen, base_lr=learning_rate)
              
    torch.save(learn.model.state_dict(), path_models/(prefix+"state_dict.pkl"))
    learn.export(path_models/(prefix+"export.pkl"))
    return None
    

training_procedure(args.sample_rate_majority_class, args.epochs_frozen, args.epochs_unfrozen, args.learning_rate, args.random_erasing_max_count, args.random_erasing_probability, None)

sys.stdout = old_stdout
log_file.close()