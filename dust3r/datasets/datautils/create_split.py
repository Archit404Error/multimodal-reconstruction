import pickle
import numpy as np
import os, json, sys
from os.path import join
import glob
from tqdm import tqdm

submodule_path = ( "/share/phoenix/nfs05/S8/gc492/foundation/data" )
assert os.path.exists(submodule_path)
sys.path.insert(0, submodule_path)
from data_helpers import *
from depth_helpers import *

import PIL 
from PIL import Image # I forget which one is the correct one
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None
Image.LOAD_TRUNCATED_IMAGES = True


def save_all_pairs():

    outpath = '/share/phoenix/nfs06/S9/gc492/data/megascenes/pairs/grouped_data'
    path = '/share/phoenix/nfs06/S9/gc492/data/megascenes/pairs/alldata'
    indices = os.listdir(path)
    print(indices)

    largepairs = []
    medpairs = []
    smallpairs = []

    for i, idx in enumerate(tqdm(indices)):
        try:
            tp = f'/share/phoenix/nfs06/S9/gc492/data/megascenes/pairs/alldata/{idx}'
            with open(join(tp, f'large_overlap_{idx}.pkl'), 'rb') as f:
                lp = pickle.load(f)
            with open(join(tp, f'med_overlap_{idx}.pkl'), 'rb') as f:
                mp = pickle.load(f)
            with open(join(tp, f'small_overlap_{idx}.pkl'), 'rb') as f:
                sp = pickle.load(f)
        
            print(f"{idx}: {len(lp)} large pairs, {len(mp)} medium pairs, {len(sp)} small pairs") # {len(imgdict)} images, 

            largepairs.extend(lp)
            medpairs.extend(mp)
            smallpairs.extend(sp)

            if (i!=0 and i%10==0) or i==len(indices)-1:
                with open(f"{outpath}/large_pairs_{i}.pkl", 'wb') as f:
                    pickle.dump(largepairs, f)
                with open(f"{outpath}/med_pairs_{i}.pkl", 'wb') as f:
                    pickle.dump(medpairs, f)
                with open(f"{outpath}/small_pairs_{i}.pkl", 'wb') as f:
                    pickle.dump(smallpairs, f)

                largepairs = []
                medpairs = []
                smallpairs = []

            

        except Exception as e:
            print(e)
            continue


def save_clean_dicts():
    outpath = '/share/phoenix/nfs06/S9/gc492/data/megascenes/pairs/clean_imgdicts'
    path = '/share/phoenix/nfs06/S9/gc492/data/megascenes/pairs/alldata'
    indices = os.listdir(path)
    print(indices)

    full_imgdict = {}

    for i, idx in enumerate(tqdm(indices)):
        try:
            tp = f'{path}/{idx}'
            with open(join(tp, f'img_dict_{idx}.pkl'), 'rb') as f:
                imgdict = pickle.load(f)
            
            print(f"{idx}: {len(imgdict)} images") 

            for k,v in imgdict.items():
                try:
                    full_imgdict[k] = {}
                    full_imgdict[k]['extrinsics'] = extrinsics_to_matrix(v['extrinsics'])
                    full_imgdict[k]['intrinsics'] = intrinsics_to_matrix(v['intrinsics'])
                    full_imgdict[k]['pointxyz'] = np.stack(v['pointxyz'])[:,0:3]
                                
                except Exception as e:
                    print(e)
                    continue
            
            if (i!=0 and i%10==0) or i==len(indices)-1:
                with open(f"{outpath}/imgdict_{i}.pkl", 'wb') as f:
                    pickle.dump(full_imgdict, f)
                full_imgdict = {}


        except Exception as e:
            print(e)
            continue

# to check ram usage
# from 43G -> 81G (roughly 38G, same as the size of the pickles which is 36G)
def load_clean_dicts():
    path = '/share/phoenix/nfs06/S9/gc492/data/megascenes/pairs/clean_imgdicts'
    picks = glob.glob(f"{path}/*.pkl")
    print(picks)

    full_imgdict = {}

    for i, pp in enumerate(tqdm(picks)):
        with open(pp, 'rb') as f:
            full_imgdict.update(pickle.load(f))

save_all_pairs()