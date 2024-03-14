import torch
import os
import time
import numpy as np
import cv2
from tqdm import tqdm
import nni

from tools import save_tensor
from tools.transformer import mixup_transform
from tools.plot import VisualBoard
from tools.parser import logger
from tools.solver import Cosine_lr_scheduler, Plateau_lr_scheduler, ALRS
from detlib.mmocr.tools.nni_attack_test import evaluate
from detlib.utils import init_detector_weight

scheduler_factor = {
    'plateau': Plateau_lr_scheduler,
    'cosine': Cosine_lr_scheduler,
    'ALRS': ALRS,
}

detector_info = {
    "PS_IC15":{"script_path":"detlib/mmocr/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_adv.py",
    "weight_path":"https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth"},
    "DB_r50":{"script_path":"detlib/mmocr/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015_adv.py",
    "weight_path":"https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth"},
    "DBPP_r50":{"script_path":"detlib/mmocr/configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015_adv.py",
    "weight_path":"https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth"}
}

def search(cfg, data_root, detector_attacker, args=None):
    
    tuner_params = nni.get_next_parameter()
    detector_attacker.weight["PS_IC15"] = float(tuner_params['pse_factor'])
    detector_attacker.weight["DB_r50"] = float(tuner_params['db_factor'])
    detector_attacker.weight["DBPP_r50"] = float(tuner_params['dbpp_factor'])
    detector_attacker.init_universal_patch()
    img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]

    p_obj = detector_attacker.patch_obj.patch
    optimizer = torch.optim.Adam([p_obj], lr=cfg.ATTACKER.START_LEARNING_RATE, amsgrad=True)
    scheduler = scheduler_factor[cfg.ATTACKER.scheduler](optimizer)
    detector_attacker.attacker.set_optimizer(optimizer)
    loss_array = []

    for epoch in range(10):
        #attack 10 epochs
        ep_loss = 0
        for index,img_name in enumerate(img_names):
            
            loss = detector_attacker.attack(img_name, mode='parallel')
            ep_loss += loss
        
        ep_loss /= len(img_names)
        print('           ep loss : ', ep_loss)
        if cfg.ATTACKER.scheduler == 'plateau':
            scheduler.step(ep_loss)
        elif cfg.ATTACKER.scheduler != 'ALRS':
            scheduler.step()
    
    # begin testing
    patch = detector_attacker.universal_patch.cpu()
    # input patch & detector names, output h-mean
    ps_recall = evaluate(patch,detector_info["PS_IC15"])
    db_recall = evaluate(patch,detector_info["DB_r50"])
    dbpp_recall = evaluate(patch,detector_info["DBPP_r50"])
    nni.report_intermediate_result(ps_recall)
    nni.report_intermediate_result(db_recall)
    nni.report_intermediate_result(dbpp_recall)
    if ps_recall < db_recall and ps_recall < dbpp_recall:
        constraint = 0
    else:
        constraint = 1
    print("pse: {},db: {}, dbpp: {}".format(ps_recall,db_recall,dbpp_recall))
    nni.report_final_result(float(ps_recall+db_recall+dbpp_recall)/3.0 + constraint)





if __name__ == '__main__':
    from tools.parser import ConfigParser
    from attack.attacker_text import UniversalAttacker
    import warnings

    warnings.filterwarnings('ignore')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = ConfigParser("./configs/parallel.yaml")
    detector_attacker = UniversalAttacker(cfg, device)
    detector_attacker.weight = init_detector_weight(cfg.DETECTOR)
    data_root = cfg.DATA.TRAIN.IMG_DIR
    search(cfg, data_root, detector_attacker)
