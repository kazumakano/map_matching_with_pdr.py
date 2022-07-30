import os.path as path
from datetime import datetime, timedelta
from typing import Any, Union
import numpy as np
import torch
import deep_pdr.script.parameter as dpdr_param
import deep_pdr.script.utility as dpdr_util
import particle_filter.script.parameter as pf_param
import particle_filter.script.utility as pf_util
from deep_pdr.script.log import Log as DpdrLog
from deep_pdr.script.speed_model import DualCNNLSTM
from deep_pdr.script.speed_predictor import SpeedPredictor
from map_matching.script.map import Map
from particle_filter.script.log import Log as PfLog
from particle_filter.script.resample import resample
from particle_filter.script.truth import Truth
from particle_filter_with_pdr.script.window import Window
from script.particle import Particle


def _set_main_params(conf: dict[str, Any]) -> None:
    global BEGIN, END, INERTIAL_LOG_FILE, RSSI_LOG_FILE, SPEED_MODEL_STATE_FILE, INIT_DIRECT, INIT_DIRECT_SD, INIT_POS, INIT_POS_SD, LOST_TJ_POLICY, PARTICLE_NUM, RESULT_DIR_NAME

    BEGIN = datetime.strptime(conf["begin"], "%Y-%m-%d %H:%M:%S")
    END = datetime.strptime(conf["end"], "%Y-%m-%d %H:%M:%S")
    INERTIAL_LOG_FILE = str(conf["inertial_log_file"])
    RSSI_LOG_FILE = str(conf["rssi_log_file"])
    SPEED_MODEL_STATE_FILE = str(conf["speed_model_state_file"])
    INIT_DIRECT = np.float16(conf["init_direct"])
    INIT_DIRECT_SD = np.float16(conf["init_direct_sd"])
    INIT_POS = np.array(conf["init_pos"], dtype=np.float16)
    INIT_POS_SD = np.float16(conf["init_pos_sd"])
    LOST_TJ_POLICY = np.int8(conf["lost_tj_policy"])
    PARTICLE_NUM = np.int16(conf["particle_num"])
    RESULT_DIR_NAME = None if conf["result_dir_name"] is None else str(conf["result_dir_name"])

def map_matching_with_pdr(conf: dict[str, Any], gpu_id: Union[int, None], enable_show: bool = True) -> None:
    device = dpdr_util.get_device(gpu_id)
    print(f"main.py: device is {device}")
    
    inertial_log = DpdrLog(BEGIN, END, path.join(dpdr_param.ROOT_DIR, "log/", INERTIAL_LOG_FILE))
    pdr_result_dir = dpdr_util.make_result_dir(RESULT_DIR_NAME)

    speed_model = DualCNNLSTM(16, 12, 64, 1)
    speed_model.load_state_dict(torch.load(path.join(dpdr_param.ROOT_DIR, "model/", SPEED_MODEL_STATE_FILE)))
    speedor = SpeedPredictor(device, inertial_log, speed_model, pdr_result_dir)
    speedor.export()
    
    rssi_log = PfLog(BEGIN, END, path.join(pf_param.ROOT_DIR, "log/observed/", RSSI_LOG_FILE))
    pf_result_dir = pf_util.make_result_dir(None if pdr_result_dir is None else path.basename(pdr_result_dir))
    map = Map(rssi_log.mac_list, pf_result_dir)
    if pf_param.TRUTH_LOG_FILE is not None:
        truth = Truth(BEGIN, END, pf_result_dir)

    if pf_param.ENABLE_DRAW_BEACONS:
        map.draw_beacons(True)
    if pf_param.ENABLE_SAVE_VIDEO:
        map.init_recorder()

    particles = np.empty(PARTICLE_NUM, dtype=Particle)
    poses = np.empty((PARTICLE_NUM, 2), dtype=np.float16)    # positions
    directs = np.empty(PARTICLE_NUM, dtype=np.float16)       # directions
    for i in range(PARTICLE_NUM):
        poses[i] = np.random.normal(loc=INIT_POS, scale=INIT_POS_SD, size=2).astype(np.float16)
        directs[i] = np.float16(np.random.normal(loc=INIT_DIRECT, scale=INIT_DIRECT_SD) % 360)

    estim_pos = np.array(INIT_POS, dtype=np.float16)
    lost_ts_buf = np.empty(0, dtype=datetime)
    t = BEGIN
    while t <= END:
        print(f"main.py: {t.time()}")
        win = Window(t, rssi_log, map.resolution, speedor.speed, speedor.ts)

        for i in range(PARTICLE_NUM):
            particles[i] = Particle(map, poses[i], directs[i], estim_pos)
            if win.particle_stride is None:
                particles[i].random_walk()
            else:
                particles[i].walk(None, win.particle_stride)
            particles[i].set_likelihood(estim_pos, map, win.strength_weight_list, win.subject_dist_list)

        poses, directs = resample(particles)

        if LOST_TJ_POLICY == 1:
            if not pf_param.IS_LOST:
                estim_pos = pf_util.estim_pos(particles)
                map.draw_particles(estim_pos, particles)
            if pf_param.TRUTH_LOG_FILE is not None:
                map.draw_truth(truth.update_err(t, estim_pos, map.resolution, pf_param.IS_LOST), True)

        elif LOST_TJ_POLICY == 2:
            if pf_param.TRUTH_LOG_FILE is not None and pf_param.IS_LOST:
                last_estim_pos = estim_pos
                lost_ts_buf = np.hstack((lost_ts_buf, t))
            elif not pf_param.IS_LOST:
                estim_pos = pf_util.estim_pos(particles)
                map.draw_particles(estim_pos, particles)

                if pf_param.TRUTH_LOG_FILE is not None:
                    buf_len = len(lost_ts_buf)
                    for i, lt in enumerate(lost_ts_buf):
                        lerped_pos = pf_util.get_lerped_pos(estim_pos, last_estim_pos, i, buf_len)
                        map.draw_lerped_pos(lerped_pos, True)
                        map.draw_truth(truth.update_err(lt, lerped_pos, map.resolution, True), True)
                    lost_ts_buf = np.empty(0, dtype=datetime)
                    map.draw_truth(truth.update_err(t, estim_pos, map.resolution, False), True)

        if pf_param.ENABLE_SAVE_VIDEO:
            map.record()
        if enable_show:
            map.show()

        t += timedelta(seconds=pf_param.WIN_STRIDE)

    print("main.py: reached end of log")
    if pf_param.ENABLE_SAVE_IMG:
        map.save_img()
    if pf_param.ENABLE_SAVE_VIDEO:
        map.save_video()
    if pf_param.ENABLE_WRITE_CONF:
        pf_util.write_conf(conf, pf_result_dir)
    if pf_param.TRUTH_LOG_FILE is not None:
        truth.export_err()
    if enable_show:
        map.show(0)

if __name__ == "__main__":
    import argparse
    from map_matching.script.parameter import set_params as set_mm_params
    from particle_filter_with_pdr.script.parameter import set_params as set_pfpdr_params

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf_file", help="specify config file", metavar="PATH_TO_CONF_FILE")
    parser.add_argument("-g", "--gpu_id", type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("--no_display", action="store_true", help="run without display")
    args = parser.parse_args()

    set_mm_params(args.conf_file)
    conf = set_pfpdr_params(args.conf_file)
    _set_main_params(conf)

    map_matching_with_pdr(conf, args.gpu_id, not args.no_display)
