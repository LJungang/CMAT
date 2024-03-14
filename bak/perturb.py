

def grad_perturb(detector, cfg, now_step, index):
    if cfg.DETECTOR.PERTURB.GATE == 'grad_descend':
        try:
            freq = cfg.DETECTOR.PERTURB.GRAD_DESCEND.PERTURB_FREQ
        except Exception as e:
            print(e, 'From grad perturb: Resetting model reset freq...')
            freq = 1

        print('GRAD_PERTURB: every(', freq, ' step)')
        if now_step and now_step % cfg.DETECTOR.RESET_FREQ == 0:
            print(now_step, ' : resetting the model')
            detector.reset_model()
        elif index % freq == 0:
            print(now_step, ': perturbing')
            detector.perturb()