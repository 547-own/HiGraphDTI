import logging
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

def setup_logging(args):

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    # timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    # set file
    fh = logging.FileHandler(args.logger_file_path)
    fh.setLevel(logging.DEBUG)
    # set stream
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # set format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def build_optimizer_scheduler(args, model, num_total_steps):
    optimizer = AdamW(model.parameters(), lr =args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_ratio * num_total_steps, num_training_steps = num_total_steps)
    return optimizer, scheduler

