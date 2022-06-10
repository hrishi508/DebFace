import configparser 

class ConfigParams():
    def __init__(self, cfg):
        self.seed = int(cfg["seed"])
        self.output = cfg["output"]
        self.rec = cfg["rec"]
        self.batch_size = int(cfg["batch_size"])
        self.network = cfg["network"]
        self.embedding_size = int(cfg["embedding_size"])
        self.optimizer = cfg["optimizer"]
        self.n_gender_classes = int(cfg["n_gender_classes"])
        self.n_race_classes = int(cfg["n_race_classes"])
        self.n_age_classes = int(cfg["n_age_classes"])
        self.n_id_classes = int(cfg["n_id_classes"])
        self.n_distr_classes = int(cfg["n_distr_classes"])
        self.sample_rate = int(cfg["sample_rate"])
        self.num_image = int(cfg["num_image"])
        self.warmup_step = int(cfg["warmup_step"])
        self.total_step = int(cfg["total_step"])
        self.num_epoch = int(cfg["num_epoch"])
        self.lr = float(cfg["lr"])
        self.resume = bool(cfg["resume"])
        self.val_targets = cfg["val_targets"]
        self.frequent = int(cfg["frequent"])
        self.verbose = int(cfg["verbose"])
        self.save_all_states = bool(cfg["save_all_states"])
        self.device = cfg["device"]
        self.path = cfg["path"]

def get_config(config_file):
    config_obj = configparser.ConfigParser(inline_comment_prefixes="#")
    config_obj.read(config_file)
    args = config_obj["args"]
    return args

if __name__ == "__main__":
    args = get_config("/home/hrishi/Repos/DebFace/config.ini")

    print(args["optimizer"], type(args["optimizer"]))
    print(args["resume"], type(args["resume"]))
    print(args["embedding_size"], type(args["embedding_size"]))
    print(args["n_id_classes"], type(args["n_id_classes"]))
    print(args["batch_size"], type(args["batch_size"]), type(int(args["batch_size"])))

    cfg = ConfigParams(args)
    
    print("===================== \nType converted params \n=====================")

    print(cfg.optimizer, type(cfg.optimizer))
    print(cfg.resume, type(cfg.resume))
    print(cfg.embedding_size, type(cfg.embedding_size))
    print(cfg.n_id_classes, type(cfg.n_id_classes))
    print(cfg.batch_size, type(cfg.batch_size))
