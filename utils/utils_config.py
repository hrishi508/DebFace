import configparser 

class ConfigParams():
    def __init__(self, cfg):
        self.datasets_base_path = cfg["datasets_base_path"]
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
        self.num_epoch = int(cfg["num_epoch"])
        self.lr = float(cfg["lr"])
        self.momentum = float(cfg["momentum"])
        self.weight_decay = float(cfg["weight_decay"])
        self.lr_decay_rate = float(cfg["lr_decay_rate"])
        self.lr_scheduler = (cfg["lr_scheduler"] == "True")
        self.lr_decay_milestones = [int(x) for x in cfg["lr_decay_milestones"].split(",")]
        self.device = cfg["device"]
        self.model_weights_dir = cfg["model_weights_dir"]
        self.width = int(cfg["width"])
        self.height = int(cfg["height"])
        self.num_img_filter = int(cfg["num_img_filter"])
        self.train_dataset_img_dir = cfg["train_dataset_img_dir"]
        self.train_dataset_labels = cfg["train_dataset_labels"]
        self.test_dataset_img_dir = cfg["test_dataset_img_dir"]
        self.test_dataset_labels = cfg["test_dataset_labels"]
        self.base_path = cfg["base_path"]
        self.val_dataset_size = int(cfg["val_dataset_size"])
        self.save_model_weights_every = int(cfg["save_model_weights_every"])
        self.load_weights = (cfg["load_weights"] == "True")
        self.load_weights_file = cfg["load_weights_file"]
        self.plots_dir = cfg["plots_dir"]
        self.plot_losses = (cfg["plot_losses"] == "True")
        self.trial_number = cfg["trial_number"]

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
