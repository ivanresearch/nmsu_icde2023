
class DataStru:
    def __init__(self) -> None:
        self.data_key = "ges"
        self.num_classes = 18
        self.attr_num = 5
        self.attr_len = 214
        self.data_folder = ""
        self.class_column = 0
        self.min_class = 0
        self.data_folder = ""

    def to_string(self):
        ret_str = "data key:" + str(self.data_key) + 'num of classes: ' + str(
            self.num_classes) + '\nattribute number: ' + str(
                self.attr_num) + '\nattribute length: ' + str(
                    self.attr_len) + '\nclass column: ' + str(
                        self.class_column) + '\nmin_class: ' + str(
                            self.min_class)
        return ret_str


class ModelSetting_v1:
    def __init__(self, method='') -> None:
        self.method = method
        self.learning_rate = 1e-3
        self.dropout = 0.5
        self.kernel_list = []
        self.log_folder = ""
        self.obj_folder = ""
        self.out_obj_folder = ""
        self.out_model_folder = ""
        self.cnn_setting_file = ""

    def to_string(self):
        ret_str = "method: " + str(self.method) + "\nlearning rate: " + str(
            self.learning_rate) + "\ndropout: " + str(
                self.dropout) + "\nlog_foler: " + str(
                    self.log_folder) + "\nobj_folder: " + str(self.obj_folder)
        return ret_str


class model_parameter_class:
    model_key = ""
    data_stru = None
    log_folder = ""
    obj_folder = ""
    out_obj_folder = ""
    out_model_folder = ""