import logging

class Logger(object):
    def __init__(self, txt_path):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.WARNING)
        self.txt_path = txt_path
        self.logger = logging.getLogger('train')
        self.formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        self.logger.setLevel(logging.INFO)

    def __console(self, level, message):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        file_handler = logging.FileHandler(self.txt_path, mode='a')
        console_handler = logging.StreamHandler()

        file_handler.setFormatter(self.formatter)
        console_handler.setFormatter(self.formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)

        self.logger.removeHandler(file_handler)
        self.logger.removeHandler(console_handler)

        file_handler.close()

    def debug(self, message):
        self.__console('debug', message)

    def info(self, message):
        self.__console('info', message)

    def warning(self, message):
        self.__console('warning', message)

    def error(self, message):
        self.__console('error', message)

def log_metrics(metrics, logger, tensorboard_logger, epoch):
    def log_single_class(data, tag):
        logger.info(
            '{:>15} \t\tI-Auroc:{:.2f} \tI-F1:{:.2f} \tI-AP:{:.2f} \tP-Auroc:{:.2f} \tP-F1:{:.2f} \tP-AP:{:.2f}'.
            format(tag,
                   data['auroc_im'],
                   data['f1_im'],
                   data['ap_im'],
                   data['auroc_px'],
                   data['f1_px'],
                   data['ap_px'])
        )
        # Adding scalar metrics to TensorBoard
        for metric_name in ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']:
            tensorboard_logger.add_scalar(f'{tag}-{metric_name}', data[metric_name], epoch)

    for tag, data in metrics.items():
        log_single_class(data, tag)



