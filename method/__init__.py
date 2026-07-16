from .adaclip import AdaCLIP
from .prompted_features import install_prompted_feature_api

install_prompted_feature_api(AdaCLIP)

from .trainer import AdaCLIP_Trainer