# iANRCRSettings - настройки
from dataclasses import dataclass

iANRCRDetectModelPath = "model/model01.pb"


@dataclass
class iANRCRConfig:
    detect_conf_thresh: float = 0.25
    detect_max_output_size: int = 100
    detect_iou_threshold: float = 0.45
    detect_width: int = 1024
    detect_height: int = 1024
    types_of_object_detection: int = 10
    max_distance_between_charactersW: float = 2.0
    max_distance_between_charactersH: float = 0.8
    min_symbols_in_number: int = 8
    correct_number: bool = True
    memory_number_frames: int = 4
    memory_number_repeat: int = 2

# max_distance_between_charactersW и max_distance_between_charactersH считаются в высотах символов