from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MISA_TA import MISA_TA
from .FusionNets.MISA_T_A import MISA_T_A
from .FusionNets.MISA_T import MISA_T
from .FusionNets.MISA_A import MISA_A
text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'misa_ta':MISA_TA,
    'misa_t_a':MISA_T_A,
    'misa_a':MISA_A,
    'misa_t':MISA_T,
    }