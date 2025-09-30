from .MISA_A.manager import MISA_A
from .MISA_T.manager import MISA_T
from .MISA_TA.manager import MISA_TA
from .MISA_T_A.manager import MISA_T_A

method_map = {
    'misa_t':MISA_T,
    'misa_a': MISA_A,
    'misa_ta': MISA_TA,
    'misa_t_a': MISA_T_A
}
