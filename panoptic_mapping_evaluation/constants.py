from phonenumbers import format_out_of_country_keeping_alpha_chars
from pkg_resources import fixup_namespace_packages


PRQ_KEY = "PRQ"
PRQ_THING_KEY = "PRQ_th"
PRQ_STUFF_KEY = "PRQ_st"
SRQ_KEY = "SRQ"
RRQ_KEY = "RRQ"
TP_KEY = "TP"
FP_KEY = "FP"
FN_KEY = "FN"
MIOU_KEY = "mIoU"

TP_IOU_THRESHOLD = 0.25

PRQ_SRQ_RRQ_KEYS = [
    PRQ_KEY,
    PRQ_STUFF_KEY,
    PRQ_THING_KEY,
    SRQ_KEY,
    RRQ_KEY,
]

TP_FP_FN_KEYS = [
    TP_KEY,
    FP_KEY,
    FN_KEY,
]

