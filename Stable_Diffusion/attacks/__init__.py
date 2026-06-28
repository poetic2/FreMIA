from . import clid, naive, pia, sec


ATTACKS = {
    "clid": clid.run,
    "naive": naive.run,
    "pia": pia.run,
    "sec": sec.run,
}

CLID_POSTPROCESS = {
    "get_l_clidavg_last3": clid.get_l_clidavg_last3,
    "deal_data_weight_avg": clid.deal_data_weight_avg,
}
